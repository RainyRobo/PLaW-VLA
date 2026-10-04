from collections.abc import Callable, Mapping, Sequence
import dataclasses
import re
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import numpy as np
from openpi_client import image_tools

from openpi.models import tokenizer as _tokenizer
from openpi.policies import ee_pose_utils
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats


T = TypeVar("T")
S = TypeVar("S")


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)

        def repack(node):
            if isinstance(node, Mapping):
                result = {}
                for new_key, child in node.items():
                    result[new_key] = repack(child)
                    if isinstance(child, str):
                        pad_key = f"{child}_is_pad"
                        if pad_key in flat_item:
                            result[f"{new_key}_is_pad"] = flat_item[pad_key]
                return result
            if not isinstance(node, str):
                raise TypeError(f"Unsupported RepackTransform leaf type: {type(node)!r}")
            return flat_item[node]

        return repack(self.structure)


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        return apply_tree(
            data, #tree
            self.norm_stats, #selector
            self._normalize_quantile if self.use_quantiles else self._normalize, #fn
            strict=self.strict, 
        )

    def _normalize(self, x, stats: NormStats):
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # Make sure that all the keys in the norm stats are present in the data.
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    def _unnormalize(self, x, stats: NormStats):
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01, stats.q99
        if (dim := q01.shape[-1]) < x.shape[-1]:
            return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        resized_images = {}
        for k, v in data["image"].items():
            if v.ndim == 3: # [H,W,C]
                resized_images[k] = image_tools.resize_with_pad(v, self.height, self.width)
            elif v.ndim == 4: # [T,H,W,C]
                resized_images[k] = image_tools.resize_with_pad(v, 256, 256)
            else:
                raise ValueError(f"Unsupported image shape {v.shape} for key {k}")
        data["image"] = resized_images
        return data


@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    stride: int

    def __call__(self, data: DataDict) -> DataDict:
        data["actions"] = data["actions"][:: self.stride]
        return data


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None = None
    # If true, treat the selected prefix as canonical ee pose blocks
    # [x, y, z, qw, qx, qy, qz, gripper] and use quaternion-aware deltas.
    ee_pose: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data

        state, actions = data["state"], data["actions"]
        if self.ee_pose:
            dims = _resolve_prefix_dims(state, actions, self.mask)
            if dims == 0:
                return data
            actions[..., :dims] = _canonical_ee_delta_from_absolute(state[..., :dims], actions[..., :dims])
            data["actions"] = actions
            return data
        if self.mask is None:
            return data
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None = None
    # If true, treat the selected prefix as canonical ee pose blocks
    # [x, y, z, qw, qx, qy, qz, gripper] and use quaternion-aware composition.
    ee_pose: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data

        state, actions = data["state"], data["actions"]
        if self.ee_pose:
            dims = _resolve_prefix_dims(state, actions, self.mask)
            if dims == 0:
                return data
            actions[..., :dims] = _canonical_ee_absolute_from_delta(state[..., :dims], actions[..., :dims])
            data["actions"] = actions
            return data
        if self.mask is None:
            return data
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


def _expand_state_for_actions(state: np.ndarray, actions: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != state.ndim + 1:
        raise ValueError(f"Expected actions rank to be state rank + 1, got {actions.shape=} and {state.shape=}.")
    return np.expand_dims(state, axis=-2)


def _resolve_prefix_dims(state: np.ndarray, actions: np.ndarray, mask: Sequence[bool] | None) -> int:
    if mask is None:
        return min(np.asarray(state).shape[-1], np.asarray(actions).shape[-1])
    prefix_mask = np.asarray(mask, dtype=bool).reshape(-1)
    false_indices = np.flatnonzero(~prefix_mask)
    if false_indices.size == 0:
        return int(prefix_mask.size)
    prefix_dim = int(false_indices[0])
    if np.any(prefix_mask[prefix_dim:]):
        raise ValueError("Expected action mask to define a contiguous enabled prefix.")
    return prefix_dim


def _canonical_ee_arm_slices(vector_dim: int) -> tuple[slice, ...]:
    if vector_dim <= 0 or vector_dim % 8 != 0:
        raise ValueError(f"Canonical EE vectors must have a positive multiple-of-8 dimension, got {vector_dim}.")
    return tuple(slice(offset, offset + 8) for offset in range(0, vector_dim, 8))


def _canonical_ee_delta_from_absolute(state: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Express every chunk step relative to the current state.

    Position and gripper are subtracted from that state. Orientation is the
    quaternion rotation from the current orientation. Later steps do not
    subtract the previous absolute target.
    """
    state_expanded = _expand_state_for_actions(state, actions)
    delta = np.asarray(actions, dtype=np.float32).copy()
    for arm_slice in _canonical_ee_arm_slices(state_expanded.shape[-1]):
        current_arm = state_expanded[..., arm_slice]
        target_arm = delta[..., arm_slice]
        target_quat = ee_pose_utils.align_quaternion_sign(current_arm[..., 3:7], target_arm[..., 3:7])
        target_arm[..., :3] = target_arm[..., :3] - current_arm[..., :3]
        target_arm[..., 3:7] = ee_pose_utils.quaternion_delta(current_arm[..., 3:7], target_quat)
        target_arm[..., 7:8] = target_arm[..., 7:8] - current_arm[..., 7:8]
        delta[..., arm_slice] = target_arm
    return delta.astype(np.float32)


def _canonical_ee_absolute_from_delta(state: np.ndarray, actions: np.ndarray) -> np.ndarray:
    state_expanded = _expand_state_for_actions(state, actions)
    absolute = np.asarray(actions, dtype=np.float32).copy()
    for arm_slice in _canonical_ee_arm_slices(state_expanded.shape[-1]):
        current_arm = state_expanded[..., arm_slice]
        delta_arm = absolute[..., arm_slice]
        delta_quat = ee_pose_utils.canonicalize_quaternion_sign(delta_arm[..., 3:7])
        delta_arm[..., :3] = current_arm[..., :3] + delta_arm[..., :3]
        delta_arm[..., 3:7] = ee_pose_utils.quaternion_apply_delta(current_arm[..., 3:7], delta_quat)
        delta_arm[..., 7:8] = current_arm[..., 7:8] + delta_arm[..., 7:8]
        absolute[..., arm_slice] = delta_arm
    return absolute.astype(np.float32)


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(prompt, state)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}


@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputs(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        state, actions = data["state"], data.get("actions")
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(prompt, state, actions)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer
    action_horizon: int
    action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        # Model outputs are saved in "actions", but for FAST models they represent tokens.
        tokens = data.pop("actions")
        actions = self.tokenizer.extract_actions(tokens.astype(np.int32), self.action_horizon, self.action_dim)
        return {
            **data,
            "actions": actions,
        }


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task."""

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    # Recent LeRobot metadata exposes this as a pandas.DataFrame with index=task and
    # a `task_index` column, while older paths may still provide a direct mapping.
    tasks: object

    def __post_init__(self) -> None:
        object.__setattr__(self, "tasks", _normalize_lerobot_task_mapping(self.tasks))

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")

        return {**data, "prompt": prompt}


def _normalize_lerobot_task_mapping(tasks: object) -> dict[int, str]:
    if isinstance(tasks, Mapping):
        normalized = _normalize_lerobot_task_mapping_from_mapping(tasks)
        if normalized:
            return normalized

    # LeRobot v3 metadata commonly returns a pandas.DataFrame with task text in the index
    # and a `task_index` column.
    if hasattr(tasks, "iterrows") and hasattr(tasks, "columns") and hasattr(tasks, "index"):
        columns = getattr(tasks, "columns")
        if "task_index" in columns:
            normalized = {}
            for task, row in tasks.iterrows():
                normalized[int(row["task_index"])] = str(task)
            if normalized:
                return normalized

    if hasattr(tasks, "to_dict"):
        try:
            as_dict = tasks.to_dict()
        except Exception:
            as_dict = None
        if isinstance(as_dict, Mapping):
            normalized = _normalize_lerobot_task_mapping_from_mapping(as_dict)
            if normalized:
                return normalized

    raise TypeError(f"Unsupported LeRobot task metadata type: {type(tasks)!r}")


def _normalize_lerobot_task_mapping_from_mapping(tasks: Mapping) -> dict[int, str]:
    if not tasks:
        return {}

    if "task_index" in tasks and isinstance(tasks["task_index"], Mapping):
        return {int(task_index): str(task) for task, task_index in tasks["task_index"].items()}

    first_key, first_value = next(iter(tasks.items()))
    if isinstance(first_key, (int, np.integer)):
        return {int(task_index): _stringify_prompt(prompt) for task_index, prompt in tasks.items()}
    if isinstance(first_value, (int, np.integer)):
        return {int(task_index): _stringify_prompt(prompt) for prompt, task_index in tasks.items()}

    return {}


def _stringify_prompt(prompt: object) -> str:
    if isinstance(prompt, bytes):
        return prompt.decode("utf-8")
    return str(prompt)


@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    """Zero-pads states and actions to the model action dimension."""

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        return data


def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")


def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
    """Transform the structure of a nested dictionary using a set of patterns.

    The transformation is defined using the `patterns` dictionary. The keys are the
    input keys that should be matched and the values are the new names inside the output
    dictionary. If the value is None, the input key is removed.

    Both keys and values should represent flattened paths using '/' as the separator.
    Keys can be regular expressions and values can include backreferences to the
    matched groups (see `re.sub` for more details). Note that the regular expression
    must match the entire key.

    The order inside the `patterns` dictionary is important. Only the first pattern that
    matches the input key will be used.

    See unit tests for more examples.

    Args:
        patterns: A mapping from old keys to new keys.
        tree: The nested dictionary to transform.

    Returns:
        The transformed nested dictionary.
    """
    data = flatten_dict(tree)

    # Compile the patterns.
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None
                break
        else:
            # Use the original key if no match is found.
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # Validate the output structure to make sure that it can be unflattened.
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i : i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k]) # normalize
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


def split_temporal_valid_mask(
    data: Mapping[str, object],
    key: str,
    *,
    history_len: int,
    future_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve per-frame validity masks for SplitTemporalFrames outputs.

    LeRobot emits `{key}_is_pad` over the full temporal window. This helper converts it
    into validity masks aligned with `{key}_history` and `{key}_future`.
    """
    history_mask = np.ones((history_len,), dtype=bool)
    future_mask = np.ones((future_len,), dtype=bool)

    pad_key = f"{key}_is_pad"
    if pad_key not in data:
        return history_mask, future_mask

    is_pad = np.asarray(data[pad_key], dtype=bool).reshape(-1)
    full_len = history_len + future_len
    if is_pad.shape[0] != full_len:
        raise ValueError(
            f"Expected {pad_key!r} to contain {full_len} entries "
            f"({history_len} history + {future_len} future), got {is_pad.shape[0]}."
        )

    history_mask = ~is_pad[:history_len]
    future_mask = ~is_pad[history_len:]
    return history_mask, future_mask


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )


@dataclasses.dataclass(frozen=True)
class SplitTemporalFrames(DataTransformFn):
    """Split temporal image sequences into current, history, and future frames."""

    frame_indices: Sequence[int]
    image_keys: Sequence[str]
    
    def __post_init__(self):
        frame_indices = tuple(self.frame_indices)
        if not frame_indices:
            raise ValueError("frame_indices must not be empty")
        if 0 not in frame_indices:
            raise ValueError(f"frame_indices must include 0 for the current frame, got {frame_indices}")
        if tuple(sorted(frame_indices)) != frame_indices or len(set(frame_indices)) != len(frame_indices):
            raise ValueError(f"frame_indices must be strictly increasing, got {frame_indices}")

        current_idx = frame_indices.index(0)
        object.__setattr__(self, "_frame_indices", frame_indices)
        object.__setattr__(self, "_current_idx", current_idx)
        object.__setattr__(self, "_history_len", current_idx + 1)
        object.__setattr__(self, "_full_sequence_len", len(frame_indices))
    
    def __call__(self, data: DataDict) -> DataDict:
        result = dict(data)
        
        for key in self.image_keys:
            if key in data:
                image = data[key]
                if hasattr(image, 'ndim') and image.ndim == 4:
                    if image.shape[0] == self._full_sequence_len:
                        current_frame = image[self._current_idx]
                        history_frames = image[: self._history_len]
                        future_frames = image[self._history_len :]
                    elif 0 < image.shape[0] <= self._history_len:
                        # Inference clients often send history-only stacks. Treat the last frame as
                        # current and all preceding frames as history.
                        current_frame = image[-1]
                        history_frames = image
                        future_frames = image[:0]
                    else:
                        raise ValueError(
                            f"Expected {self._full_sequence_len} frames for a full temporal window or at most "
                            f"{self._history_len} frames for history-only inference, but got shape {image.shape} "
                            f"for key '{key}'."
                        )
                    base_key = key
                    result[f"{base_key}_current"] = current_frame
                    result[f"{base_key}_history"] = history_frames
                    result[f"{base_key}_future"] = future_frames
                else:
                    raise ValueError(f"Expected 4D array for key '{key}', but got shape {image.shape}")
        return result
