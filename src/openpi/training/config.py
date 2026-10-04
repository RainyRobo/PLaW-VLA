"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import json
import logging
import pathlib
from typing import Any, ClassVar, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
import numpy as np
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.tokenizer as _tokenizer
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter

_LOCAL_REPO_CHILDREN_MANIFEST = ".openpi_child_datasets_manifest.json"
_LOCAL_REPO_CHILDREN_MANIFEST_VERSION = 1


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Location of dataset assets such as normalization statistics.

    Assets are copied into each checkpoint under ``assets/<asset_id>``.
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms. Prefer a relative id such as "my_robot". Absolute paths are stored under
    # assets/<config>/ with the leading slash removed.
    asset_id: str | Sequence[str] | None = None

@dataclasses.dataclass(frozen=True)
class WorldModelDataConfig:
    """Temporal world-model settings.

    `history_num_frames` and `future_num_frames` are counts, not span lengths.
    History includes the current frame at offset `0`.

    Example:
    - `history_num_frames=3`, `frame_stride=5` gives history offsets `(-10, -5, 0)`.
      This is 3 history frames spanning 10 steps into the past.
    - `future_num_frames=2`, `frame_stride=5` gives future offsets `(5, 10)`.

    To train with a sampled range like `[1, 6]`, set:
    - `history_num_frames=6`
    - `train_min_history_num_frames=1`

    Then each batch draws a history length from `1..6`. The same pattern
    applies to `future_num_frames` and `train_min_future_num_frames`.
    """

    # Pre-repack image keys matching the dataset (used by data_loader for delta_timestamps).
    # Activation is controlled by `model.enable_world_model`.
    image_keys: Sequence[str] | None = None

    # Explicit temporal configuration for the legacy count/stride mode.
    # In the explicit time-offset mode below, history/future counts are derived
    # from `time_offsets_s`, and `frame_stride` is unused.
    history_num_frames: int = 3
    future_num_frames: int = 2
    frame_stride: int | None = 5
    # Optional explicit temporal offsets in seconds. When set, these replace the
    # implicit `frame_stride / fps` sampling for image queries and keep the same
    # real-time window across datasets with different FPS.
    time_offsets_s: Sequence[float] | None = None

    # Optional per-batch training-time sampling ranges. When set below the resolved maxima,
    # the loader samples a shorter history/future length for the whole batch.
    train_min_history_num_frames: int | None = None
    train_min_future_num_frames: int | None = None
    # Optional non-uniform sampling powers used when drawing variable temporal lengths.
    # `0.0` keeps the original uniform distribution; `1.0` samples with weights proportional
    # to `k`; larger values bias more strongly toward longer contexts.
    train_history_sample_power: float = 1.0
    train_future_sample_power: float = 1.0
    # Variable history sampling always keeps the most recent `k` history frames (suffix).
    # Variable future sampling always keeps the nearest `k` future horizons (prefix).

    def __post_init__(self) -> None:
        if self.train_min_history_num_frames is not None and self.train_min_history_num_frames <= 0:
            raise ValueError(
                f"train_min_history_num_frames must be > 0, got {self.train_min_history_num_frames}"
            )
        if self.train_min_future_num_frames is not None and self.train_min_future_num_frames <= 0:
            raise ValueError(
                f"train_min_future_num_frames must be > 0, got {self.train_min_future_num_frames}"
            )
        if self.train_history_sample_power < 0:
            raise ValueError(
                f"train_history_sample_power must be >= 0, got {self.train_history_sample_power}"
            )
        if self.train_future_sample_power < 0:
            raise ValueError(
                f"train_future_sample_power must be >= 0, got {self.train_future_sample_power}"
            )

        if self.time_offsets_s is not None:
            time_offsets = tuple(float(offset) for offset in self.time_offsets_s)
            if not time_offsets:
                raise ValueError("time_offsets_s must not be empty.")
            if 0.0 not in time_offsets:
                raise ValueError(f"time_offsets_s must include 0.0 for the current frame, got {time_offsets}.")
            if any(next_offset <= offset for offset, next_offset in zip(time_offsets, time_offsets[1:])):
                raise ValueError(f"time_offsets_s must be strictly increasing, got {time_offsets}.")

            current_idx = time_offsets.index(0.0)
            resolved_history = current_idx + 1
            resolved_future = len(time_offsets) - current_idx - 1
            object.__setattr__(self, "history_num_frames", resolved_history)
            object.__setattr__(self, "future_num_frames", resolved_future)
        else:
            if self.frame_stride is None or self.frame_stride <= 0:
                raise ValueError(f"frame_stride must be > 0, got {self.frame_stride}")
            if self.history_num_frames <= 0:
                raise ValueError(f"history_num_frames must be > 0, got {self.history_num_frames}")
            if self.future_num_frames < 0:
                raise ValueError(f"future_num_frames must be >= 0, got {self.future_num_frames}")

        if self.train_min_history_num_frames is not None and self.train_min_history_num_frames > self.history_num_frames:
            raise ValueError(
                "train_min_history_num_frames must be <= the resolved history length "
                f"({self.history_num_frames}), got {self.train_min_history_num_frames}."
            )
        if self.train_min_future_num_frames is not None and self.train_min_future_num_frames > self.future_num_frames:
            raise ValueError(
                "train_min_future_num_frames must be <= the resolved future length "
                f"({self.future_num_frames}), got {self.train_min_future_num_frames}."
            )
    def resolve_history_num_frames(self) -> int:
        return self.history_num_frames

    def resolve_future_num_frames(self) -> int:
        return self.future_num_frames

    def resolve_frame_indices(self) -> tuple[int, ...]:
        if self.time_offsets_s is not None:
            return self.resolve_layout_indices()
        if self.frame_stride is None:
            raise ValueError("frame_stride must be set when using count/stride world-model sampling.")
        history_offsets = tuple(range(-(self.history_num_frames - 1) * self.frame_stride, 1, self.frame_stride))
        future_offsets = tuple(range(self.frame_stride, (self.future_num_frames + 1) * self.frame_stride, self.frame_stride))
        return history_offsets + future_offsets

    def resolve_layout_indices(self) -> tuple[int, ...]:
        """Resolve placeholder indices used only for temporal layout/splitting."""
        if self.time_offsets_s is None:
            return self.resolve_frame_indices()
        return tuple(range(-(self.history_num_frames - 1), self.future_num_frames + 1))

    def resolve_time_offsets(self, fps: float) -> tuple[float, ...]:
        if self.time_offsets_s is not None:
            return tuple(float(offset) for offset in self.time_offsets_s)
        return tuple(index / fps for index in self.resolve_frame_indices())


ActionSemantics: TypeAlias = Literal[
    "joint_position",
    "joint_effector_position",
    "ee_pose",
    "ee_pose_command",
    "hand_pose",
]


@dataclasses.dataclass(frozen=True)
class ActionSpaceDescriptor:
    """Audit metadata describing how a dataset's state/actions align for training."""

    state_semantics: ActionSemantics
    action_semantics: ActionSemantics
    state_dim: int
    action_dim: int
    canonical_space_id: str
    supports_delta_from_state: bool = False
    state_action_alignment_mask: tuple[bool, ...] | None = None
    audit_note: str = ""

    def validate_for_stage(self, *, dataset_type: str, training_stage: str, has_actions: bool) -> None:
        if training_stage == "post_training" and not has_actions:
            raise ValueError(
                f"Dataset type {dataset_type!r} cannot be used for post_training because actions are unavailable."
            )

    def validate_for_shared_delta(self, *, dataset_type: str) -> tuple[bool, ...]:
        if not self.supports_delta_from_state or self.state_action_alignment_mask is None:
            raise ValueError(
                f"Dataset type {dataset_type!r} does not support shared delta actions in canonical space. "
                f"Audit note: {self.audit_note or 'none'}"
            )
        if self.state_semantics != self.action_semantics:
            raise ValueError(
                f"Dataset type {dataset_type!r} cannot use shared delta actions because state semantics "
                f"{self.state_semantics!r} do not match action semantics {self.action_semantics!r}."
            )
        mask = tuple(bool(value) for value in self.state_action_alignment_mask)
        if not any(mask):
            raise ValueError(f"Dataset type {dataset_type!r} has an empty shared-delta alignment mask.")
        if len(mask) > self.state_dim or len(mask) > self.action_dim:
            raise ValueError(
                f"Dataset type {dataset_type!r} has invalid shared-delta mask length {len(mask)} for "
                f"state_dim={self.state_dim}, action_dim={self.action_dim}."
            )
        return mask


def normalize_asset_id(asset_id: str) -> str:
    """Return a relative asset directory for ``assets/<config>/<asset_id>/``.

    An absolute repo path such as ``/data/my_robot`` becomes ``data/my_robot``.
    A relative id is unchanged. This is the same layout copied into checkpoints.
    """
    path = pathlib.PurePath(asset_id)
    parts = list(path.parts)
    if path.is_absolute() and parts:
        parts = parts[1:]
    clean_parts = [part for part in parts if part not in ("", ".", "..")]
    if not clean_parts:
        raise ValueError(f"Cannot derive an asset directory from asset_id={asset_id!r}.")
    return pathlib.PurePosixPath(*clean_parts).as_posix()


def _is_lerobot_dataset_dir(path: pathlib.Path) -> bool:
    return (path / "meta" / "info.json").is_file()


def _local_repo_children_manifest_path(path: pathlib.Path) -> pathlib.Path:
    return path / _LOCAL_REPO_CHILDREN_MANIFEST


def _scan_local_repo_child_datasets(path: pathlib.Path) -> list[str]:
    resolved_path = path.resolve()
    return [
        child.resolve().as_posix()
        for child in sorted(resolved_path.iterdir(), key=lambda p: p.name)
        if child.is_dir() and _is_lerobot_dataset_dir(child)
    ]


def _load_local_repo_children_manifest(path: pathlib.Path) -> list[str] | None:
    manifest_path = _local_repo_children_manifest_path(path)
    if not manifest_path.is_file():
        return None

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logging.warning("Failed to read child-dataset manifest %s: %s", manifest_path, exc)
        return None

    if payload.get("version") != _LOCAL_REPO_CHILDREN_MANIFEST_VERSION:
        return None

    resolved_path = path.resolve()
    if payload.get("parent_path") != resolved_path.as_posix():
        return None

    parent_stat = resolved_path.stat()
    if payload.get("parent_mtime_ns") != parent_stat.st_mtime_ns:
        return None

    if payload.get("parent_ctime_ns") != parent_stat.st_ctime_ns:
        return None

    children = payload.get("children")
    if not isinstance(children, list) or not all(isinstance(child, str) for child in children):
        return None

    return children


def _write_local_repo_children_manifest(path: pathlib.Path, child_datasets: Sequence[str]) -> pathlib.Path:
    resolved_path = path.resolve()
    payload = {
        "version": _LOCAL_REPO_CHILDREN_MANIFEST_VERSION,
        "parent_path": resolved_path.as_posix(),
        "parent_mtime_ns": resolved_path.stat().st_mtime_ns,
        "parent_ctime_ns": resolved_path.stat().st_ctime_ns,
        "children": list(child_datasets),
    }

    manifest_path = _local_repo_children_manifest_path(resolved_path)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def precache_local_repo_parent_manifests(
    repo_id: str | Sequence[str] | None,
    *,
    refresh: bool = False,
) -> int:
    if repo_id is None:
        return 0

    if isinstance(repo_id, Sequence) and not isinstance(repo_id, str):
        return sum(precache_local_repo_parent_manifests(item, refresh=refresh) for item in repo_id)

    path = pathlib.Path(repo_id).expanduser()
    if not path.is_dir():
        return 0

    resolved_path = path.resolve()
    if _is_lerobot_dataset_dir(resolved_path):
        return 0

    if not refresh and _load_local_repo_children_manifest(resolved_path) is not None:
        return 0

    child_datasets = _scan_local_repo_child_datasets(resolved_path)
    if not child_datasets:
        return 0

    manifest_path = _write_local_repo_children_manifest(resolved_path, child_datasets)
    logging.info(
        "Wrote child-dataset manifest %s with %d child datasets for %s.",
        manifest_path,
        len(child_datasets),
        resolved_path,
    )
    return 1


def _expand_local_repo_ids(
    repo_id: str | Sequence[str] | None,
) -> str | list[str] | None:
    if repo_id is None:
        return None

    if isinstance(repo_id, Sequence) and not isinstance(repo_id, str):
        expanded_repo_ids: list[str] = []
        for item in repo_id:
            expanded = _expand_local_repo_ids(str(item))
            if expanded is None:
                continue
            if isinstance(expanded, list):
                expanded_repo_ids.extend(expanded)
            else:
                expanded_repo_ids.append(expanded)
        return expanded_repo_ids

    path = pathlib.Path(repo_id).expanduser()
    if not path.is_dir():
        return repo_id

    resolved_path = path.resolve()
    if _is_lerobot_dataset_dir(resolved_path):
        return resolved_path.as_posix()

    child_datasets = _load_local_repo_children_manifest(resolved_path)
    if child_datasets is not None:
        logging.info(
            "Expanded local LeRobot parent directory %s into %d child datasets via manifest cache.",
            resolved_path,
            len(child_datasets),
        )
        return child_datasets

    child_datasets = _scan_local_repo_child_datasets(resolved_path)
    if child_datasets:
        logging.info(
            "Expanded local LeRobot parent directory %s into %d child datasets.",
            resolved_path,
            len(child_datasets),
        )
        return child_datasets

    return resolved_path.as_posix()


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | Sequence[str] | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | Sequence[str] | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None
    # Optional per-child normalization stats when `repo_id` expands into multiple child datasets.
    per_repo_norm_stats: dict[str, dict[str, _transforms.NormStats]] | None = None
    # Optional dataset-type tag, primarily used by multi-dataset training configs.
    dataset_type: str | None = None
    # Optional action-space audit descriptor for canonical alignment checks.
    action_space_descriptor: ActionSpaceDescriptor | None = None
    # Versioned description of the pre-normalization representation.  Norm
    # stats with a different or missing contract must not be loaded silently.
    normalization_contract: str | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)

    # World model specific transforms (frame splitting, temporal processing, etc.)
    # Applied after repack but before robot-specific data transforms
    world_model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)

    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)
    # Optional frame offsets, in dataset steps, used when querying future actions for `action_sequence_keys`.
    # If None, the loader falls back to the default contiguous offsets `[0, 1, ..., action_horizon - 1]`.
    action_sequence_offsets: Sequence[int] | None = None
    # Optional uniform action-query spacing in seconds. When set, the loader
    # generates a uniform sequence starting from `action_time_start_s` (or `step`
    # when `action_time_start_s` is omitted).
    action_time_step_s: float | None = None
    # Optional first action-query timestamp in seconds for the uniform
    # `action_time_step_s` mode.
    action_time_start_s: float | None = None
    # Optional explicit action query offsets in seconds. When set, these replace
    # `action_sequence_offsets / fps` and allow cross-dataset time alignment.
    action_sequence_time_offsets_s: Sequence[float] | None = None

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # World model configuration for collecting temporal frames
    world_model: WorldModelDataConfig = dataclasses.field(default_factory=WorldModelDataConfig)

    def __post_init__(self) -> None:
        if self.action_time_step_s is not None and self.action_time_step_s <= 0:
            raise ValueError(f"action_time_step_s must be > 0, got {self.action_time_step_s}")
        if self.action_time_start_s is not None and self.action_time_step_s is None:
            raise ValueError("action_time_start_s requires action_time_step_s to be set.")
        if self.action_time_step_s is not None and self.action_sequence_time_offsets_s is not None:
            raise ValueError("action_time_step_s and action_sequence_time_offsets_s are mutually exclusive.")
        if self.action_time_step_s is not None and self.action_sequence_offsets is not None:
            raise ValueError("action_time_step_s and action_sequence_offsets are mutually exclusive.")

    def resolve_action_time_offsets(self, action_horizon: int, fps: float) -> tuple[float, ...]:
        if self.action_sequence_time_offsets_s is not None:
            action_times = tuple(float(offset) for offset in self.action_sequence_time_offsets_s)
            if len(action_times) != action_horizon:
                raise ValueError(
                    "action_sequence_time_offsets_s length must match action_horizon: "
                    f"expected {action_horizon}, got {len(action_times)}."
                )
            return action_times

        if self.action_time_step_s is not None:
            start = self.action_time_step_s if self.action_time_start_s is None else self.action_time_start_s
            return tuple(start + self.action_time_step_s * step for step in range(action_horizon))

        if self.action_sequence_offsets is None:
            action_offsets = tuple(range(action_horizon))
        else:
            action_offsets = tuple(int(offset) for offset in self.action_sequence_offsets)
            if len(action_offsets) != action_horizon:
                raise ValueError(
                    "action_sequence_offsets length must match action_horizon: "
                    f"expected {action_horizon}, got {len(action_offsets)}."
                )
        return tuple(t / fps for t in action_offsets)


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str | Sequence[str] = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None
    # Whether to resolve norm_stats during config construction.
    load_norm_stats: tyro.conf.Suppress[bool] = True

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = _expand_local_repo_ids(self.repo_id if self.repo_id is not tyro.MISSING else None)
        asset_id = (
            _expand_local_repo_ids(self.assets.asset_id)
            if self.assets.asset_id is not None
            else repo_id
        )
        base = self.base_config or DataConfig()
        norm_stats = None
        per_repo_norm_stats = None
        if self.load_norm_stats:
            norm_stats, per_repo_norm_stats = self._resolve_norm_stats(
                epath.Path(self.assets.assets_dir or assets_dirs), repo_id, asset_id
            )
        return dataclasses.replace(
            base,
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=norm_stats,
            per_repo_norm_stats=per_repo_norm_stats,
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _effective_world_model_enabled(self, model_config: _model.BaseModelConfig) -> bool:
        return bool(getattr(model_config, "enable_world_model", False))

    def _resolve_training_stage(self, model_config: _model.BaseModelConfig) -> Literal["wm_alignment", "post_training"]:
        stage = getattr(model_config, "training_stage", "post_training")
        if stage not in ("wm_alignment", "post_training"):
            raise ValueError(f"Unsupported training stage {stage!r}.")
        return stage

    def _resolve_world_model_config(self, model_config: _model.BaseModelConfig) -> WorldModelDataConfig:
        return (
            self.base_config.world_model if self.base_config else WorldModelDataConfig()
        )
    
    def _create_name_change_map(self, repack_transforms: _transforms.Group) -> dict[str, str]:
        """Derive a mapping from original (pre-repack) keys to repacked keys."""
        name_change_map: dict[str, str] = {}
        for transform in repack_transforms.inputs:
            if isinstance(transform, _transforms.RepackTransform):
                structure = transform.structure
                flat_structure = _transforms.flatten_dict(structure)
                if not name_change_map:
                    name_change_map = {old_key: new_key for new_key, old_key in flat_structure.items()}
                else:
                    composed: dict[str, str] = {}
                    for new_key, old_key in flat_structure.items():
                        initial_key = name_change_map.get(old_key, old_key)
                        composed[initial_key] = new_key
                    name_change_map = composed
        return name_change_map

    def _create_world_model_transforms(
        self,
        wm_enabled: bool,
        wm_config: WorldModelDataConfig | None = None,
        model_config: _model.BaseModelConfig | None = None,
        image_keys: Sequence[str] | None = None,
    ) -> _transforms.Group:
        if wm_config is None:
            if model_config is None:
                raise ValueError("model_config must be provided when wm_config is not supplied.")
            wm_config = self._resolve_world_model_config(model_config)

        if not wm_enabled:
            return _transforms.Group(inputs=[])

        return _transforms.Group(
            inputs=[
                _transforms.SplitTemporalFrames(
                    frame_indices=wm_config.resolve_layout_indices(),
                    image_keys=image_keys,
                )
            ]
        )

    def _load_norm_stats(
        self,
        assets_dir: epath.Path,
        asset_id: str | Sequence[str] | None,
    ) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None

        asset_ids = [asset_id] if isinstance(asset_id, str) else [str(item) for item in asset_id]
        for single_asset_id in asset_ids:
            try:
                data_assets_dir = str(assets_dir / normalize_asset_id(single_asset_id))
                norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
                logging.info(f"Loaded norm stats from {data_assets_dir}")
                return norm_stats
            except FileNotFoundError:
                logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None

    def _resolve_norm_stats(
        self,
        assets_dir: epath.Path,
        repo_id: str | Sequence[str] | None,
        asset_id: str | Sequence[str] | None,
    ) -> tuple[dict[str, _transforms.NormStats] | None, dict[str, dict[str, _transforms.NormStats]] | None]:
        if asset_id is None:
            return None, None

        repo_ids = list(repo_id) if isinstance(repo_id, Sequence) and not isinstance(repo_id, str) else None
        asset_ids = list(asset_id) if isinstance(asset_id, Sequence) and not isinstance(asset_id, str) else None
        if repo_ids is None or asset_ids is None or len(repo_ids) != len(asset_ids):
            return self._load_norm_stats(assets_dir, asset_id), None

        first_norm_stats = None
        per_repo_norm_stats: dict[str, dict[str, _transforms.NormStats]] = {}
        for single_repo_id, single_asset_id in zip(repo_ids, asset_ids, strict=True):
            try:
                data_assets_dir = str(assets_dir / normalize_asset_id(single_asset_id))
                stats = _normalize.load(_download.maybe_download(data_assets_dir))
                logging.info(f"Loaded norm stats from {data_assets_dir}")
                per_repo_norm_stats[single_repo_id] = stats
                if first_norm_stats is None:
                    first_norm_stats = stats
            except FileNotFoundError:
                logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")

        return first_norm_stats, per_repo_norm_stats or None


def _load_local_dataset_info(repo_id: str) -> dict[str, Any] | None:
    repo_path = pathlib.Path(repo_id).expanduser()
    if not repo_path.is_dir():
        return None

    info_path = repo_path.resolve() / "meta" / "info.json"
    if not info_path.is_file():
        return None

    try:
        return json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logging.warning("Failed to read dataset info %s: %s", info_path, exc)
        return None


def _load_local_dataset_raw_stats(repo_id: str) -> dict[str, Any] | None:
    repo_path = pathlib.Path(repo_id).expanduser()
    if not repo_path.is_dir():
        return None
    stats_path = repo_path.resolve() / "meta" / "stats.json"
    if not stats_path.is_file():
        stats_path = repo_path.resolve() / "meta" / "norm_stats.json"
    if not stats_path.is_file():
        raise ValueError(
            f"Local LIBERO dataset {repo_path.resolve()} has no meta/norm_stats.json. "
            "Refusing to guess gripper semantics; generate raw dataset statistics first."
        )
    try:
        raw = json.loads(stats_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to read raw dataset stats {stats_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Expected an object in raw dataset stats {stats_path}.")
    return raw


def _raw_feature_last_dim_range(stats: dict[str, Any], key: str) -> tuple[float, float]:
    feature = stats.get(key)
    if not isinstance(feature, dict):
        raise ValueError(f"Raw dataset stats are missing feature {key!r}.")
    q01 = feature.get("q01")
    q99 = feature.get("q99")
    if not isinstance(q01, list) or not q01 or not isinstance(q99, list) or not q99:
        raise ValueError(f"Raw dataset stats for {key!r} must contain non-empty q01/q99 arrays.")
    return float(q01[-1]), float(q99[-1])


def _validate_local_libero_gripper_contract(
    repo_id: str | Sequence[str],
    *,
    state_format: libero_policy.LiberoStateGripperFormat,
    action_format: libero_policy.LiberoActionGripperFormat,
) -> None:
    expanded = _expand_local_repo_ids(repo_id)
    repo_ids = [expanded] if isinstance(expanded, str) else list(expanded or ())
    for single_repo_id in repo_ids:
        stats = _load_local_dataset_raw_stats(str(single_repo_id))
        if stats is None:
            continue
        state_q01, state_q99 = _raw_feature_last_dim_range(stats, "observation.state")
        action_q01, action_q99 = _raw_feature_last_dim_range(stats, "action")

        if state_format == "physical_width":
            state_ok = -0.01 <= state_q01 and state_q99 <= 0.10
        elif state_format == "open_fraction":
            state_ok = -0.05 <= state_q01 and 0.50 <= state_q99 <= 1.05
        else:
            raise ValueError(f"Unsupported LIBERO state gripper format {state_format!r}.")

        if action_format == "signed_command":
            action_ok = action_q01 <= -0.50 and action_q99 >= 0.50
        elif action_format == "binary_target":
            action_ok = -0.05 <= action_q01 <= 0.50 and 0.50 <= action_q99 <= 1.05
        elif action_format == "absolute_physical_width":
            action_ok = -0.01 <= action_q01 and action_q99 <= 0.10
        else:
            raise ValueError(f"Unsupported LIBERO action gripper format {action_format!r}.")

        if not state_ok or not action_ok:
            raise ValueError(
                "LIBERO gripper contract mismatch for "
                f"{pathlib.Path(str(single_repo_id)).resolve()}: config declares "
                f"state={state_format}, action={action_format}, but raw q01/q99 are "
                f"state=({state_q01:.6g}, {state_q99:.6g}), "
                f"action=({action_q01:.6g}, {action_q99:.6g}). "
                "Fix the dataset declaration or conversion before computing norm stats/training."
            )


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """Data config for Libero datasets.

    When ``pretrain_world_model=True``, state and actions are omitted from
    the repack transform, the policy transform zeroes state and skips
    actions, and ``action_sequence_keys`` is empty.
    """

    extra_delta_transform: bool = False
    pretrain_world_model: bool = False
    canonicalize_ee_pose_gripper: bool = False
    treat_actions_as_commands: bool = False
    # 8D end-effector delta from the current state, applied after LIBERO
    # canonicalization. Every chunk step uses that same state, matching the
    # π₀.₅ DeltaActions convention. Not compatible with pretrain_world_model.
    use_canonical_ee_delta: bool = False
    dataset_state_gripper_format: libero_policy.LiberoStateGripperFormat = "physical_width"
    dataset_action_gripper_format: libero_policy.LiberoActionGripperFormat = "signed_command"
    base_image_key: str = "observation.images.image"
    wrist_image_key: str = "observation.images.wrist_image"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        if not self.pretrain_world_model and self.canonicalize_ee_pose_gripper:
            _validate_local_libero_gripper_contract(
                self.repo_id,
                state_format=self.dataset_state_gripper_format,
                action_format=self.dataset_action_gripper_format,
            )
        repack_mapping: dict[str, str] = {
            "observation/image": self.base_image_key,
            "observation/wrist_image": self.wrist_image_key,
            "prompt": "prompt",
        }
        if not self.pretrain_world_model:
            repack_mapping["observation/state"] = "observation.state"
            repack_mapping["actions"] = "action"
        repack_transform = _transforms.Group(
            inputs=[_transforms.RepackTransform(repack_mapping)]
        )
        wm_enabled = self._effective_world_model_enabled(model_config)
        wm_config = self._resolve_world_model_config(model_config)
        if wm_enabled and not wm_config.image_keys:
            wm_config = dataclasses.replace(wm_config, image_keys=(self.base_image_key,))

        name_change_map = self._create_name_change_map(repack_transform)
        repacked_image_keys = [name_change_map.get(key, key) for key in wm_config.image_keys] if wm_enabled and wm_config.image_keys else []

        world_model_transforms = self._create_world_model_transforms(
            wm_enabled=wm_enabled,
            wm_config=wm_config,
            model_config=model_config,
            image_keys=repacked_image_keys,
        )
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(
                model_type=model_config.model_type,
                pretrain_world_model=self.pretrain_world_model,
                action_dim=model_config.action_dim,
                enable_world_model=wm_enabled,
                image_keys=repacked_image_keys,
                canonicalize_ee_pose_gripper=self.canonicalize_ee_pose_gripper,
                treat_actions_as_commands=self.treat_actions_as_commands,
                dataset_state_gripper_format=self.dataset_state_gripper_format,
                dataset_action_gripper_format=self.dataset_action_gripper_format,
            )],
            outputs=[
                libero_policy.LiberoOutputs(
                    pretrain_world_model=self.pretrain_world_model,
                    canonicalize_ee_pose_gripper=self.canonicalize_ee_pose_gripper,
                    treat_actions_as_commands=self.treat_actions_as_commands,
                )
            ],
        )

        if not self.pretrain_world_model and self.extra_delta_transform:
            if self.canonicalize_ee_pose_gripper:
                raise ValueError(
                    "extra_delta_transform=True is not supported when canonicalize_ee_pose_gripper=True "
                    "because LIBERO actions are treated as EE command-space targets."
                )
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        base_config = self.create_base_config(assets_dirs, model_config)
        normalization_contract = (
            "libero_eef_v2:"
            f"state={self.dataset_state_gripper_format}:"
            f"action={self.dataset_action_gripper_format}:"
            "canonical_state=open_fraction:canonical_action=ee_delta"
        )
        if self.load_norm_stats and base_config.norm_stats is not None:
            asset_id = base_config.asset_id
            if not isinstance(asset_id, str):
                raise ValueError(f"Expected one LIBERO asset_id string, got {asset_id!r}.")
            _normalize.validate_contract(
                pathlib.Path(self.assets.assets_dir or assets_dirs) / normalize_asset_id(asset_id),
                normalization_contract,
            )
        data_config = dataclasses.replace(
            base_config,
            repack_transforms=repack_transform,
            world_model=wm_config,
            world_model_transforms=world_model_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=() if self.pretrain_world_model else ("action",),
            normalization_contract=normalization_contract,
        )
        if self.use_canonical_ee_delta:
            if self.pretrain_world_model:
                raise ValueError("use_canonical_ee_delta requires pretrain_world_model=False.")
            ee_mask = _transforms.make_bool_mask(8)
            data_config = dataclasses.replace(
                data_config,
                data_transforms=data_config.data_transforms.push(
                    inputs=[_transforms.DeltaActions(mask=ee_mask, ee_pose=True)],
                    outputs=[_transforms.AbsoluteActions(mask=ee_mask, ee_pose=True)],
                ),
            )
        return data_config


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "plaw-vla"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # train stage
    training_stage: Literal["wm_alignment", "post_training"] = "post_training"

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # If true, enable model-level gradient checkpointing. This reduces activation
    # memory but usually slows down each optimization step due to recomputation.
    enable_gradient_checkpointing: bool = True

    # Fine-grained control over which modules use gradient checkpointing.
    # None means all eligible modules. Supported values:
    #   "language_model"  — PaliGemma language backbone (largest memory saver)
    #   "vision_tower"    — SigLIP vision encoder
    #   "action_expert"   — Action decoder Gemma expert
    #   "wm_expert"       — World-model Gemma expert
    # Example: ["language_model", "vision_tower"] disables GC on both experts.
    gradient_checkpointing_modules: list[str] | None = None

    # Passed to PyTorch DDP. Setting this to False avoids per-step autograd graph
    # traversal for unused-parameter detection when the training graph is stable.
    ddp_find_unused_parameters: bool = True

    # Passed to PyTorch DDP. When None, the trainer keeps its existing heuristic.
    # Setting this to True can improve throughput if the graph is stable every step.
    ddp_static_graph: bool | None = None

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")
        object.__setattr__(self.model, "training_stage", self.training_stage)


# Use `get_config` if you need to get a config by name in your code.
_LIBERO_REPO_ID = "data/libero_v3_eef"
# 0.2s spacing through +1.2s. History and future are both 6 frames, so each side
# splits evenly into V-JEPA2 tubelets of size 2. The +1.0s frame is paired with +1.2s.
_LIBERO_TIME_OFFSETS_S = (
    -1.0,
    -0.8,
    -0.6,
    -0.4,
    -0.2,
    0.0,
    0.2,
    0.4,
    0.6,
    0.8,
    1.0,
    1.2,
)


def converted_libero_data(repo_id: str, asset_id: str) -> LeRobotLiberoDataConfig:
    """Data config for ``examples/libero/convert_libero_data_to_lerobot.py``.

    The converter writes 7D signed gripper commands and the default camera keys.
    ``asset_id`` must be a relative name; norm stats are stored at
    ``assets/<config>/<asset_id>/``.
    """
    return LeRobotLiberoDataConfig(
        repo_id=repo_id,
        assets=AssetsConfig(asset_id=asset_id),
        canonicalize_ee_pose_gripper=True,
        treat_actions_as_commands=True,
        dataset_state_gripper_format="physical_width",
        dataset_action_gripper_format="signed_command",
        use_canonical_ee_delta=True,
        base_config=DataConfig(
            prompt_from_task=True,
            action_time_step_s=0.1,
            world_model=WorldModelDataConfig(
                image_keys=("observation.images.image",),
                frame_stride=None,
                time_offsets_s=_LIBERO_TIME_OFFSETS_S,
            ),
        ),
    )


def _libero_data(
    *,
    pretrain_world_model: bool,
    use_canonical_ee_delta: bool,
    action_time_start_s: float | None = None,
) -> LeRobotLiberoDataConfig:
    """LIBERO LeRobot dataset used by the released three-stage recipe."""
    return LeRobotLiberoDataConfig(
        repo_id=_LIBERO_REPO_ID,
        load_norm_stats=not pretrain_world_model,
        assets=AssetsConfig(asset_id="libero_v3_eef"),
        pretrain_world_model=pretrain_world_model,
        canonicalize_ee_pose_gripper=True,
        treat_actions_as_commands=use_canonical_ee_delta,
        dataset_state_gripper_format="physical_width",
        dataset_action_gripper_format="absolute_physical_width",
        base_image_key="observation.images.head",
        wrist_image_key="observation.images.wrist_right",
        use_canonical_ee_delta=use_canonical_ee_delta,
        base_config=DataConfig(
            prompt_from_task=True,
            action_time_step_s=0.1,
            action_time_start_s=action_time_start_s,
            world_model=WorldModelDataConfig(
                image_keys=("observation.images.head",),
                frame_stride=None,
                time_offsets_s=_LIBERO_TIME_OFFSETS_S,
            ),
        ),
    )


def _plaw_model(
    *,
    wm_loss_dropout_alpha: float = 0.0,
    vjepa2_variant: str = "vitl-256",
) -> pi0_config.Pi0Config:
    """Shared π₀.₅ + world-model architecture. Variant only changes the encoder."""
    return pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        dtype="bfloat16",
        discrete_state_input=False,
        enable_world_model=True,
        attn_implementation="sdpa",
        wm_loss_dropout_alpha=wm_loss_dropout_alpha,
        vjepa2_variant=vjepa2_variant,
        vjepa2_enable_input_projector=vjepa2_variant != "vitl-256",
        wm_slot_max_len=768,
    )


def _stage_train_config(
    name: str,
    *,
    training_stage: Literal["wm_alignment", "post_training"],
    pretrain_world_model: bool,
    use_canonical_ee_delta: bool,
    wm_loss_dropout_alpha: float,
    num_train_steps: int,
    batch_size: int,
    warmup_steps: int,
    action_time_start_s: float | None = None,
    decay_steps: int | None = None,
    decay_lr: float = 5e-5,
    vjepa2_variant: str = "vitl-256",
    ddp_find_unused_parameters: bool = True,
    ddp_static_graph: bool | None = None,
) -> TrainConfig:
    return TrainConfig(
        name=name,
        exp_name=name,
        project_name="plaw-vla",
        model=_plaw_model(
            wm_loss_dropout_alpha=wm_loss_dropout_alpha,
            vjepa2_variant=vjepa2_variant,
        ),
        training_stage=training_stage,
        data=_libero_data(
            pretrain_world_model=pretrain_world_model,
            use_canonical_ee_delta=use_canonical_ee_delta,
            action_time_start_s=action_time_start_s,
        ),
        batch_size=batch_size,
        num_workers=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=warmup_steps,
            peak_lr=5e-5,
            decay_steps=max(num_train_steps, 100_000) if decay_steps is None else decay_steps,
            decay_lr=decay_lr,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        pytorch_weight_path=None,
        num_train_steps=num_train_steps,
        save_interval=5_000,
        wandb_enabled=True,
        ddp_find_unused_parameters=ddp_find_unused_parameters,
        ddp_static_graph=ddp_static_graph,
    )


_CONFIGS = [
    # Stage I freezes PaliGemma and the action expert. Those weights come from
    # the converted π₀.₅ checkpoint registered in openpi.training.base_checkpoints.
    _stage_train_config(
        "stage1_world_model_pretraining",
        training_stage="wm_alignment",
        pretrain_world_model=True,
        use_canonical_ee_delta=False,
        wm_loss_dropout_alpha=0.0,
        num_train_steps=100_000,
        batch_size=512,
        warmup_steps=0,
        ddp_find_unused_parameters=False,
        ddp_static_graph=True,
    ),
    # Stage II trains every module except the frozen V-JEPA2 encoder.
    _stage_train_config(
        "stage2_pretraining",
        training_stage="post_training",
        pretrain_world_model=False,
        use_canonical_ee_delta=True,
        wm_loss_dropout_alpha=0.3,
        num_train_steps=100_000,
        batch_size=256,
        warmup_steps=0,
    ),
    # Stage III keeps the 0.3 world-model dropout. The draw is shared across
    # ranks, so dropped steps skip the branch on every GPU together.
    _stage_train_config(
        "stage3_finetuning_libero",
        training_stage="post_training",
        pretrain_world_model=False,
        use_canonical_ee_delta=True,
        wm_loss_dropout_alpha=0.3,
        num_train_steps=50_000,
        batch_size=256,
        warmup_steps=10_000,
        action_time_start_s=0.0,
        # Match openpi pi05_libero: warmup to 5e-5, then hold that value.
        decay_steps=1_000_000,
        decay_lr=5e-5,
    ),
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05_world_model",
        model=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="dummy",
            action_expert_variant="dummy",
            world_model_expert_variant="dummy",
        ),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05_world_model",
        wandb_enabled=False,
    ),
]


if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
