# Derived from openpi (Copyright 2024 Physical Intelligence, Inc.; Apache-2.0).
# Modified for PLaW-VLA by the PLaW-VLA authors, 2026.
"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import copy
import dataclasses
import difflib
import json
import logging
import os
import pathlib
from typing import Any, ClassVar, Literal, Protocol, TypeAlias

import etils.epath as epath
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.tokenizer as _tokenizer
import openpi.policies.agibot_policy as agibot_policy
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.egodex_policy as egodex_policy
import openpi.policies.intern_a1_policy as intern_a1_policy
import openpi.policies.libero_plus_policy as libero_plus_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.optimizer as _optimizer
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType

_LOCAL_REPO_CHILDREN_MANIFEST = ".child_datasets_manifest.json"
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

    # Explicit temporal configuration for count/stride sampling.
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
    # Source groups retain independently validated transforms and statistics for each child.
    child_configs: tuple["DataConfig", ...] = ()

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


def _raw_feature_dim_range(stats: dict[str, Any], key: str, *, index: int = -1) -> tuple[float, float]:
    feature = stats.get(key)
    if not isinstance(feature, dict):
        raise ValueError(f"Raw dataset stats are missing feature {key!r}.")
    q01 = feature.get("q01")
    q99 = feature.get("q99")
    if not isinstance(q01, list) or not q01 or not isinstance(q99, list) or not q99:
        raise ValueError(f"Raw dataset stats for {key!r} must contain non-empty q01/q99 arrays.")
    if not -len(q01) <= index < len(q01) or not -len(q99) <= index < len(q99):
        raise ValueError(f"Raw dataset stats for {key!r} are missing dimension {index}.")
    return float(q01[index]), float(q99[index])


def _validate_local_libero_gripper_contract(
    repo_id: str | Sequence[str],
    *,
    state_format: libero_policy.LiberoStateGripperFormat,
    action_format: libero_policy.LiberoActionGripperFormat,
    state_input_format: libero_policy.LiberoStateInputFormat = "canonical",
) -> None:
    expanded = _expand_local_repo_ids(repo_id)
    repo_ids = [expanded] if isinstance(expanded, str) else list(expanded or ())
    for single_repo_id in repo_ids:
        stats = _load_local_dataset_raw_stats(str(single_repo_id))
        if stats is None:
            continue
        state_q01, state_q99 = _raw_feature_dim_range(stats, "observation.state")
        action_q01, action_q99 = _raw_feature_dim_range(stats, "action")

        if state_input_format == "two_finger_qpos":
            if state_format != "physical_width":
                raise ValueError("two_finger_qpos requires physical finger positions in metres.")
            other_q01, other_q99 = _raw_feature_dim_range(stats, "observation.state", index=-2)
            state_q01 = min(state_q01, other_q01)
            state_q99 = max(state_q99, other_q99)
            state_ok = -0.10 <= state_q01 <= state_q99 <= 0.10
        elif state_input_format != "canonical":
            raise ValueError(f"Unsupported LIBERO state input format {state_input_format!r}.")
        elif state_format == "physical_width":
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
                f"input={state_input_format}, state={state_format}, action={action_format}, but raw q01/q99 are "
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
    dataset_state_input_format: libero_policy.LiberoStateInputFormat = "canonical"
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
                state_input_format=self.dataset_state_input_format,
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
                state_input_format=self.dataset_state_input_format,
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
class LocalStateActionLayout:
    state_dim: int
    action_dim: int
    state_names: tuple[str, ...] | None = None
    action_names: tuple[str, ...] | None = None

    def infer_state_semantics(self) -> ActionSemantics | None:
        return _infer_semantics_from_feature_names(self.state_names)

    def infer_action_semantics(self) -> ActionSemantics | None:
        return _infer_semantics_from_feature_names(self.action_names)

def _normalize_feature_names(names: Any) -> tuple[str, ...] | None:
    if not isinstance(names, Sequence) or isinstance(names, str | bytes):
        return None
    return tuple(str(name) for name in names)

def _infer_semantics_from_feature_names(names: tuple[str, ...] | None) -> ActionSemantics | None:
    if not names:
        return None
    normalized = tuple(str(name) for name in names)
    if any("effector" in name for name in normalized):
        return "joint_effector_position"
    if any("joint_" in name for name in normalized):
        return "joint_position"
    if any("quaternion" in name for name in normalized):
        return "ee_pose"
    return None

def _resolve_local_state_action_layout(repo_id: str | Sequence[str] | None) -> LocalStateActionLayout | None:
    expanded_repo_id = _expand_local_repo_ids(repo_id)
    if expanded_repo_id is None:
        return None

    repo_ids = [expanded_repo_id] if isinstance(expanded_repo_id, str) else [str(item) for item in expanded_repo_id]
    resolved_layouts: set[tuple[int, int, tuple[str, ...] | None, tuple[str, ...] | None]] = set()
    for single_repo_id in repo_ids:
        info = _load_local_dataset_info(single_repo_id)
        if info is None:
            continue
        features = info.get("features", {})
        if not isinstance(features, dict):
            continue
        state_feature = features.get("observation.state", {})
        action_feature = features.get("action", features.get("actions", {}))
        state_shape = tuple(state_feature.get("shape") or ())
        action_shape = tuple(action_feature.get("shape") or ())
        if not state_shape or not action_shape:
            continue
        resolved_layouts.add(
            (
                int(state_shape[-1]),
                int(action_shape[-1]),
                _normalize_feature_names(state_feature.get("names")),
                _normalize_feature_names(action_feature.get("names")),
            )
        )

    if len(resolved_layouts) > 1:
        raise ValueError(
            "Mixed local state/action layouts are not supported in one config: "
            f"{sorted(resolved_layouts)}"
        )
    if not resolved_layouts:
        return None

    state_dim, action_dim, state_names, action_names = resolved_layouts.pop()
    return LocalStateActionLayout(
        state_dim=state_dim,
        action_dim=action_dim,
        state_names=state_names,
        action_names=action_names,
    )

def _infer_local_agibot_eef_types(repo_id: str | Sequence[str] | None) -> set[str]:
    if repo_id is None:
        return set()

    repo_ids = [repo_id] if isinstance(repo_id, str) else [str(item) for item in repo_id]
    eef_types: set[str] = set()
    for single_repo_id in repo_ids:
        info = _load_local_dataset_info(single_repo_id)
        if info is None:
            continue
        eef_type = info.get("agibot_eef_type") or info.get("embodiment")
        if isinstance(eef_type, str) and eef_type.strip():
            eef_types.add(eef_type.strip().lower())
    return eef_types

def _resolve_agibot_eef_type(
    repo_id: str | Sequence[str] | None,
    explicit_eef_type: Literal["gripper", "dexhand"] | None,
) -> Literal["gripper", "dexhand"]:
    inferred_eef_types = _infer_local_agibot_eef_types(repo_id)

    if len(inferred_eef_types) > 1:
        raise ValueError(
            f"Mixed AgiBot embodiments are not supported in one training config: {sorted(inferred_eef_types)}"
        )

    if explicit_eef_type is not None:
        if inferred_eef_types and inferred_eef_types != {explicit_eef_type}:
            raise ValueError(
                f"Configured AgiBot embodiment {explicit_eef_type!r} does not match dataset metadata "
                f"{sorted(inferred_eef_types)}."
            )
        return explicit_eef_type

    if not inferred_eef_types:
        logging.info("Could not infer AgiBot embodiment from dataset metadata; defaulting to gripper layout.")
        return "gripper"

    inferred_eef_type = inferred_eef_types.pop()
    if inferred_eef_type not in {"gripper", "dexhand"}:
        raise ValueError(f"Unsupported AgiBot embodiment {inferred_eef_type!r}.")
    return inferred_eef_type

def _agibot_native_action_dim(
    eef_type: Literal["gripper", "dexhand"],
    *,
    canonical_gripper_action_space: Literal["joint_effector_position", "ee_pose"] = "joint_effector_position",
) -> int:
    if eef_type == "gripper" and canonical_gripper_action_space == "ee_pose":
        return 16
    return 22 if eef_type == "gripper" else 32

def _agibot_delta_mask(
    eef_type: Literal["gripper", "dexhand"],
    *,
    canonical_gripper_action_space: Literal["joint_effector_position", "ee_pose"] = "joint_effector_position",
) -> tuple[bool, ...]:
    if eef_type == "gripper" and canonical_gripper_action_space == "ee_pose":
        return tuple()
    return _AGIBOT_GRIPPER_DELTA_MASK if eef_type == "gripper" else _AGIBOT_DEXHAND_DELTA_MASK

def _agibot_state_mask(
    eef_type: Literal["gripper", "dexhand"],
    *,
    canonical_gripper_action_space: Literal["joint_effector_position", "ee_pose"] = "joint_effector_position",
) -> tuple[bool, ...]:
    if eef_type == "gripper" and canonical_gripper_action_space == "ee_pose":
        return (False,) * 16
    return _AGIBOT_GRIPPER_STATE_MASK if eef_type == "gripper" else _AGIBOT_DEXHAND_STATE_MASK

def _agibot_action_mask(
    eef_type: Literal["gripper", "dexhand"],
    *,
    canonical_gripper_action_space: Literal["joint_effector_position", "ee_pose"] = "joint_effector_position",
) -> tuple[bool, ...]:
    if eef_type == "gripper" and canonical_gripper_action_space == "ee_pose":
        return (False,) * 16
    return _AGIBOT_GRIPPER_ACTION_MASK if eef_type == "gripper" else _AGIBOT_DEXHAND_ACTION_MASK

def _agibot_repack_mapping(pretrain_world_model: bool) -> dict[str, str]:
    repack_mapping: dict[str, str] = {
        "top_head": "observation.images.head",
        "hand_left": "observation.images.hand_left",
        "hand_right": "observation.images.hand_right",
        "prompt": "prompt",
    }
    if not pretrain_world_model:
        repack_mapping["state"] = "observation.state"
        repack_mapping["actions"] = "actions"
    return repack_mapping

@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    """Shared ALOHA-style data transforms used by RoboTwin datasets.

    When ``pretrain_world_model=True``, state and actions are omitted from
    the repack transform, the policy transform zeroes state and skips
    actions, and ``action_sequence_keys`` is empty.
    """

    use_delta_joint_actions: bool = True
    default_prompt: str | None = None
    adapt_to_pi: bool = True
    native_action_dim: int = 14
    pretrain_world_model: bool = False

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group | None] = None
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    def _build_repack_transforms(self) -> _transforms.Group:
        if self.repack_transforms is not None:
            return self.repack_transforms
        mapping: dict[str, str] = {
            "cam_high": "observation.images.cam_high",
            "cam_left_wrist": "observation.images.cam_left_wrist",
            "cam_right_wrist": "observation.images.cam_right_wrist",
            "prompt": "prompt",
        }
        if not self.pretrain_world_model:
            mapping["state"] = "observation.state"
            mapping["actions"] = "action"
        return _transforms.Group(inputs=[_transforms.RepackTransform(mapping)])

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        base_config = self.create_base_config(assets_dirs, model_config)
        wm_enabled = self._effective_world_model_enabled(model_config)
        wm_config = self._resolve_world_model_config(model_config)
        image_keys = tuple(wm_config.image_keys) if wm_config.image_keys else _ALOHA_DEFAULT_IMAGE_KEYS
        effective_wm_config = dataclasses.replace(wm_config, image_keys=image_keys)
        repack = self._build_repack_transforms()

        name_change_map = self._create_name_change_map(repack)
        repacked_image_keys = [name_change_map.get(key, key) for key in image_keys] if wm_enabled else []

        world_model_transforms = self._create_world_model_transforms(
            wm_enabled=wm_enabled,
            wm_config=effective_wm_config,
            model_config=model_config,
            image_keys=repacked_image_keys,
        )

        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(
                adapt_to_pi=self.adapt_to_pi,
                pretrain_world_model=self.pretrain_world_model,
                action_dim=model_config.action_dim,
                native_action_dim=self.native_action_dim,
                enable_world_model=wm_enabled,
                image_keys=repacked_image_keys,
            )],
            outputs=[
                aloha_policy.AlohaOutputs(
                    adapt_to_pi=self.adapt_to_pi,
                    pretrain_world_model=self.pretrain_world_model,
                    native_action_dim=self.native_action_dim,
                )
            ],
        )
        if not self.pretrain_world_model and self.use_delta_joint_actions:
            if self.native_action_dim != 14:
                raise ValueError(
                    "use_delta_joint_actions=True is only supported for canonical 14D Aloha joint actions, "
                    f"got native_action_dim={self.native_action_dim}."
                )
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            base_config,
            world_model=effective_wm_config,
            repack_transforms=repack,
            world_model_transforms=world_model_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=() if self.pretrain_world_model else self.action_sequence_keys,
        )

_LIBERO_PLUS_DEFAULT_IMAGE_KEYS: tuple[str, ...] = ("observation.images.front",)

@dataclasses.dataclass(frozen=True)
class LeRobotLiberoPlusDataConfig(DataConfigFactory):
    """Data config for Libero Plus datasets."""

    extra_delta_transform: bool = False
    pretrain_world_model: bool = False
    canonicalize_ee_pose_gripper: bool = False
    treat_actions_as_commands: bool = True

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_mapping: dict[str, str] = {
            "observation/front_image": "observation.images.front",
            "observation/wrist_image": "observation.images.wrist",
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
        image_keys = tuple(wm_config.image_keys) if wm_config.image_keys else _LIBERO_PLUS_DEFAULT_IMAGE_KEYS

        name_change_map = self._create_name_change_map(repack_transform)
        repacked_image_keys = [name_change_map.get(key, key) for key in image_keys] if wm_enabled else []

        world_model_transforms = self._create_world_model_transforms(
            wm_enabled=wm_enabled,
            wm_config=wm_config,
            model_config=model_config,
            image_keys=repacked_image_keys,
        )
        data_transforms = _transforms.Group(
            inputs=[libero_plus_policy.LiberoPlusInputs(
                model_type=model_config.model_type,
                pretrain_world_model=self.pretrain_world_model,
                action_dim=model_config.action_dim,
                enable_world_model=wm_enabled,
                image_keys=repacked_image_keys,
                canonicalize_ee_pose_gripper=self.canonicalize_ee_pose_gripper,
                treat_actions_as_commands=self.treat_actions_as_commands,
            )],
            outputs=[
                libero_plus_policy.LiberoPlusOutputs(
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
                    "because LIBERO+ actions are treated as EE command-space targets."
                )
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            world_model_transforms=world_model_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=() if self.pretrain_world_model else ("action",),
        )

_EGODEX_DEFAULT_IMAGE_KEYS: tuple[str, ...] = ("observation.images.top",)

_EGODEX_DEFAULT_ACTION_STRIDE = 3

@dataclasses.dataclass(frozen=True)
class LeRobotEgoDexDataConfig(DataConfigFactory):
    """
    Data config for EgoDex dataset in LeRobot format.

    To build EgoDex v3 datasets from raw data, see examples/egodex/convert_egodex_to_lerobot.py.
    The converter keeps per-frame absolute 48 DoF hand states/actions. The
    loader assembles the queried future action chunk at training time. Uniform
    action-time settings override the default ``action_stride`` spacing. For
    camera-frame delta actions, enable ``use_delta_actions=True`` and the
    transform will be applied during training/stat computation instead of
    preprocessing. Wrist images are still masked during training. When
    ``pretrain_world_model=True``, only images + prompt are loaded; state is
    zeroed and actions are omitted.
    """

    pretrain_world_model: bool = False
    use_delta_actions: bool = False
    action_stride: int = _EGODEX_DEFAULT_ACTION_STRIDE

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        if self.action_stride <= 0:
            raise ValueError(f"action_stride must be > 0, got {self.action_stride}.")
        base_config = self.create_base_config(assets_dirs, model_config)
        if self.pretrain_world_model:
            base_config = dataclasses.replace(base_config, norm_stats=None)

        repack_mapping = {
            "observation/image": "observation.images.top",
            "prompt": "task",
        }
        if not self.pretrain_world_model:
            repack_mapping["state"] = "observation.state"
            repack_mapping["actions"] = "actions"
            if self.use_delta_actions:
                repack_mapping["camera_extrinsic"] = "egodex.camera_extrinsic"

        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    repack_mapping
                )
            ]
        )
        wm_enabled = self._effective_world_model_enabled(model_config)
        wm_config = self._resolve_world_model_config(model_config)
        image_keys = tuple(wm_config.image_keys) if wm_config.image_keys else _EGODEX_DEFAULT_IMAGE_KEYS

        name_change_map = self._create_name_change_map(repack_transform)
        repacked_image_keys = [name_change_map.get(key, key) for key in image_keys] if wm_enabled else []

        world_model_transforms = self._create_world_model_transforms(
            wm_enabled=wm_enabled,
            wm_config=wm_config,
            model_config=model_config,
            image_keys=repacked_image_keys,
        )
        input_transforms: list[_transforms.DataTransformFn] = []
        if not self.pretrain_world_model and self.use_delta_actions:
            input_transforms.append(egodex_policy.EgoDexDeltaActions())
        input_transforms.append(egodex_policy.EgoDexInputs(
            model_type=model_config.model_type,
            action_dim=model_config.action_dim,
            pretrain_world_model=self.pretrain_world_model,
            enable_world_model=wm_enabled,
            image_keys=repacked_image_keys,
        ))
        data_transforms = _transforms.Group(
            inputs=input_transforms,
            outputs=[egodex_policy.EgoDexOutputs(
                action_dim=model_config.action_dim,
                pretrain_world_model=self.pretrain_world_model,
            )],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            base_config,
            repack_transforms=repack_transform,
            world_model_transforms=world_model_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=() if self.pretrain_world_model else ("actions",),
            action_sequence_offsets=(
                None
                if self.pretrain_world_model
                else tuple(self.action_stride * (i + 1) for i in range(model_config.action_horizon))
            ),
        )

@dataclasses.dataclass(frozen=True)
class LeRobotAgiBotWorldDataConfig(DataConfigFactory):
    """Data config for AgiBotWorld training.

    When ``pretrain_world_model=False`` (default), expects canonical
    ``observation.state`` / ``actions`` fields. Embodiment metadata is used
    only to select the native action dimension and masking layout. When
    ``pretrain_world_model=True``, only images + prompt are loaded; state is
    zeroed and actions are omitted.
    """

    repo_id: str | Sequence[str] = tyro.MISSING
    pretrain_world_model: bool = False
    use_delta_joint_actions: bool = True
    eef_type: Literal["gripper", "dexhand"] | None = None
    canonical_gripper_action_space: Literal["joint_effector_position", "ee_pose"] = "joint_effector_position"
    canonicalize_gripper_openness: bool = True

    action_sequence_keys: Sequence[str] = ("actions",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        base_config = self.create_base_config(assets_dirs, model_config)
        wm_enabled = self._effective_world_model_enabled(model_config)
        wm_config = self._resolve_world_model_config(model_config)
        eef_type = _resolve_agibot_eef_type(base_config.repo_id, self.eef_type)
        canonicalize_gripper_openness = self.canonicalize_gripper_openness and eef_type == "gripper"
        native_action_dim = _agibot_native_action_dim(
            eef_type,
            canonical_gripper_action_space=self.canonical_gripper_action_space,
        )
        if eef_type == "gripper" and self.canonical_gripper_action_space == "ee_pose" and self.use_delta_joint_actions:
            raise ValueError(
                "use_delta_joint_actions=True is not supported when canonical_gripper_action_space='ee_pose' "
                "because quaternion pose actions are not subtraction-compatible."
            )

        repack_mapping = _agibot_repack_mapping(self.pretrain_world_model)
        repack_transform = _transforms.Group(
            inputs=[_transforms.RepackTransform(repack_mapping)]
        )

        name_change_map = self._create_name_change_map(repack_transform)
        repacked_image_keys = [name_change_map.get(key, key) for key in wm_config.image_keys] if wm_enabled and wm_config.image_keys else []

        world_model_transforms = self._create_world_model_transforms(
            wm_enabled=wm_enabled,
            wm_config=wm_config,
            model_config=model_config,
            image_keys=repacked_image_keys,
        )
        data_transforms = _transforms.Group(
            inputs=[agibot_policy.AGIBotInputs(
                action_dim=model_config.action_dim,
                pretrain_world_model=self.pretrain_world_model,
                state_mask=_agibot_state_mask(
                    eef_type,
                    canonical_gripper_action_space=self.canonical_gripper_action_space,
                ),
                action_mask=_agibot_action_mask(
                    eef_type,
                    canonical_gripper_action_space=self.canonical_gripper_action_space,
                ),
                native_action_dim=native_action_dim,
                enable_world_model=wm_enabled,
                image_keys=repacked_image_keys,
                canonicalize_gripper_openness=canonicalize_gripper_openness,
                state_semantics=(
                    "ee_pose"
                    if eef_type == "gripper" and self.canonical_gripper_action_space == "ee_pose"
                    else "joint_effector_position"
                ),
            )],
            outputs=[agibot_policy.AGIBotOutputs(
                native_action_dim=native_action_dim,
                pretrain_world_model=self.pretrain_world_model,
                canonicalize_gripper_openness=canonicalize_gripper_openness,
                state_semantics=(
                    "ee_pose"
                    if eef_type == "gripper" and self.canonical_gripper_action_space == "ee_pose"
                    else "joint_effector_position"
                ),
            )],
        )
        if not self.pretrain_world_model and self.use_delta_joint_actions:
            data_transforms = data_transforms.push(
                inputs=[
                    _transforms.DeltaActions(
                        _agibot_delta_mask(
                            eef_type,
                            canonical_gripper_action_space=self.canonical_gripper_action_space,
                        )
                    )
                ],
                outputs=[
                    _transforms.AbsoluteActions(
                        _agibot_delta_mask(
                            eef_type,
                            canonical_gripper_action_space=self.canonical_gripper_action_space,
                        )
                    )
                ],
            )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            base_config,
            repack_transforms=repack_transform,
            world_model_transforms=world_model_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=() if self.pretrain_world_model else self.action_sequence_keys,
        )

_AGIBOT_GRIPPER_DELTA_MASK = _transforms.make_bool_mask(14, -6)

_AGIBOT_GRIPPER_STATE_MASK = _transforms.make_bool_mask(-16, 4)

_AGIBOT_GRIPPER_ACTION_MASK = _transforms.make_bool_mask(-16, 6)

_AGIBOT_DEXHAND_DELTA_MASK = _transforms.make_bool_mask(14, -16)

_AGIBOT_DEXHAND_STATE_MASK = _transforms.make_bool_mask(-26, 4)

_AGIBOT_DEXHAND_ACTION_MASK = _transforms.make_bool_mask(-26, 6)

_INTERN_A1_DELTA_MASK = _transforms.make_bool_mask(7, -1, 7, -1)

_INTERN_A1_DEFAULT_IMAGE_KEYS: tuple[str, ...] = ("observation.images.cam_high",)

_ALOHA_DEFAULT_IMAGE_KEYS: tuple[str, ...] = ("observation.images.cam_high",)

@dataclasses.dataclass(frozen=True)
class LeRobotInternA1DataConfig(DataConfigFactory):
    """Data config for canonical InternData-A1 LeRobot v3 datasets."""

    default_prompt: str | None = None
    pretrain_world_model: bool = False
    use_delta_joint_actions: bool = False
    canonical_action_space: Literal["joint_position", "ee_pose"] = "joint_position"
    action_sequence_keys: Sequence[str] = ("actions",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        base_config = self.create_base_config(assets_dirs, model_config)
        if self.pretrain_world_model:
            base_config = dataclasses.replace(base_config, norm_stats=None)
        wm_enabled = self._effective_world_model_enabled(model_config)
        wm_config = self._resolve_world_model_config(model_config)
        if self.canonical_action_space == "ee_pose" and self.use_delta_joint_actions:
            raise ValueError(
                "use_delta_joint_actions=True is not supported when canonical_action_space='ee_pose' "
                "because quaternion pose actions are not subtraction-compatible."
            )

        image_keys = tuple(wm_config.image_keys) if wm_config.image_keys else _INTERN_A1_DEFAULT_IMAGE_KEYS
        world_model_transforms = self._create_world_model_transforms(
            wm_enabled=wm_enabled,
            wm_config=wm_config,
            model_config=model_config,
            image_keys=image_keys,
        )

        data_transforms = _transforms.Group(
            inputs=[
                intern_a1_policy.InternA1Inputs(
                    model_type=model_config.model_type,
                    pretrain_world_model=self.pretrain_world_model,
                    action_dim=model_config.action_dim,
                    enable_world_model=wm_enabled,
                    image_keys=image_keys,
                    state_semantics=self.canonical_action_space,
                )
            ],
            outputs=[
                intern_a1_policy.InternA1Outputs(
                    pretrain_world_model=self.pretrain_world_model,
                    state_semantics=self.canonical_action_space,
                )
            ],
        )

        if not self.pretrain_world_model and self.use_delta_joint_actions:
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(_INTERN_A1_DELTA_MASK)],
                outputs=[_transforms.AbsoluteActions(_INTERN_A1_DELTA_MASK)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            base_config,
            repack_transforms=_transforms.Group(),
            world_model_transforms=world_model_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=() if self.pretrain_world_model else self.action_sequence_keys,
        )

DatasetType: TypeAlias = Literal[
    "agibot",
    "egodex",
    "intern_a1",
    "libero",
    "libero_plus",
    "robotwin",
]

@dataclasses.dataclass(frozen=True)
class MultiDatasetPretrainDatasetSpec:
    """Spec for one dataset group in multi-dataset training."""

    repo_id: str | Sequence[str]
    dataset_type: DatasetType
    weight: float = 1.0
    # Optional override for world-model temporal image keys for this dataset type.
    # If None, _DEFAULT_IMAGE_KEYS[dataset_type] is used.
    image_keys: Sequence[str] | None = None
    # Optional override for the world-model frame stride for this dataset.
    # This is useful when mixing datasets with different FPS while keeping a
    # similar real-time temporal step across the mixture.
    world_model_frame_stride: int | None = None

@dataclasses.dataclass(frozen=True)
class MultiDatasetPretrainDataConfig(DataConfigFactory):
    """Unified multi-dataset config for action/world-model training."""

    repo_id: str = "multi_dataset_pretrain"
    datasets: tyro.conf.Suppress[Sequence[MultiDatasetPretrainDatasetSpec]] = ()
    use_canonical_delta_actions: bool = False

    _TYPE_FACTORIES: ClassVar[dict[str, type[DataConfigFactory]]] = {
        "agibot": LeRobotAgiBotWorldDataConfig,
        "egodex": LeRobotEgoDexDataConfig,
        "intern_a1": LeRobotInternA1DataConfig,
        "libero": LeRobotLiberoDataConfig,
        "libero_plus": LeRobotLiberoPlusDataConfig,
        "robotwin": LeRobotAlohaDataConfig,
    }
    _FACTORY_OVERRIDES: ClassVar[dict[str, dict[str, Any]]] = {
        # Disable local per-dataset delta transforms so multi-dataset training
        # can apply one shared delta stage after canonical alignment auditing.
        "agibot": {"use_delta_joint_actions": False},
        "egodex": {"use_delta_actions": False},
        "intern_a1": {"use_delta_joint_actions": False},
        "libero": {
            "extra_delta_transform": False,
            "canonicalize_ee_pose_gripper": True,
            # LIBERO and Stage III converters store absolute EEF targets
            # and physical gripper widths before the shared delta transform.
            "dataset_state_gripper_format": "physical_width",
            "dataset_action_gripper_format": "absolute_physical_width",
        },
        "libero_plus": {"extra_delta_transform": False, "canonicalize_ee_pose_gripper": True},
        # RoboTwin defaults to the ee16 layout; local metadata can override it.
        "robotwin": {"use_delta_joint_actions": False, "adapt_to_pi": False, "native_action_dim": 16},
    }

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        configs = self.create_all(assets_dirs, model_config)
        return configs[0][0]

    def create_all(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> list[tuple[DataConfig, float]]:
        """Create a DataConfig for each dataset spec, paired with its sampling weight."""
        if not self.datasets:
            raise ValueError("MultiDatasetPretrainDataConfig requires at least one dataset spec.")

        result: list[tuple[DataConfig, float]] = []
        training_stage = self._resolve_training_stage(model_config)
        for spec in self.datasets:
            expanded_repo_id = _expand_local_repo_ids(spec.repo_id)
            repo_ids = [expanded_repo_id] if isinstance(expanded_repo_id, str) else list(expanded_repo_id or ())
            if not repo_ids:
                raise ValueError(f"Dataset group {spec.dataset_type!r} does not contain any datasets.")
            asset_ids = self.assets.asset_id
            if isinstance(asset_ids, Sequence) and not isinstance(asset_ids, str) and len(asset_ids) != len(repo_ids):
                raise ValueError(f"Dataset group {spec.dataset_type!r} must have one asset ID per child dataset.")
            children = []
            for index, repo_id in enumerate(repo_ids):
                child_assets = dataclasses.replace(
                    self.assets,
                    asset_id=asset_ids[index]
                    if isinstance(asset_ids, Sequence) and not isinstance(asset_ids, str)
                    else asset_ids,
                )
                child_spec = dataclasses.replace(spec, repo_id=repo_id)
                children.append(
                    self._create_child_config(child_spec, child_assets, assets_dirs, model_config, training_stage)
                )
            if len(children) == 1:
                data_config = children[0]
            else:
                data_config = dataclasses.replace(
                    children[0],
                    repo_id=tuple(child.repo_id for child in children),
                    asset_id=tuple(child.asset_id for child in children),
                    norm_stats=None,
                    per_repo_norm_stats={
                        child.repo_id: child.norm_stats for child in children if child.norm_stats is not None
                    }
                    or None,
                    action_space_descriptor=None,
                    normalization_contract=None,
                    child_configs=tuple(children),
                )
            result.append((data_config, spec.weight))
            logging.info(
                "MultiDatasetPretrain: stage=%s, type=%s, children=%d, weight=%s",
                training_stage,
                spec.dataset_type,
                len(children),
                spec.weight,
            )
        return result

    def _create_child_config(
        self,
        spec: MultiDatasetPretrainDatasetSpec,
        assets: AssetsConfig,
        assets_dirs: pathlib.Path,
        model_config: _model.BaseModelConfig,
        training_stage: Literal["wm_alignment", "post_training"],
    ) -> DataConfig:
        factory = dataclasses.replace(self._create_factory_for_spec(spec, training_stage=training_stage), assets=assets)
        data_config = factory.create(assets_dirs, model_config)
        descriptor = self._audit_action_space(spec, data_config)
        self._validate_descriptor_against_local_metadata(
            dataset_type=spec.dataset_type,
            repo_id=data_config.repo_id,
            descriptor=descriptor,
        )
        pretrain_flags = [
            bool(transform.pretrain_world_model)
            for transform in data_config.data_transforms.inputs
            if hasattr(transform, "pretrain_world_model")
        ]
        descriptor.validate_for_stage(
            dataset_type=spec.dataset_type,
            training_stage=training_stage,
            has_actions=bool(data_config.action_sequence_keys),
        )
        if training_stage == "post_training" and any(pretrain_flags):
            raise ValueError(
                f"Dataset type {spec.dataset_type!r} uses pretrain_world_model semantics during post_training."
            )
        data_config = dataclasses.replace(
            data_config, dataset_type=spec.dataset_type, action_space_descriptor=descriptor
        )
        use_delta = training_stage == "post_training" and self.use_canonical_delta_actions
        if use_delta:
            data_config = self._apply_shared_delta_transform(
                data_config, dataset_type=spec.dataset_type, descriptor=descriptor
            )
        contract = data_config.normalization_contract or self._normalization_contract(
            data_config, descriptor, model_config.action_dim, use_delta
        )
        data_config = dataclasses.replace(data_config, normalization_contract=contract)
        if self.load_norm_stats and data_config.norm_stats is not None:
            if not isinstance(data_config.asset_id, str):
                raise ValueError(f"Expected one asset ID for child dataset {data_config.repo_id!r}.")
            stats_dir = epath.Path(assets.assets_dir or assets_dirs) / normalize_asset_id(data_config.asset_id)
            _normalize.validate_contract(_download.maybe_download(str(stats_dir)), contract)
        return data_config

    def _normalization_contract(
        self,
        data_config: DataConfig,
        descriptor: ActionSpaceDescriptor,
        model_action_dim: int,
        use_delta: bool,
    ) -> str:
        policy = data_config.data_transforms.inputs[0]
        policy_options = {
            name: getattr(policy, name)
            for name in ("adapt_to_pi", "canonicalize_gripper_openness", "state_semantics")
            if hasattr(policy, name)
        }
        for name in ("state_mask", "action_mask"):
            mask = getattr(policy, name, None)
            if mask is not None:
                policy_options[name] = [bool(value) for value in mask]
        return "pretraining_v1:" + json.dumps(
            {
                "dataset_type": data_config.dataset_type,
                "canonical_space": descriptor.canonical_space_id,
                "state_dim": descriptor.state_dim,
                "action_dim": descriptor.action_dim,
                "model_action_dim": model_action_dim,
                "state_semantics": descriptor.state_semantics,
                "action_semantics": descriptor.action_semantics,
                "action_representation": (
                    "none"
                    if not data_config.action_sequence_keys
                    else "delta_from_current_state"
                    if use_delta
                    else "absolute"
                ),
                "delta_mask": descriptor.state_action_alignment_mask if use_delta else None,
                "policy_options": policy_options,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    _DEFAULT_IMAGE_KEYS: ClassVar[dict[str, tuple[str, ...]]] = {
        "agibot": ("observation.images.head",),
        "egodex": ("observation.images.top",),
        "intern_a1": ("observation.images.cam_high",),
        "libero": ("observation.images.image",),
        "libero_plus": _LIBERO_PLUS_DEFAULT_IMAGE_KEYS,
        "robotwin": ("observation.images.cam_high",),
    }

    def _apply_factory_field_overrides(
        self,
        kwargs: dict[str, Any],
        *,
        field_names: set[str],
        overrides: dict[str, Any],
    ) -> None:
        for key, value in overrides.items():
            if key in field_names:
                kwargs[key] = value

    def _local_layout_matches_ee_pose(
        self,
        local_layout: LocalStateActionLayout | None,
        *,
        expected_dim: int,
    ) -> bool:
        return (
            local_layout is not None
            and local_layout.state_dim == expected_dim
            and local_layout.action_dim == expected_dim
            and local_layout.infer_state_semantics() == "ee_pose"
            and local_layout.infer_action_semantics() == "ee_pose"
        )

    def _apply_training_stage_overrides(
        self,
        kwargs: dict[str, Any],
        *,
        spec: MultiDatasetPretrainDatasetSpec,
        field_names: set[str],
        training_stage: Literal["wm_alignment", "post_training"],
        local_layout: LocalStateActionLayout | None,
    ) -> None:
        if "pretrain_world_model" in field_names:
            kwargs["pretrain_world_model"] = training_stage == "wm_alignment"
        self._apply_factory_field_overrides(
            kwargs,
            field_names=field_names,
            overrides=self._FACTORY_OVERRIDES.get(spec.dataset_type, {}),
        )
        if training_stage == "post_training" and self.use_canonical_delta_actions:
            if spec.dataset_type in {"libero", "libero_plus"} and "treat_actions_as_commands" in field_names:
                kwargs["treat_actions_as_commands"] = True
        if spec.dataset_type == "agibot" and "canonical_gripper_action_space" in field_names:
            if self._local_layout_matches_ee_pose(local_layout, expected_dim=16):
                kwargs["canonical_gripper_action_space"] = "ee_pose"
        if spec.dataset_type == "intern_a1" and "canonical_action_space" in field_names:
            if self._local_layout_matches_ee_pose(local_layout, expected_dim=intern_a1_policy.CANONICAL_ACTION_DIM):
                kwargs["canonical_action_space"] = "ee_pose"

    def _apply_aloha_layout_overrides(
        self,
        kwargs: dict[str, Any],
        *,
        spec: MultiDatasetPretrainDatasetSpec,
        field_names: set[str],
        local_layout: LocalStateActionLayout | None,
    ) -> None:
        if not {"adapt_to_pi", "native_action_dim"}.issubset(field_names) or local_layout is None:
            return
        if local_layout.state_dim != local_layout.action_dim:
            raise ValueError(
                f"Dataset type {spec.dataset_type!r} uses an Aloha-style config but metadata reports "
                f"observation.state dim {local_layout.state_dim} and action dim {local_layout.action_dim}."
            )
        kwargs["adapt_to_pi"] = local_layout.action_dim == 14
        kwargs["native_action_dim"] = local_layout.action_dim

    def _create_factory_for_spec(
        self,
        spec: MultiDatasetPretrainDatasetSpec,
        *,
        training_stage: Literal["wm_alignment", "post_training"],
    ) -> DataConfigFactory:
        factory_cls = self._TYPE_FACTORIES.get(spec.dataset_type)
        if factory_cls is None:
            raise ValueError(
                f"Unknown dataset type '{spec.dataset_type}'. "
                f"Available: {list(self._TYPE_FACTORIES.keys())}"
            )

        base = self.base_config or DataConfig()
        type_image_keys = (
            tuple(spec.image_keys)
            if spec.image_keys is not None
            else self._DEFAULT_IMAGE_KEYS.get(spec.dataset_type, ())
        )
        world_model_updates: dict[str, Any] = {"image_keys": type_image_keys}
        if spec.world_model_frame_stride is not None:
            world_model_updates["frame_stride"] = spec.world_model_frame_stride
        wm = dataclasses.replace(base.world_model, **world_model_updates)
        per_type_base = dataclasses.replace(base, world_model=wm)

        repo_id = spec.repo_id
        kwargs: dict[str, Any] = {
            "repo_id": repo_id,
            "base_config": per_type_base,
        }
        factory_field_names = {f.name for f in dataclasses.fields(factory_cls)}
        if self.assets.assets_dir or self.assets.asset_id:
            kwargs["assets"] = self.assets
        if "load_norm_stats" in factory_field_names:
            kwargs["load_norm_stats"] = self.load_norm_stats
        local_layout = _resolve_local_state_action_layout(repo_id)
        self._apply_training_stage_overrides(
            kwargs,
            spec=spec,
            field_names=factory_field_names,
            training_stage=training_stage,
            local_layout=local_layout,
        )
        self._apply_aloha_layout_overrides(
            kwargs,
            spec=spec,
            field_names=factory_field_names,
            local_layout=local_layout,
        )

        return factory_cls(**kwargs)

    def _apply_shared_delta_transform(
        self,
        data_config: DataConfig,
        *,
        dataset_type: str,
        descriptor: ActionSpaceDescriptor,
    ) -> DataConfig:
        if (
            descriptor.state_semantics == "ee_pose"
            and descriptor.action_semantics == "ee_pose"
            and descriptor.state_dim == descriptor.action_dim
        ):
            ee_prefix_mask = _transforms.make_bool_mask(descriptor.action_dim)
            return dataclasses.replace(
                data_config,
                data_transforms=data_config.data_transforms.push(
                    inputs=[_transforms.DeltaActions(mask=ee_prefix_mask, ee_pose=True)],
                    outputs=[_transforms.AbsoluteActions(mask=ee_prefix_mask, ee_pose=True)],
                ),
            )
        mask = descriptor.validate_for_shared_delta(dataset_type=dataset_type)
        return dataclasses.replace(
            data_config,
            data_transforms=data_config.data_transforms.push(
                inputs=[_transforms.DeltaActions(mask)],
                outputs=[_transforms.AbsoluteActions(mask)],
            ),
        )

    def _audit_action_space(
        self,
        spec: MultiDatasetPretrainDatasetSpec,
        data_config: DataConfig,
    ) -> ActionSpaceDescriptor:
        match spec.dataset_type:
            case "robotwin":
                return self._audit_robotwin_action_space(data_config)
            case "agibot":
                return self._audit_agibot_action_space(data_config)
            case "intern_a1":
                return self._audit_intern_a1_action_space(data_config)
            case "libero" | "libero_plus":
                return self._audit_libero_action_space(data_config, dataset_type=spec.dataset_type)
            case "egodex":
                return self._audit_egodex_action_space(data_config)
            case _:
                raise ValueError(f"Unsupported dataset type {spec.dataset_type!r} for action-space auditing.")

    def _validate_descriptor_against_local_metadata(
        self,
        *,
        dataset_type: str,
        repo_id: str | Sequence[str] | None,
        descriptor: ActionSpaceDescriptor,
    ) -> None:
        if dataset_type in {"egodex", "libero", "libero_plus"}:
            # These datasets intentionally canonicalize their raw 8D/7D storage
            # into a different training space inside the policy transform, so
            # the post-transform descriptor does not match raw info.json
            # dimensions.
            return
        local_layout = _resolve_local_state_action_layout(repo_id)
        if local_layout is None:
            return
        if descriptor.state_dim != local_layout.state_dim or descriptor.action_dim != local_layout.action_dim:
            raise ValueError(
                f"Dataset type {dataset_type!r} audit descriptor "
                f"(state_dim={descriptor.state_dim}, action_dim={descriptor.action_dim}) does not match "
                f"local metadata (state_dim={local_layout.state_dim}, action_dim={local_layout.action_dim}). "
                f"Audit note: {descriptor.audit_note or 'none'}"
            )
        local_state_semantics = local_layout.infer_state_semantics()
        if local_state_semantics is not None and descriptor.state_semantics != local_state_semantics:
            raise ValueError(
                f"Dataset type {dataset_type!r} audit descriptor state semantics {descriptor.state_semantics!r} "
                f"do not match local metadata semantics {local_state_semantics!r}. "
                f"Audit note: {descriptor.audit_note or 'none'}"
            )
        local_action_semantics = local_layout.infer_action_semantics()
        if local_action_semantics is not None and descriptor.action_semantics != local_action_semantics:
            raise ValueError(
                f"Dataset type {dataset_type!r} audit descriptor action semantics {descriptor.action_semantics!r} "
                f"do not match local metadata semantics {local_action_semantics!r}. "
                f"Audit note: {descriptor.audit_note or 'none'}"
            )

    def _audit_robotwin_action_space(self, data_config: DataConfig) -> ActionSpaceDescriptor:
        inputs = data_config.data_transforms.inputs
        if not inputs or not isinstance(inputs[0], aloha_policy.AlohaInputs):
            raise ValueError("RobotWin multi-dataset configs must start with AlohaInputs.")
        transform = inputs[0]
        if transform.adapt_to_pi:
            return ActionSpaceDescriptor(
                state_semantics="joint_position",
                action_semantics="joint_position",
                state_dim=transform.native_action_dim,
                action_dim=transform.native_action_dim,
                canonical_space_id=f"bimanual_joint_position_{transform.native_action_dim}",
                supports_delta_from_state=True,
                state_action_alignment_mask=_transforms.make_bool_mask(6, -1, 6, -1),
                audit_note="RobotWin joint-space layouts can use the shared subtraction delta on joint/gripper-aligned prefixes.",
            )
        if transform.native_action_dim == 16:
            return ActionSpaceDescriptor(
                state_semantics="ee_pose",
                action_semantics="ee_pose",
                state_dim=transform.native_action_dim,
                action_dim=transform.native_action_dim,
                canonical_space_id="bimanual_ee_pose_16",
                supports_delta_from_state=False,
                audit_note="RobotWin ee16 uses xyz+quaternion pose control; generic subtraction delta is invalid.",
            )
        return ActionSpaceDescriptor(
            state_semantics="joint_position",
            action_semantics="joint_position",
            state_dim=transform.native_action_dim,
            action_dim=transform.native_action_dim,
            canonical_space_id=f"bimanual_robotwin_{transform.native_action_dim}",
            supports_delta_from_state=False,
            audit_note="RobotWin raw semantics could not be normalized to a shared subtraction-compatible delta rule.",
        )

    def _audit_agibot_action_space(self, data_config: DataConfig) -> ActionSpaceDescriptor:
        inputs = data_config.data_transforms.inputs
        if not inputs or not isinstance(inputs[0], agibot_policy.AGIBotInputs):
            raise ValueError("AgiBot multi-dataset configs must start with AGIBotInputs.")
        transform = inputs[0]
        if transform.state_semantics == "ee_pose":
            return ActionSpaceDescriptor(
                state_semantics="ee_pose",
                action_semantics="ee_pose",
                state_dim=16,
                action_dim=16,
                canonical_space_id="bimanual_ee_pose_16",
                supports_delta_from_state=False,
                audit_note="AgiBot gripper ee canonicalization uses xyz+quaternion+gripper per arm; generic subtraction delta is invalid.",
            )
        if transform.native_action_dim == 22:
            return ActionSpaceDescriptor(
                state_semantics="joint_effector_position",
                action_semantics="joint_effector_position",
                state_dim=20,
                action_dim=22,
                canonical_space_id="agibot_gripper_joint_effector",
                supports_delta_from_state=True,
                state_action_alignment_mask=_AGIBOT_GRIPPER_DELTA_MASK,
                audit_note="Comparable delta block is the shared 14D joint-position prefix; non-comparable tail is masked.",
            )
        if transform.native_action_dim == 32:
            return ActionSpaceDescriptor(
                state_semantics="joint_effector_position",
                action_semantics="joint_effector_position",
                state_dim=30,
                action_dim=32,
                canonical_space_id="agibot_dexhand_joint_effector",
                supports_delta_from_state=True,
                state_action_alignment_mask=_AGIBOT_DEXHAND_DELTA_MASK,
                audit_note="Comparable delta block is the shared 14D joint-position prefix; hand/head/waist tails are not delta-compatible.",
            )
        raise ValueError(f"Unsupported AgiBot native action dim {transform.native_action_dim}.")

    def _audit_intern_a1_action_space(self, data_config: DataConfig) -> ActionSpaceDescriptor:
        inputs = data_config.data_transforms.inputs
        if not inputs or not isinstance(inputs[0], intern_a1_policy.InternA1Inputs):
            raise ValueError("Intern-A1 multi-dataset configs must start with InternA1Inputs.")
        transform = inputs[0]
        if transform.state_semantics == "ee_pose":
            return ActionSpaceDescriptor(
                state_semantics="ee_pose",
                action_semantics="ee_pose",
                state_dim=intern_a1_policy.CANONICAL_STATE_DIM,
                action_dim=intern_a1_policy.CANONICAL_ACTION_DIM,
                canonical_space_id="bimanual_ee_pose_16",
                supports_delta_from_state=False,
                audit_note="Intern-A1 ee canonicalization uses ee_to_robot_pose + gripper per arm; generic subtraction delta is invalid.",
            )
        return ActionSpaceDescriptor(
            state_semantics="joint_position",
            action_semantics="joint_position",
            state_dim=intern_a1_policy.CANONICAL_STATE_DIM,
            action_dim=intern_a1_policy.CANONICAL_ACTION_DIM,
            canonical_space_id="bimanual_joint_position_16",
            supports_delta_from_state=True,
            state_action_alignment_mask=_INTERN_A1_DELTA_MASK,
            audit_note="Intern-A1 canonical state/actions share the same 16D bimanual joint+gripper layout.",
        )

    def _build_libero_descriptor(
        self,
        *,
        canonicalized: bool,
        treat_actions_as_commands: bool,
        action_gripper_format: libero_policy.LiberoActionGripperFormat,
        dataset_label: str,
    ) -> ActionSpaceDescriptor:
        action_semantics = "ee_pose" if canonicalized and treat_actions_as_commands else "ee_pose_command"
        if canonicalized and treat_actions_as_commands:
            if action_gripper_format == "absolute_physical_width":
                audit_note = f"{dataset_label} stores absolute EEF poses and physical gripper widths, canonicalized before the shared delta transform."
            elif action_gripper_format == "binary_target":
                audit_note = (
                    f"{dataset_label} maps raw Cartesian-control xyz/rotation deltas plus binary open-high gripper "
                    "targets into canonical absolute ee pose targets before the shared canonical delta transform."
                )
            else:
                audit_note = (
                    f"{dataset_label} maps raw Cartesian-control actions into canonical absolute ee pose targets "
                    "before the shared canonical delta transform."
                )
        elif canonicalized:
            audit_note = (
                f"{dataset_label} canonicalizes state/action to xyz+quaternion+gripper, but actions are Cartesian "
                "control commands rather than subtraction-compatible next-state targets."
            )
        else:
            audit_note = (
                f"{dataset_label} raw state/action are not subtraction-compatible: state stores two finger "
                "positions while actions are Cartesian control commands."
            )
        return ActionSpaceDescriptor(
            state_semantics="ee_pose",
            action_semantics=action_semantics,
            state_dim=8,
            action_dim=8 if canonicalized else 7,
            canonical_space_id="single_arm_ee_pose_gripper_8" if canonicalized else "single_arm_ee_pose_command_raw",
            supports_delta_from_state=False,
            audit_note=audit_note,
        )

    def _audit_libero_action_space(self, data_config: DataConfig, *, dataset_type: str) -> ActionSpaceDescriptor:
        inputs = data_config.data_transforms.inputs
        transform = inputs[0] if inputs else None
        if dataset_type == "libero":
            if not isinstance(transform, libero_policy.LiberoInputs):
                raise ValueError("Libero multi-dataset configs must start with LiberoInputs.")
            return self._build_libero_descriptor(
                canonicalized=transform.canonicalize_ee_pose_gripper,
                treat_actions_as_commands=transform.treat_actions_as_commands,
                action_gripper_format=transform.dataset_action_gripper_format,
                dataset_label="Libero",
            )
        else:
            if not isinstance(transform, libero_plus_policy.LiberoPlusInputs):
                raise ValueError("Libero+ multi-dataset configs must start with LiberoPlusInputs.")
            return self._build_libero_descriptor(
                canonicalized=transform.canonicalize_ee_pose_gripper,
                treat_actions_as_commands=transform.treat_actions_as_commands,
                action_gripper_format="signed_command",
                dataset_label="Libero+",
            )

    def _audit_egodex_action_space(self, data_config: DataConfig) -> ActionSpaceDescriptor:
        inputs = data_config.data_transforms.inputs
        if not inputs:
            raise ValueError("EgoDex multi-dataset configs must define data transforms.")
        egodex_transform = next((transform for transform in inputs if isinstance(transform, egodex_policy.EgoDexInputs)), None)
        if egodex_transform is None:
            raise ValueError("EgoDex multi-dataset configs must include EgoDexInputs.")
        return ActionSpaceDescriptor(
            state_semantics="hand_pose",
            action_semantics="hand_pose",
            state_dim=egodex_transform.action_dim,
            action_dim=egodex_transform.action_dim,
            canonical_space_id=f"egodex_hand_pose_{egodex_transform.action_dim}",
            supports_delta_from_state=False,
            audit_note="EgoDex delta actions require camera-frame conversion and cannot use the shared subtraction transform.",
        )


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

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.CosineDecaySchedule = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.AdamW = dataclasses.field(default_factory=_optimizer.AdamW)
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

    def __post_init__(self) -> None:
        if not isinstance(self.optimizer, _optimizer.AdamW):
            raise ValueError("PyTorch training supports the AdamW optimizer configuration.")
        if not isinstance(self.lr_schedule, _optimizer.CosineDecaySchedule):
            raise ValueError("PyTorch training supports the cosine decay learning-rate schedule configuration.")
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")
        for name in ("batch_size", "num_train_steps", "log_interval", "save_interval"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        if not isinstance(self.num_workers, int) or isinstance(self.num_workers, bool) or self.num_workers < 0:
            raise ValueError(f"num_workers must be a non-negative integer, got {self.num_workers!r}.")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or not 0 <= self.seed < 2**32:
            raise ValueError(f"seed must be an integer in [0, 2**32), got {self.seed!r}.")
        if self.training_stage not in ("wm_alignment", "post_training"):
            raise ValueError(f"Unsupported training_stage {self.training_stage!r}.")
        model = copy.copy(self.model)
        object.__setattr__(model, "training_stage", self.training_stage)
        object.__setattr__(self, "model", model)


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
        dataset_state_input_format="two_finger_qpos",
        dataset_state_gripper_format="physical_width",
        dataset_action_gripper_format="signed_command",
        use_canonical_ee_delta=True,
        base_config=DataConfig(
            prompt_from_task=True,
            action_time_step_s=0.1,
            action_time_start_s=0.0,
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
    """LIBERO LeRobot dataset used for Stage III fine-tuning."""
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


def _policy_model(
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


def _pretraining_data(*, world_model_only: bool) -> MultiDatasetPretrainDataConfig:
    root = pathlib.Path(os.environ.get("DATA_ROOT", pathlib.Path(__file__).resolve().parents[3] / "data/pretrain")).expanduser()
    sources = [("intern_a1", "intern_a1", 0.20), ("agibotworld", "agibot", 0.30), ("robotwin", "robotwin", 0.15), ("libero", "libero", 0.08 if world_model_only else 0.10)]
    if world_model_only:
        sources.append(("egodex", "egodex", 0.15))
    return MultiDatasetPretrainDataConfig(
        datasets=tuple(MultiDatasetPretrainDatasetSpec(repo_id=str(root / directory), dataset_type=kind, weight=weight) for directory, kind, weight in sources),
        use_canonical_delta_actions=not world_model_only,
        load_norm_stats=not world_model_only,
        base_config=DataConfig(prompt_from_task=True, action_time_step_s=0.1, action_time_start_s=0.0,
            world_model=WorldModelDataConfig(history_num_frames=6, future_num_frames=6, frame_stride=None, time_offsets_s=(-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2))),
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
        model=_policy_model(
            wm_loss_dropout_alpha=wm_loss_dropout_alpha,
            vjepa2_variant=vjepa2_variant,
        ),
        training_stage=training_stage,
        data=(
            _libero_data(pretrain_world_model=False, use_canonical_ee_delta=True, action_time_start_s=action_time_start_s)
            if name == "stage3_finetuning_libero"
            else _pretraining_data(world_model_only=pretrain_world_model)
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
    dataclasses.replace(
        _stage_train_config("stage3_finetuning_robotwin", training_stage="post_training", pretrain_world_model=False, use_canonical_ee_delta=True, wm_loss_dropout_alpha=0.3, num_train_steps=50_000, batch_size=256, warmup_steps=10_000, action_time_start_s=0.0),
        data=MultiDatasetPretrainDataConfig(
            datasets=(MultiDatasetPretrainDatasetSpec(repo_id=str(pathlib.Path(os.environ.get("DATA_ROOT", "data/pretrain")) / "robotwin"), dataset_type="robotwin"),),
            use_canonical_delta_actions=True,
            base_config=DataConfig(prompt_from_task=True, action_time_step_s=0.1, action_time_start_s=0.0, world_model=WorldModelDataConfig(history_num_frames=6, future_num_frames=6, frame_stride=None, time_offsets_s=(-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2))),
        ),
        policy_metadata={"robotwin_action_type": "ee", "robotwin_native_action_dim": 16},
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
