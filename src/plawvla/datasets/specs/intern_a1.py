# Copyright 2026 PLaW-VLA authors. SPDX-License-Identifier: Apache-2.0
"""Shared InternData-A1 conversion helpers for direct LeRobot v3 builders."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
import dataclasses
import errno
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import shutil
import tarfile
from typing import Any, Literal
import warnings

import av
from lerobot.datasets.video_utils import get_video_duration_in_s
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

from plawvla.datasets.common.lerobot_v3 import DirectVideoLeRobotDataset
from plawvla.datasets.common.lerobot_v3 import MergeLinkMode
from plawvla.shared import normalize

DEFAULT_SOURCE_ROOT = Path("data/raw/intern_a1/sim_updated")
DEFAULT_RAW_OUTPUT_ROOT = Path("data/raw/intern_a1/lerobot_v21_raw")
DEFAULT_CANONICAL_OUTPUT_ROOT = Path("data/pretrain/intern_a1")
DEFAULT_ASSET_ROOT = Path("assets/intern_a1")

KNOWN_EMBODIMENTS = ("franka", "genie1", "lift2", "split_aloha")
KNOWN_CATEGORIES = ("articulation_tasks", "basic_tasks", "long_horizon_tasks", "pick_and_place_tasks")

CANONICAL_STATE_DIM = 16
CANONICAL_ACTION_DIM = 16
CANONICAL_CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
CANONICAL_IMAGE_MASK_NAMES = ("cam_high", "cam_left_wrist", "cam_right_wrist")
_VALIDATED_SOURCE_VIDEO_PATHS: set[str] = set()
_SOURCE_VIDEO_DURATIONS_S: dict[str, float] = {}
JOINT_CANONICAL_STATE_NAMES = (
    "left_joint_0",
    "left_joint_1",
    "left_joint_2",
    "left_joint_3",
    "left_joint_4",
    "left_joint_5",
    "left_joint_6",
    "left_gripper",
    "right_joint_0",
    "right_joint_1",
    "right_joint_2",
    "right_joint_3",
    "right_joint_4",
    "right_joint_5",
    "right_joint_6",
    "right_gripper",
)
EE_POSE_CANONICAL_STATE_NAMES = (
    "left_position.x",
    "left_position.y",
    "left_position.z",
    "left_quaternion.w",
    "left_quaternion.x",
    "left_quaternion.y",
    "left_quaternion.z",
    "left_gripper",
    "right_position.x",
    "right_position.y",
    "right_position.z",
    "right_quaternion.w",
    "right_quaternion.x",
    "right_quaternion.y",
    "right_quaternion.z",
    "right_gripper",
)


def build_future_action_chunks(actions: np.ndarray, action_horizon: int) -> np.ndarray:
    """Build clamped future action chunks matching LeRobot delta_timestamps semantics."""
    if action_horizon <= 0:
        raise ValueError(f"action_horizon must be positive, got {action_horizon}.")

    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"Expected actions with shape [num_frames, dim], got {actions.shape}.")
    if actions.shape[0] == 0:
        raise ValueError("Cannot build future action chunks from an empty episode.")

    query_indices = np.minimum(
        np.arange(actions.shape[0], dtype=np.int64)[:, None] + np.arange(action_horizon, dtype=np.int64)[None, :],
        actions.shape[0] - 1,
    )
    return actions[query_indices]

@dataclasses.dataclass(frozen=True)
class InternA1Layout:
    embodiment: str
    state_semantics: Literal["joint_position", "ee_pose"]
    left_state_key: str
    left_gripper_key: str
    left_state_dim: int
    head_camera_key: str
    left_camera_key: str
    right_state_key: str | None = None
    right_gripper_key: str | None = None
    right_state_dim: int = 0
    right_camera_key: str | None = None

    @property
    def left_action_key(self) -> str:
        return self.left_state_key.replace("states.", "actions.", 1)

    @property
    def left_gripper_action_key(self) -> str:
        return self.left_gripper_key.replace("states.", "actions.", 1)

    @property
    def right_action_key(self) -> str | None:
        if self.right_state_key is None:
            return None
        return self.right_state_key.replace("states.", "actions.", 1)

    @property
    def right_gripper_action_key(self) -> str | None:
        if self.right_gripper_key is None:
            return None
        return self.right_gripper_key.replace("states.", "actions.", 1)


def normalize_filters(values: Iterable[str] | None, *, known_values: tuple[str, ...]) -> tuple[str, ...]:
    if not values:
        return known_values

    normalized = tuple(str(value).strip().lower() for value in values)
    invalid = sorted(set(normalized) - set(known_values))
    if invalid:
        raise ValueError(f"Unsupported values {invalid}; expected a subset of {known_values}.")
    return normalized


def infer_embodiment(path: Path) -> str | None:
    parts = {part.lower() for part in path.parts}
    for embodiment in KNOWN_EMBODIMENTS:
        if embodiment in parts:
            return embodiment
    return None


def infer_category(path: Path) -> str | None:
    parts = {part.lower() for part in path.parts}
    for category in KNOWN_CATEGORIES:
        if category in parts:
            return category
    return None


def discover_archives(
    source_root: Path,
    *,
    embodiments: Iterable[str] | None = None,
    categories: Iterable[str] | None = None,
) -> list[Path]:
    normalized_embodiments = normalize_filters(embodiments, known_values=KNOWN_EMBODIMENTS)
    normalized_categories = normalize_filters(categories, known_values=KNOWN_CATEGORIES)

    archives: list[Path] = []
    for category in normalized_categories:
        for embodiment in normalized_embodiments:
            archives.extend(sorted((source_root / category / embodiment).glob("*.tar.gz")))
    return archives


def find_dataset_root_in_archive(archive_path: Path) -> PurePosixPath:
    # Stream tar headers and stop at the first dataset marker instead of
    # materializing the full member list up front.
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            if member.isfile() and member.name.endswith("meta/info.json"):
                return PurePosixPath(member.name).parent.parent

    raise ValueError(f"{archive_path} does not contain a LeRobot meta/info.json file.")


def discover_dataset_dirs(
    dataset_root: Path,
    *,
    embodiments: Iterable[str] | None = None,
    categories: Iterable[str] | None = None,
) -> dict[str, list[Path]]:
    normalized_embodiments = normalize_filters(embodiments, known_values=KNOWN_EMBODIMENTS)
    normalized_categories = normalize_filters(categories, known_values=KNOWN_CATEGORIES)

    search_roots: list[Path] = []
    for category in normalized_categories:
        category_root = dataset_root / category
        if not category_root.is_dir():
            continue
        for embodiment in normalized_embodiments:
            embodiment_root = category_root / embodiment
            if embodiment_root.is_dir():
                search_roots.append(embodiment_root)
    if not search_roots:
        search_roots = [dataset_root]

    grouped: dict[str, list[Path]] = defaultdict(list)
    for search_root in search_roots:
        direct_children = sorted(path for path in search_root.iterdir() if path.is_dir()) if search_root.is_dir() else []
        direct_info_paths = [child / "meta" / "info.json" for child in direct_children]
        if all(info_path.is_file() for info_path in direct_info_paths):
            candidate_info_paths = direct_info_paths
        else:
            candidate_info_paths = sorted(search_root.rglob("meta/info.json"))

        for info_path in candidate_info_paths:
            if not info_path.is_file():
                continue
            dataset_dir = info_path.parent.parent
            embodiment = infer_embodiment(dataset_dir)
            category = infer_category(dataset_dir)
            if embodiment is None or category is None:
                continue
            if embodiment not in normalized_embodiments or category not in normalized_categories:
                continue
            if not is_complete_dataset_dir(dataset_dir):
                warnings.warn(
                    f"Skipping incomplete InternData-A1 dataset dir: {dataset_dir}",
                    stacklevel=2,
                )
                continue
            grouped[embodiment].append(dataset_dir.resolve())

    return {embodiment: sorted(set(paths)) for embodiment, paths in grouped.items()}


def load_dataset_info(dataset_dir: Path) -> dict:
    return json.loads((dataset_dir / "meta" / "info.json").read_text(encoding="utf-8"))


def _has_episode_files(dataset_dir: Path) -> bool:
    data_dir = dataset_dir / "data"
    return any(data_dir.glob("chunk-*/episode_*.parquet")) or any(data_dir.glob("chunk-*/file-*.parquet"))


def is_complete_dataset_dir(dataset_dir: Path) -> bool:
    dataset_dir = dataset_dir.expanduser().resolve()
    return (
        (dataset_dir / "meta" / "info.json").is_file()
        and (dataset_dir / "meta" / "tasks.jsonl").is_file()
        and _has_episode_files(dataset_dir)
    )


def infer_layout_from_info(info: dict, *, dataset_dir: Path | None = None) -> InternA1Layout:
    features = info.get("features", {})
    embodiment = infer_embodiment(dataset_dir) if dataset_dir is not None else None

    if {"states.ee_to_robot_pose", "states.gripper.position"} <= features.keys():
        return InternA1Layout(
            embodiment=embodiment or "franka",
            state_semantics="ee_pose",
            left_state_key="states.ee_to_robot_pose",
            left_gripper_key="states.gripper.position",
            left_state_dim=int(features["states.ee_to_robot_pose"]["shape"][0]),
            head_camera_key="images.rgb.head",
            left_camera_key="images.rgb.hand",
        )

    if {"states.joint.position", "states.gripper.position"} <= features.keys():
        return InternA1Layout(
            embodiment=embodiment or "franka",
            state_semantics="joint_position",
            left_state_key="states.joint.position",
            left_gripper_key="states.gripper.position",
            left_state_dim=int(features["states.joint.position"]["shape"][0]),
            head_camera_key="images.rgb.head",
            left_camera_key="images.rgb.hand",
        )

    dual_arm_ee_keys = {
        "states.left_ee_to_robot_pose",
        "states.left_gripper.position",
        "states.right_ee_to_robot_pose",
        "states.right_gripper.position",
    }
    if dual_arm_ee_keys <= features.keys():
        return InternA1Layout(
            embodiment=embodiment or "genie1",
            state_semantics="ee_pose",
            left_state_key="states.left_ee_to_robot_pose",
            left_gripper_key="states.left_gripper.position",
            left_state_dim=int(features["states.left_ee_to_robot_pose"]["shape"][0]),
            right_state_key="states.right_ee_to_robot_pose",
            right_gripper_key="states.right_gripper.position",
            right_state_dim=int(features["states.right_ee_to_robot_pose"]["shape"][0]),
            head_camera_key="images.rgb.head",
            left_camera_key="images.rgb.hand_left",
            right_camera_key="images.rgb.hand_right",
        )

    dual_arm_keys = {
        "states.left_joint.position",
        "states.left_gripper.position",
        "states.right_joint.position",
        "states.right_gripper.position",
    }
    if dual_arm_keys <= features.keys():
        return InternA1Layout(
            embodiment=embodiment or "genie1",
            state_semantics="joint_position",
            left_state_key="states.left_joint.position",
            left_gripper_key="states.left_gripper.position",
            left_state_dim=int(features["states.left_joint.position"]["shape"][0]),
            right_state_key="states.right_joint.position",
            right_gripper_key="states.right_gripper.position",
            right_state_dim=int(features["states.right_joint.position"]["shape"][0]),
            head_camera_key="images.rgb.head",
            left_camera_key="images.rgb.hand_left",
            right_camera_key="images.rgb.hand_right",
        )

    raise ValueError(f"Unsupported InternData-A1 schema with features: {sorted(features)}")


def _column_to_vector(column: Any) -> np.ndarray:
    values = column_to_numpy(column)
    if values.dtype == object:
        values = np.asarray(values.tolist())
    return np.asarray(values)


def column_to_numpy(column: Any) -> np.ndarray:
    array = column.combine_chunks() if hasattr(column, "combine_chunks") else column
    if hasattr(array, "to_numpy"):
        try:
            return np.asarray(array.to_numpy(zero_copy_only=False))
        except TypeError:
            return np.asarray(array.to_numpy())
    return np.asarray(array.to_pylist())


def column_to_matrix(column: Any) -> np.ndarray:
    values = column_to_numpy(column)
    if values.dtype == object:
        values = np.asarray(values.tolist(), dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    return values


def compute_direct_norm_stats(
    dataset_dirs: list[Path],
    action_horizon: int,
    batch_size: int = 32,
    max_frames: int | None = None,
) -> dict[str, normalize.NormStats]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    stats = {
        "state": normalize.RunningStats(),
        "actions": normalize.RunningStats(),
    }
    remaining_frames = max_frames
    num_full_batches = 0
    pending_state: list[np.ndarray] = []
    pending_actions: list[np.ndarray] = []
    pending_size = 0

    def iter_parquet_paths(dataset_dir: Path) -> list[Path]:
        episode_paths = sorted(dataset_dir.glob("data/chunk-*/episode_*.parquet"))
        if episode_paths:
            return episode_paths
        return sorted(dataset_dir.glob("data/chunk-*/file-*.parquet"))

    def flush_full_batches() -> None:
        nonlocal pending_state, pending_actions, pending_size, num_full_batches
        while pending_size >= batch_size:
            state_batch = np.concatenate(pending_state, axis=0)
            action_batch = np.concatenate(pending_actions, axis=0)
            stats["state"].update(state_batch[:batch_size])
            stats["actions"].update(action_batch[:batch_size])
            num_full_batches += 1

            remainder_state = state_batch[batch_size:]
            remainder_actions = action_batch[batch_size:]
            pending_state = [remainder_state] if len(remainder_state) else []
            pending_actions = [remainder_actions] if len(remainder_actions) else []
            pending_size = len(remainder_state)

    def flush_remainder() -> None:
        nonlocal pending_state, pending_actions, pending_size, num_full_batches
        if pending_size <= 0:
            return
        state_batch = np.concatenate(pending_state, axis=0)
        action_batch = np.concatenate(pending_actions, axis=0)
        stats["state"].update(state_batch)
        stats["actions"].update(action_batch)
        num_full_batches += 1
        pending_state = []
        pending_actions = []
        pending_size = 0

    for dataset_dir in dataset_dirs:
        for parquet_path in iter_parquet_paths(dataset_dir):
            table = pq.read_table(parquet_path, columns=["observation.state", "actions"])
            state = column_to_matrix(table.column("observation.state"))
            actions = column_to_matrix(table.column("actions"))
            action_chunks = build_future_action_chunks(actions, action_horizon)

            if remaining_frames is not None:
                if remaining_frames <= 0:
                    flush_full_batches()
                    flush_remainder()
                    return {key: value.get_statistics() for key, value in stats.items()}
                state = state[:remaining_frames]
                action_chunks = action_chunks[:remaining_frames]
                remaining_frames -= len(state)

            pending_state.append(state)
            pending_actions.append(action_chunks)
            pending_size += len(state)
            flush_full_batches()

    flush_remainder()
    if num_full_batches <= 0:
        raise ValueError("No full batches were created. Reduce batch_size or increase the dataset size.")

    return {key: value.get_statistics() for key, value in stats.items()}


def init_output_norm_stats() -> dict[str, normalize.RunningStats]:
    return {
        "state": normalize.RunningStats(),
        "actions": normalize.RunningStats(),
    }


def update_output_norm_stats_from_prepared_episode(
    accumulators: dict[str, normalize.RunningStats],
    prepared_episode: dict[str, Any],
    *,
    action_horizon: int,
) -> None:
    columns = prepared_episode["episode_columns"]
    state = np.asarray(columns["observation.state"], dtype=np.float32)
    actions = np.asarray(columns["actions"], dtype=np.float32)
    accumulators["state"].update(state)
    accumulators["actions"].update(build_future_action_chunks(actions, action_horizon))


def finalize_output_norm_stats(
    accumulators: dict[str, normalize.RunningStats],
) -> dict[str, normalize.NormStats] | None:
    try:
        return {key: value.get_statistics() for key, value in accumulators.items()}
    except ValueError:
        return None


def _link_or_copy_file(source_path: Path, dest_path: Path) -> None:
    try:
        os.link(source_path, dest_path)
    except OSError as exc:
        if exc.errno in {errno.EXDEV, errno.EEXIST, errno.EPERM, errno.ENOTSUP}:
            shutil.copy2(source_path, dest_path)
            return
        raise


def _video_feature_from_raw(raw_feature: dict[str, Any]) -> dict[str, Any]:
    feature = {
        "dtype": "video",
        "shape": tuple(raw_feature["shape"]),
        "names": raw_feature.get("names"),
    }
    video_info = raw_feature.get("video_info") or raw_feature.get("info")
    if video_info is not None:
        feature["video_info"] = dict(video_info)
    return feature


def _add_common_features(features: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {key: dict(value) for key, value in features.items()}
    out["task_index"] = {"dtype": "int64", "shape": (1,), "names": None}
    out["episode_index"] = {"dtype": "int64", "shape": (1,), "names": None}
    out["frame_index"] = {"dtype": "int64", "shape": (1,), "names": None}
    out["timestamp"] = {"dtype": "float32", "shape": (1,), "names": None}
    out["index"] = {"dtype": "int64", "shape": (1,), "names": None}
    return out


def _build_output_features(info: dict[str, Any], layout: InternA1Layout) -> dict[str, dict[str, Any]]:
    raw_features = info["features"]
    canonical_names = (
        EE_POSE_CANONICAL_STATE_NAMES
        if layout.state_semantics == "ee_pose"
        else JOINT_CANONICAL_STATE_NAMES
    )
    features: dict[str, dict[str, Any]] = {
        CANONICAL_CAMERA_KEYS[0]: _video_feature_from_raw(raw_features[layout.head_camera_key]),
        CANONICAL_CAMERA_KEYS[1]: _video_feature_from_raw(raw_features[layout.left_camera_key]),
        "observation.state": {
            "dtype": "float32",
            "shape": (CANONICAL_STATE_DIM,),
            "names": list(canonical_names),
        },
        "observation.state_mask": {
            "dtype": "bool",
            "shape": (CANONICAL_STATE_DIM,),
            "names": list(canonical_names),
        },
        "actions": {
            "dtype": "float32",
            "shape": (CANONICAL_ACTION_DIM,),
            "names": list(canonical_names),
        },
        "actions_mask": {
            "dtype": "bool",
            "shape": (CANONICAL_ACTION_DIM,),
            "names": list(canonical_names),
        },
        "observation.image_mask": {
            "dtype": "bool",
            "shape": (len(CANONICAL_IMAGE_MASK_NAMES),),
            "names": list(CANONICAL_IMAGE_MASK_NAMES),
        },
    }
    if layout.right_camera_key is not None:
        features[CANONICAL_CAMERA_KEYS[2]] = _video_feature_from_raw(raw_features[layout.right_camera_key])
    return _add_common_features(features)


def _load_task_lookup(dataset_dir: Path) -> dict[int, str]:
    task_lookup: dict[int, str] = {}
    with (dataset_dir / "meta" / "tasks.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            task_lookup[int(record["task_index"])] = str(record["task"])
    if not task_lookup:
        raise ValueError(f"No tasks found in {dataset_dir / 'meta' / 'tasks.jsonl'}")
    return task_lookup


def _episode_index_from_path(parquet_path: Path) -> int:
    return int(parquet_path.stem.split("_")[-1])


def _video_source_path(dataset_dir: Path, info: dict[str, Any], video_key: str, episode_index: int) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    video_path = info["video_path"].format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
        video_key=video_key,
    )
    return dataset_dir / video_path


def _canonical_video_sources(
    dataset_dir: Path,
    info: dict[str, Any],
    layout: InternA1Layout,
    episode_index: int,
) -> dict[str, Path]:
    raw_to_canonical = {
        layout.head_camera_key: CANONICAL_CAMERA_KEYS[0],
        layout.left_camera_key: CANONICAL_CAMERA_KEYS[1],
    }
    if layout.right_camera_key is not None:
        raw_to_canonical[layout.right_camera_key] = CANONICAL_CAMERA_KEYS[2]

    videos: dict[str, Path] = {}
    for raw_key, canonical_key in raw_to_canonical.items():
        source_path = _video_source_path(dataset_dir, info, raw_key, episode_index)
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source video for {canonical_key}: {source_path}")
        videos[canonical_key] = source_path
    return videos


def _validate_source_video_decodes(video_path: Path) -> None:
    resolved = str(video_path.expanduser().resolve())
    if resolved in _VALIDATED_SOURCE_VIDEO_PATHS:
        return

    try:
        with av.open(resolved, mode="r") as container:
            for _ in container.decode(video=0):
                pass
    except Exception as exc:
        raise RuntimeError(f"Source video failed full decode validation: {video_path}") from exc

    _VALIDATED_SOURCE_VIDEO_PATHS.add(resolved)


def _validate_episode_videos(videos: dict[str, Path]) -> None:
    # Decode the original per-episode assets up front so corrupt AV1 inputs are
    # rejected during conversion rather than after they have been packed into
    # longer shared shards and surfaced much later in training.
    for video_key, source_path in sorted(videos.items()):
        try:
            _validate_source_video_decodes(source_path)
        except Exception as exc:
            raise RuntimeError(f"Video validation failed for {video_key}: {source_path}") from exc


def _get_source_video_duration_s(video_path: Path) -> float:
    resolved = str(video_path.expanduser().resolve())
    cached = _SOURCE_VIDEO_DURATIONS_S.get(resolved)
    if cached is not None:
        return cached

    duration_s = float(get_video_duration_in_s(Path(resolved)))
    _SOURCE_VIDEO_DURATIONS_S[resolved] = duration_s
    return duration_s


def _compute_episode_video_durations(videos: dict[str, Path]) -> dict[str, float]:
    return {
        video_key: _get_source_video_duration_s(video_path)
        for video_key, video_path in sorted(videos.items())
    }


def _pack_canonical_vectors(
    *,
    left_state: np.ndarray,
    left_gripper: np.ndarray,
    right_state: np.ndarray | None = None,
    right_gripper: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    num_frames = left_state.shape[0]
    vectors = np.zeros((num_frames, CANONICAL_STATE_DIM), dtype=np.float32)
    mask = np.zeros((num_frames, CANONICAL_STATE_DIM), dtype=bool)

    if left_state.shape[1] > 7:
        raise ValueError(f"Left state dimension {left_state.shape[1]} exceeds canonical capacity.")
    vectors[:, : left_state.shape[1]] = left_state
    mask[:, : left_state.shape[1]] = True
    vectors[:, 7] = left_gripper[:, 0]
    mask[:, 7] = True

    if right_state is not None and right_gripper is not None:
        if right_state.shape[1] > 7:
            raise ValueError(f"Right state dimension {right_state.shape[1]} exceeds canonical capacity.")
        vectors[:, 8 : 8 + right_state.shape[1]] = right_state
        mask[:, 8 : 8 + right_state.shape[1]] = True
        vectors[:, 15] = right_gripper[:, 0]
        mask[:, 15] = True

    return vectors, mask


def _convert_episode_arrays(table: Any, layout: InternA1Layout) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left_state = column_to_matrix(table.column(layout.left_state_key))
    left_gripper_state = column_to_matrix(table.column(layout.left_gripper_key))
    right_state = right_gripper_state = None
    if layout.right_state_key is not None and layout.right_gripper_key is not None:
        right_state = column_to_matrix(table.column(layout.right_state_key))
        right_gripper_state = column_to_matrix(table.column(layout.right_gripper_key))
    state, state_mask = _pack_canonical_vectors(
        left_state=left_state,
        left_gripper=left_gripper_state,
        right_state=right_state,
        right_gripper=right_gripper_state,
    )

    left_actions = column_to_matrix(table.column(layout.left_action_key))
    left_gripper_actions = column_to_matrix(table.column(layout.left_gripper_action_key))

    right_actions = right_gripper_actions = None
    if layout.right_state_key is not None and layout.right_gripper_key is not None:
        right_actions = column_to_matrix(table.column(layout.right_action_key))
        right_gripper_actions = column_to_matrix(table.column(layout.right_gripper_action_key))

    actions, actions_mask = _pack_canonical_vectors(
        left_state=left_actions,
        left_gripper=left_gripper_actions,
        right_state=right_actions,
        right_gripper=right_gripper_actions,
    )
    return state, state_mask, actions, actions_mask


def _image_mask(num_frames: int, layout: InternA1Layout) -> np.ndarray:
    return np.tile(
        np.array(
            [
                True,
                layout.left_camera_key is not None,
                layout.right_camera_key is not None,
            ],
            dtype=bool,
        ),
        (num_frames, 1),
    )


class CanonicalInternA1Dataset(DirectVideoLeRobotDataset):

    def consolidate(self, run_compute_stats: bool = True) -> None:  # noqa: FBT001, FBT002
        del run_compute_stats
        self.finalize()


def load_committed_episode_count(output_root: Path) -> int | None:
    info_path = output_root / "meta" / "info.json"
    if not info_path.exists():
        return None

    info = json.loads(info_path.read_text(encoding="utf-8"))
    total_episodes = info.get("total_episodes")
    try:
        committed = int(total_episodes)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid total_episodes value in {info_path}: {total_episodes!r}") from exc

    if committed < 0:
        raise ValueError(f"Invalid negative total_episodes value in {info_path}: {committed}")
    return committed


def open_dataset_for_append(
    repo_id: str,
    output_root: Path,
    *,
    episodes: list[int] | None = None,
) -> CanonicalInternA1Dataset:
    return CanonicalInternA1Dataset(repo_id=repo_id, root=output_root, episodes=episodes)


def validate_resume_dataset(
    dataset: CanonicalInternA1Dataset,
    *,
    info: dict[str, Any],
    layout: InternA1Layout,
) -> None:
    expected_fps = int(info["fps"])
    if dataset.meta.fps != expected_fps:
        raise ValueError(
            f"Cannot resume {dataset.root}: expected fps={expected_fps}, found {dataset.meta.fps}."
        )
    if dataset.meta.robot_type != "intern_a1":
        raise ValueError(
            f"Cannot resume {dataset.root}: expected robot_type='intern_a1', found {dataset.meta.robot_type!r}."
        )

    expected_features = _build_output_features(info, layout)
    for key, expected in expected_features.items():
        actual = dataset.features.get(key)
        if actual is None:
            raise ValueError(f"Cannot resume {dataset.root}: missing expected feature {key!r}.")
        if actual["dtype"] != expected["dtype"] or tuple(actual["shape"]) != tuple(expected["shape"]):
            raise ValueError(
                f"Cannot resume {dataset.root}: feature {key!r} has dtype/shape "
                f"{actual['dtype']}/{tuple(actual['shape'])}, expected "
                f"{expected['dtype']}/{tuple(expected['shape'])}."
            )


def _write_norm_stats(
    dataset_root: Path,
    *,
    action_horizon: int,
    stats_batch_size: int,
    run_compute_stats: bool,
    norm_stats: dict[str, normalize.NormStats] | None = None,
) -> None:
    if not run_compute_stats:
        return
    if norm_stats is None:
        norm_stats = compute_direct_norm_stats(
            [dataset_root],
            action_horizon,
            batch_size=stats_batch_size,
        )
    normalize.save(dataset_root, norm_stats)

def resolve_episode_commit_batch_size(requested: int | None, *, workers: int, episode_count: int) -> int:
    if episode_count <= 0:
        return 1
    if requested is None:
        requested = min(max(8, workers), 16)
    return max(1, min(requested, episode_count))


def _prepare_episode(
    *,
    dataset_dir: Path,
    parquet_path: Path,
    info: dict[str, Any],
    layout: InternA1Layout,
    task_lookup: dict[int, str],
    validate_source_videos: bool = True,
) -> dict[str, Any]:
    table = pq.read_table(parquet_path)
    state, state_mask, actions, actions_mask = _convert_episode_arrays(table, layout)
    num_frames = state.shape[0]
    episode_index = _episode_index_from_path(parquet_path)

    task_indices = (
        _column_to_vector(table.column("task_index")).astype(np.int64)
        if "task_index" in table.schema.names
        else np.zeros(num_frames, dtype=np.int64)
    )
    tasks = [task_lookup[int(task_index)] for task_index in task_indices]

    frame_indices = (
        _column_to_vector(table.column("frame_index")).astype(np.int64)
        if "frame_index" in table.schema.names
        else np.arange(num_frames, dtype=np.int64)
    )
    timestamps = (
        _column_to_vector(table.column("timestamp")).astype(np.float32)
        if "timestamp" in table.schema.names
        else np.arange(num_frames, dtype=np.float32) / float(info["fps"])
    )
    image_mask = _image_mask(num_frames, layout)
    videos = _canonical_video_sources(dataset_dir, info, layout, episode_index)
    if validate_source_videos:
        _validate_episode_videos(videos)
    video_durations = _compute_episode_video_durations(videos)
    return {
        "episode_columns": {
            "size": num_frames,
            "task": tasks,
            "frame_index": frame_indices.astype(np.int64, copy=False),
            "timestamp": timestamps.astype(np.float32, copy=False),
            "observation.state": state,
            "observation.state_mask": state_mask,
            "actions": actions,
            "actions_mask": actions_mask,
            "observation.image_mask": image_mask,
        },
        "videos": videos,
        "video_durations": video_durations,
        "summary": {
            "raw_episode_index": episode_index,
            "frame_count": num_frames,
            "task": tasks[0] if tasks else "",
        },
    }


def _save_prepared_episode_batch(
    dataset: CanonicalInternA1Dataset,
    prepared_batch: list[dict[str, Any]],
    *,
    link_mode: MergeLinkMode = "auto",
    passthrough_video_workers: int | None = None,
    probe_video_durations: bool = False,
    on_episode_saved: Callable[[int, int], None] | None = None,
) -> None:
    if not prepared_batch:
        return
    start_episode_index = dataset.meta.total_episodes
    episode_batch = []
    for offset, item in enumerate(prepared_batch):
        columns = dict(item["episode_columns"])
        episode_data = dataset.create_episode_buffer(start_episode_index + offset)
        episode_data["size"] = int(columns.pop("size"))
        episode_data["task"] = list(columns.pop("task"))
        for key, value in columns.items():
            episode_data[key] = value
        episode_batch.append(episode_data)
    dataset.save_episode_batch(
        episode_batch,
        videos=[item["videos"] for item in prepared_batch],
        video_durations=[dict(item["video_durations"]) for item in prepared_batch],
        link_mode=link_mode,
        passthrough_video_workers=passthrough_video_workers,
        probe_video_durations=probe_video_durations,
        on_episode_saved=on_episode_saved,
    )

def _convert_dataset(
    dataset_dir: Path,
    *,
    input_root: Path,
    output_root: Path,
    action_horizon: int,
    stats_batch_size: int,
    overwrite: bool,
    run_compute_stats: bool,
    write_task_norm_stats: bool,
    validate_source_videos: bool,
    max_episodes_per_dataset: int | None,
    show_progress: bool,
    episode_commit_batch_size: int | None = None,
) -> dict[str, Any]:
    relative_path = dataset_dir.resolve().relative_to(input_root.resolve())
    target_root = output_root / relative_path
    embodiment = infer_embodiment(relative_path)

    if (target_root / "meta" / "info.json").exists():
        if not overwrite:
            return {
                "dataset_dir": str(dataset_dir),
                "output_dir": str(target_root),
                "status": "skipped",
                "embodiment": embodiment,
                "episodes_written": 0,
                "frames_written": 0,
            }
        shutil.rmtree(target_root)

    info = load_dataset_info(dataset_dir)
    layout = infer_layout_from_info(info, dataset_dir=dataset_dir)
    task_lookup = _load_task_lookup(dataset_dir)
    parquet_paths = sorted(dataset_dir.glob("data/chunk-*/episode_*.parquet"))
    if max_episodes_per_dataset is not None:
        parquet_paths = parquet_paths[: max_episodes_per_dataset]

    dataset = CanonicalInternA1Dataset.create(
        repo_id=relative_path.as_posix(),
        root=target_root,
        fps=int(info["fps"]),
        robot_type="intern_a1",
        features=_build_output_features(info, layout),
    )

    episode_summaries: list[dict[str, Any]] = []
    commit_batch_size = resolve_episode_commit_batch_size(
        episode_commit_batch_size,
        workers=os.cpu_count() or 1,
        episode_count=len(parquet_paths),
    )
    prepared_batch: list[dict[str, Any]] = []
    output_norm_stats = init_output_norm_stats() if run_compute_stats and write_task_norm_stats else None
    for parquet_path in tqdm(parquet_paths, desc=f"Converting {relative_path.as_posix()}", disable=not show_progress):
        prepared_batch.append(
            _prepare_episode(
                dataset_dir=dataset_dir,
                parquet_path=parquet_path,
                info=info,
                layout=layout,
                task_lookup=task_lookup,
                validate_source_videos=validate_source_videos,
            )
        )
        while len(prepared_batch) >= commit_batch_size:
            batch = prepared_batch[:commit_batch_size]
            del prepared_batch[:commit_batch_size]
            _save_prepared_episode_batch(dataset, batch)
            episode_summaries.extend(dict(item["summary"]) for item in batch)
            if output_norm_stats is not None:
                for item in batch:
                    update_output_norm_stats_from_prepared_episode(
                        output_norm_stats,
                        item,
                        action_horizon=action_horizon,
                    )

    if prepared_batch:
        _save_prepared_episode_batch(dataset, prepared_batch)
        episode_summaries.extend(dict(item["summary"]) for item in prepared_batch)
        if output_norm_stats is not None:
            for item in prepared_batch:
                update_output_norm_stats_from_prepared_episode(
                    output_norm_stats,
                    item,
                    action_horizon=action_horizon,
                )

    dataset.consolidate(run_compute_stats=run_compute_stats)
    finalized_norm_stats = finalize_output_norm_stats(output_norm_stats) if output_norm_stats is not None else None
    _write_norm_stats(
        target_root,
        action_horizon=action_horizon,
        stats_batch_size=stats_batch_size,
        run_compute_stats=write_task_norm_stats,
        norm_stats=finalized_norm_stats,
    )
    return {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(target_root),
        "status": "converted",
        "embodiment": layout.embodiment,
        "episodes_written": len(episode_summaries),
        "frames_written": sum(summary["frame_count"] for summary in episode_summaries),
        "episode_summaries": episode_summaries,
    }
