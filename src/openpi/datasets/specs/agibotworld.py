# Copyright 2026 PLaW-VLA authors.
# SPDX-License-Identifier: Apache-2.0 AND MIT
# AgiBot loading/configuration portions adapted from Any4LeRobot.
# Copyright (c) 2025 Qizhi Chen (MIT).
# Modified for PLaW-VLA; see NOTICE and LICENSES/MIT-Any4LeRobot.txt.
"""Shared AgiBot-World raw-data helpers for direct LeRobot v3 builders."""

from __future__ import annotations

import dataclasses
import gc
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

try:
    import h5py
except ModuleNotFoundError:
    h5py = None
from lerobot.datasets.video_utils import get_video_duration_in_s
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

from openpi.datasets.common.lerobot_v3 import DirectVideoLeRobotDataset

HEAD_COLOR = "head_color.mp4"
HAND_LEFT_COLOR = "hand_left_color.mp4"
HAND_RIGHT_COLOR = "hand_right_color.mp4"
HAND_LEFT_FISHEYE_COLOR = "hand_left_fisheye_color.mp4"
HAND_RIGHT_FISHEYE_COLOR = "hand_right_fisheye_color.mp4"
HEAD_CENTER_FISHEYE_COLOR = "head_center_fisheye_color.mp4"
HEAD_LEFT_FISHEYE_COLOR = "head_left_fisheye_color.mp4"
HEAD_RIGHT_FISHEYE_COLOR = "head_right_fisheye_color.mp4"
BACK_LEFT_FISHEYE_COLOR = "back_left_fisheye_color.mp4"
BACK_RIGHT_FISHEYE_COLOR = "back_right_fisheye_color.mp4"

GRIPPER_CANONICAL_STATE_DIM = 16
GRIPPER_CANONICAL_ACTION_DIM = 16
GRIPPER_CANONICAL_VECTOR_NAMES = (
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
VIDEO_INFO = {
    "video.fps": 30.0,
    "video.codec": "av1",
    "video.pix_fmt": "yuv420p",
    "video.is_depth_map": False,
    "has_audio": False,
}


GRIPPER_CONFIG = {
    "images": {
        "head": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "rgb"]},
        "hand_left": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "rgb"]},
        "hand_right": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "rgb"]},
        "head_center_fisheye": {"dtype": "video", "shape": (748, 960, 3), "names": ["height", "width", "rgb"]},
        "head_left_fisheye": {"dtype": "video", "shape": (748, 960, 3), "names": ["height", "width", "rgb"]},
        "head_right_fisheye": {"dtype": "video", "shape": (748, 960, 3), "names": ["height", "width", "rgb"]},
        "back_left_fisheye": {"dtype": "video", "shape": (748, 960, 3), "names": ["height", "width", "rgb"]},
        "back_right_fisheye": {"dtype": "video", "shape": (748, 960, 3), "names": ["height", "width", "rgb"]},
        "head_depth": {"dtype": "image", "shape": (480, 640, 1), "names": ["height", "width", "channel"]},
    },
    "states": {
        "end.position": {"dtype": "float32", "shape": (2, 3), "names": None},
        "end.orientation": {"dtype": "float32", "shape": (2, 4), "names": None},
        "effector.position": {"dtype": "float32", "shape": (2,), "names": None},
    },
    "actions": {
        "end.position": {"dtype": "float32", "shape": (2, 3), "names": None},
        "end.orientation": {"dtype": "float32", "shape": (2, 4), "names": None},
        "effector.position": {"dtype": "float32", "shape": (2,), "names": None},
    },
}

DEXHAND_CONFIG = {
    "images": {
        **{k: v for k, v in GRIPPER_CONFIG["images"].items() if k not in {"hand_left", "hand_right"}},
        "hand_left_fisheye": {"dtype": "video", "shape": (748, 960, 3), "names": ["height", "width", "rgb"]},
        "hand_right_fisheye": {"dtype": "video", "shape": (748, 960, 3), "names": ["height", "width", "rgb"]},
        "head_depth": GRIPPER_CONFIG["images"]["head_depth"],
    },
    "states": {
        "joint.position": {"dtype": "float32", "shape": (14,), "names": None},
        "effector.position": {"dtype": "float32", "shape": (12,), "names": None},
        "head.position": {"dtype": "float32", "shape": (2,), "names": None},
        "waist.position": {"dtype": "float32", "shape": (2,), "names": None},
    },
    "actions": {
        "joint.position": {"dtype": "float32", "shape": (14,), "names": None},
        "effector.position": {"dtype": "float32", "shape": (12,), "names": None},
        "head.position": {"dtype": "float32", "shape": (2,), "names": None},
        "waist.position": {"dtype": "float32", "shape": (2,), "names": None},
        "robot.velocity": {"dtype": "float32", "shape": (2,), "names": None},
    },
}

AGIBOTWORLD_TASK_CONFIGS = {
    "gripper": GRIPPER_CONFIG,
    "dexhand": DEXHAND_CONFIG,
}

VIDEO_FILENAMES = {
    "head": HEAD_COLOR,
    "hand_left": HAND_LEFT_COLOR,
    "hand_right": HAND_RIGHT_COLOR,
    "hand_left_fisheye": HAND_LEFT_FISHEYE_COLOR,
    "hand_right_fisheye": HAND_RIGHT_FISHEYE_COLOR,
    "head_center_fisheye": HEAD_CENTER_FISHEYE_COLOR,
    "head_left_fisheye": HEAD_LEFT_FISHEYE_COLOR,
    "head_right_fisheye": HEAD_RIGHT_FISHEYE_COLOR,
    "back_left_fisheye": BACK_LEFT_FISHEYE_COLOR,
    "back_right_fisheye": BACK_RIGHT_FISHEYE_COLOR,
}

@dataclasses.dataclass(frozen=True)
class TaskSpec:
    json_file: Path
    task_stem: str
    task_id: str


@dataclasses.dataclass(frozen=True)
class EpisodeSelection:
    episode_id: int
    task_name: str
    init_scene_text: str
    action_config: list[dict[str, Any]]


def _normalize_task_id(task_id: str) -> str:
    task_id = str(task_id).strip()
    if not task_id:
        raise ValueError("Task id cannot be empty.")
    return task_id if task_id.startswith("task_") else f"task_{task_id}"


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, tuple | list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


class RunningNormStats:
    def __init__(self) -> None:
        self._count = 0
        self._mean: np.ndarray | None = None
        self._mean_of_squares: np.ndarray | None = None
        self._min: np.ndarray | None = None
        self._max: np.ndarray | None = None
        self._histograms: list[np.ndarray] | None = None
        self._bin_edges: list[np.ndarray] | None = None
        self._num_quantile_bins = 5000

    def update(self, batch: np.ndarray) -> None:
        batch = np.asarray(batch, dtype=np.float64)
        batch = batch.reshape(-1, batch.shape[-1])
        num_elements, vector_length = batch.shape
        if num_elements == 0:
            return

        if self._count == 0:
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch**2, axis=0)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = [np.zeros(self._num_quantile_bins, dtype=np.float64) for _ in range(vector_length)]
            self._bin_edges = [
                np.linspace(self._min[i] - 1e-10, self._max[i] + 1e-10, self._num_quantile_bins + 1)
                for i in range(vector_length)
            ]
        else:
            if self._mean is None or self._mean_of_squares is None or self._min is None or self._max is None:
                raise ValueError("RunningNormStats is in an invalid partially initialized state.")
            if vector_length != self._mean.size:
                raise ValueError("The length of new vectors does not match the initialized vector length.")
            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            max_changed = np.any(new_max > self._max)
            min_changed = np.any(new_min < self._min)
            self._max = np.maximum(self._max, new_max)
            self._min = np.minimum(self._min, new_min)
            if max_changed or min_changed:
                self._adjust_histograms()

        self._count += num_elements
        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)
        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (num_elements / self._count)
        self._update_histograms(batch)

    def to_dict(self) -> dict[str, list[float]]:
        if self._count < 2 or self._mean is None or self._mean_of_squares is None:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")
        variance = self._mean_of_squares - self._mean**2
        stddev = np.sqrt(np.maximum(0, variance))
        q01, q99 = self._compute_quantiles([0.01, 0.99])
        return {
            "mean": np.asarray(self._mean, dtype=np.float32).tolist(),
            "std": np.asarray(stddev, dtype=np.float32).tolist(),
            "q01": np.asarray(q01, dtype=np.float32).tolist(),
            "q99": np.asarray(q99, dtype=np.float32).tolist(),
        }

    def _adjust_histograms(self) -> None:
        if self._histograms is None or self._bin_edges is None or self._min is None or self._max is None:
            raise ValueError("Histogram state is not initialized.")
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            new_edges = np.linspace(self._min[i], self._max[i], self._num_quantile_bins + 1)
            new_hist, _ = np.histogram(old_edges[:-1], bins=new_edges, weights=self._histograms[i])
            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def _update_histograms(self, batch: np.ndarray) -> None:
        if self._histograms is None or self._bin_edges is None:
            raise ValueError("Histogram state is not initialized.")
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _compute_quantiles(self, quantiles: list[float]) -> list[np.ndarray]:
        if self._histograms is None or self._bin_edges is None:
            raise ValueError("Histogram state is not initialized.")
        results: list[np.ndarray] = []
        for q in quantiles:
            target_count = q * self._count
            q_values: list[float] = []
            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                cumsum = np.cumsum(hist)
                idx = int(np.searchsorted(cumsum, target_count))
                idx = min(idx, len(edges) - 1)
                q_values.append(float(edges[idx]))
            results.append(np.array(q_values, dtype=np.float64))
        return results


def _add_common_features(features: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {key: dict(value) for key, value in features.items()}
    out["task_index"] = {"dtype": "int64", "shape": (1,), "names": None}
    out["episode_index"] = {"dtype": "int64", "shape": (1,), "names": None}
    out["frame_index"] = {"dtype": "int64", "shape": (1,), "names": None}
    out["timestamp"] = {"dtype": "float32", "shape": (1,), "names": None}
    out["index"] = {"dtype": "int64", "shape": (1,), "names": None}
    return out


def _video_info_features(features: dict[str, dict[str, Any]], fps: int) -> dict[str, dict[str, Any]]:
    enriched: dict[str, dict[str, Any]] = {}
    for key, value in features.items():
        feat = dict(value)
        if feat["dtype"] == "video":
            feat["video_info"] = dict(VIDEO_INFO)
            feat["video_info"]["video.fps"] = float(fps)
        # Preserve scalar `(1,)` shapes so LeRobot maps them to HF scalar values
        # instead of length-1 sequences.
        if isinstance(feat.get("shape"), tuple) and feat["shape"] != (1,):
            feat["shape"] = list(feat["shape"])
        enriched[key] = feat
    return enriched


def _get_all_tasks(src_path: Path) -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    for json_file in sorted(src_path.glob("task_info/task_*.json")):
        task_stem = json_file.stem
        task_id = task_stem.split("_", 1)[1]
        tasks.append(TaskSpec(json_file=json_file, task_stem=task_stem, task_id=task_id))
    return tasks


def _load_task_id_file(task_id_file: Path | None) -> list[str]:
    if task_id_file is None:
        return []
    task_ids: list[str] = []
    with open(task_id_file) as f:
        for raw_line in f:
            line = raw_line.strip()
            if line:
                task_ids.append(_normalize_task_id(line))
    return task_ids


def _load_task_info(task_json_path: Path) -> list[dict[str, Any]]:
    with open(task_json_path) as f:
        task_info = json.load(f)
    task_info.sort(key=lambda episode: episode["episode_id"])
    return task_info


def _select_episodes(
    task_info_list: list[dict[str, Any]],
    *,
    episode_offset: int,
    episodes_per_task: int | None,
) -> list[EpisodeSelection]:
    selected = task_info_list[episode_offset:]
    if episodes_per_task is not None:
        selected = selected[:episodes_per_task]
    return [
        EpisodeSelection(
            episode_id=int(record["episode_id"]),
            task_name=str(record["task_name"]),
            init_scene_text=str(record["init_scene_text"]),
            action_config=list(record.get("label_info", {}).get("action_config", [])),
        )
        for record in selected
    ]


def _load_depth_paths(depth_dir: Path) -> list[Path]:
    return sorted(depth_dir.glob("head_depth*"))


def _video_path_for_key(ob_dir: Path, key: str) -> Path:
    if key not in VIDEO_FILENAMES:
        raise KeyError(f"Unsupported video key: {key}")
    return ob_dir / "videos" / VIDEO_FILENAMES[key]


def _gripper_widths_to_closed_fraction(
    grippers: np.ndarray,
    *,
    gripper_max_width_mm: tuple[float, float] | None = None,
) -> np.ndarray:
    """Map sensor opening in millimetres to the raw command's 0=open, 1=close convention."""
    grippers = np.asarray(grippers, dtype=np.float32)
    if grippers.ndim == 3 and grippers.shape[-1] == 1:
        grippers = grippers[..., 0]
    if grippers.ndim != 2 or grippers.shape[1] != 2:
        raise ValueError(f"Expected bimanual grippers with shape [T, 2], got {grippers.shape}.")
    if not np.isfinite(grippers).all():
        raise ValueError("AgiBot gripper widths must be finite.")
    # Without calibration, the largest observed opening in each episode defines
    # its scale. This preserves a zero-width closed gripper and a constant open
    # gripper, but cannot identify an episode's unobserved fully open width.
    maximum = (
        np.asarray(gripper_max_width_mm, dtype=np.float32)
        if gripper_max_width_mm is not None
        else np.max(grippers, axis=0)
    )
    if maximum.shape != (2,) or not np.isfinite(maximum).all() or np.any(maximum < 0):
        raise ValueError("Expected two finite non-negative gripper maximum widths.")
    if gripper_max_width_mm is not None and np.any(maximum <= 0):
        raise ValueError("Calibrated gripper maximum widths must be positive.")
    openness = np.clip(grippers / np.maximum(maximum, 1e-6), 0.0, 1.0)
    return (1.0 - openness).astype(np.float32)


def _load_episode_arrays(
    episode_id: int,
    src_path: Path,
    task_id: str,
    task_config: dict[str, Any],
    *,
    save_depth: bool,
    gripper_max_width_mm: tuple[float, float] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[Path], dict[str, Path]]:
    ob_dir = src_path / "observations" / task_id / str(episode_id)
    proprio_dir = src_path / "proprio_stats" / task_id / str(episode_id)

    state: dict[str, np.ndarray] = {}
    action: dict[str, np.ndarray] = {}
    if h5py is None:
        raise ModuleNotFoundError("h5py is required to load raw AgiBotWorld proprio stats.")
    with h5py.File(proprio_dir / "proprio_stats.h5", "r") as f:
        for key in task_config["states"]:
            state[f"observation.states.{key}"] = np.array(f["state/" + key.replace(".", "/")], dtype=np.float32)
        for key in task_config["actions"]:
            action[f"actions.{key}"] = np.array(f["action/" + key.replace(".", "/")], dtype=np.float32)

        num_frames = len(next(iter(state.values())))
        for state_key, state_value in state.items():
            if len(state_value) != num_frames:
                raise ValueError(
                    f"Corrupt data for episode {episode_id}: state {state_key} has {len(state_value)} rows, "
                    f"expected {num_frames}."
                )
        for action_key, action_value in list(action.items()):
            if len(action_value) < num_frames:
                state_key = action_key.replace("actions", "state").replace(".", "/")
                # Sensor widths and gripper commands have different units. Fill
                # the initial interval in command units, then hold the latest
                # command until the next indexed control signal.
                is_gripper = (
                    action_key == "actions.effector.position"
                    and tuple(task_config["actions"]["effector.position"]["shape"]) == (2,)
                )
                if is_gripper:
                    padded = _gripper_widths_to_closed_fraction(
                        state["observation.states.effector.position"],
                        gripper_max_width_mm=gripper_max_width_mm,
                    )
                elif state_key in f:
                    padded = np.array(f[state_key], dtype=np.float32).copy()
                elif action_key == "actions.robot.velocity":
                    padded = np.zeros((num_frames, *action_value.shape[1:]), dtype=np.float32)
                else:
                    raise ValueError(f"Cannot align sparse action {action_key}: missing matching sensor state.")
                if not len(action_value):
                    action[action_key] = padded
                    continue
                action_index_key = "/".join(
                    [*action_key.replace("actions", "action").split(".")[:-1], "index"]
                )
                if action_index_key not in f or not f[action_index_key].size:
                    action_index_key = action_index_key.replace("/end/", "/joint/")
                if action_index_key not in f:
                    raise ValueError(f"Cannot align sparse action {action_key}: missing command index.")
                action_index = np.asarray(f[action_index_key]).reshape(-1)
                if (
                    len(action_index) != len(action_value)
                    or not np.issubdtype(action_index.dtype, np.integer)
                    or np.any(action_index < 0)
                    or np.any(action_index >= num_frames)
                    or np.any(action_index[1:] <= action_index[:-1])
                ):
                    raise ValueError(
                        f"Sparse action {action_key} requires one strictly increasing in-range index per command."
                    )
                if is_gripper:
                    command_rows = np.searchsorted(action_index, np.arange(num_frames), side="right") - 1
                    has_command = command_rows >= 0
                    padded[has_command] = action_value[command_rows[has_command]]
                else:
                    padded[action_index] = action_value
                action[action_key] = padded
            elif len(action_value) > num_frames:
                raise ValueError(
                    f"Corrupt data for episode {episode_id}: action {action_key} has {len(action_value)} rows, "
                    f"expected <= {num_frames}."
                )

    depth_paths = _load_depth_paths(ob_dir / "depth") if save_depth else []
    if save_depth and len(depth_paths) != num_frames:
        raise ValueError(
            f"Episode {episode_id} depth frame mismatch: found {len(depth_paths)} depth images, expected {num_frames}."
        )

    videos = {
        f"observation.images.{key}": _video_path_for_key(ob_dir, key)
        for key in task_config["images"]
        if "depth" not in key
    }
    return state, action, depth_paths, videos
def _compute_episode_video_durations(
    videos: dict[str, Path],
    *,
    num_frames: int,
    fps: int,
    episode_id: int,
) -> dict[str, float]:
    del num_frames, fps, episode_id
    return {
        video_key: float(get_video_duration_in_s(video_path))
        for video_key, video_path in sorted(videos.items())
    }


def _derive_gripper_state_and_actions(
    state_arrays: dict[str, np.ndarray],
    action_arrays: dict[str, np.ndarray],
    *,
    gripper_max_width_mm: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    def _pack_bimanual_pose_gripper(
        positions: np.ndarray,
        orientations: np.ndarray,
        grippers: np.ndarray,
    ) -> np.ndarray:
        positions = np.asarray(positions, dtype=np.float32)
        orientations = np.asarray(orientations, dtype=np.float32)
        grippers = np.asarray(grippers, dtype=np.float32)
        if grippers.ndim == 3 and grippers.shape[-1] == 1:
            grippers = grippers[..., 0]
        if positions.ndim != 3 or positions.shape[1:] != (2, 3):
            raise ValueError(f"Expected bimanual positions with shape [T, 2, 3], got {positions.shape}.")
        if orientations.ndim != 3 or orientations.shape[1:] != (2, 4):
            raise ValueError(f"Expected bimanual orientations with shape [T, 2, 4], got {orientations.shape}.")
        quaternion_norm = np.linalg.norm(orientations, axis=-1, keepdims=True)
        if not np.isfinite(orientations).all() or np.any(quaternion_norm <= 1e-8):
            raise ValueError("AgiBot source orientations must be finite nonzero xyzw quaternions.")
        # The official raw HDF5 schema stores xyzw; the shared EEF layout is wxyz.
        orientations = (orientations / quaternion_norm)[..., [3, 0, 1, 2]]
        if grippers.ndim != 2 or grippers.shape[1] != 2:
            raise ValueError(f"Expected bimanual grippers with shape [T, 2], got {grippers.shape}.")
        return np.concatenate(
            [
                positions[:, 0, :],
                orientations[:, 0, :],
                grippers[:, 0:1],
                positions[:, 1, :],
                orientations[:, 1, :],
                grippers[:, 1:2],
            ],
            axis=1,
        ).astype(np.float32)

    state_value = _pack_bimanual_pose_gripper(
        state_arrays["observation.states.end.position"],
        state_arrays["observation.states.end.orientation"],
        _gripper_widths_to_closed_fraction(
            state_arrays["observation.states.effector.position"], gripper_max_width_mm=gripper_max_width_mm
        ),
    )
    action_value = _pack_bimanual_pose_gripper(
        action_arrays["actions.end.position"],
        action_arrays["actions.end.orientation"],
        np.clip(np.asarray(action_arrays["actions.effector.position"], dtype=np.float32), 0.0, 1.0),
    )
    if state_value.shape[1] != GRIPPER_CANONICAL_STATE_DIM:
        raise ValueError(
            f"Derived state dimension mismatch: got {state_value.shape[1]}, expected {GRIPPER_CANONICAL_STATE_DIM}."
        )
    if action_value.shape[1] != GRIPPER_CANONICAL_ACTION_DIM:
        raise ValueError(
            f"Derived action dimension mismatch: got {action_value.shape[1]}, expected {GRIPPER_CANONICAL_ACTION_DIM}."
        )
    return state_value, action_value


def _build_frame_labels(
    *,
    num_frames: int,
    action_config: list[dict[str, Any]],
    coarse_task_text: str,
) -> tuple[list[str], int]:
    coarse_task_text = coarse_task_text.strip()
    if not coarse_task_text:
        raise ValueError("coarse_task_text cannot be empty.")

    tasks = [coarse_task_text] * num_frames

    sorted_spans = sorted(action_config, key=lambda span: (int(span["start_frame"]), int(span["end_frame"])))
    previous_end = 0
    for span in sorted_spans:
        start = int(span["start_frame"])
        end = int(span["end_frame"])
        action_text = str(span["action_text"]).strip()

        if not action_text:
            raise ValueError("Encountered empty action_text in action_config.")
        if start < previous_end:
            raise ValueError(f"Action spans overlap or are unsorted: previous_end={previous_end}, start={start}.")
        if start < 0 or start >= num_frames:
            raise ValueError(f"Action span start_frame out of range: {start} for {num_frames} frames.")
        if end <= start or end > num_frames:
            raise ValueError(f"Action span end_frame out of range: start={start}, end={end}, num_frames={num_frames}.")

        for frame_idx in range(start, end):
            tasks[frame_idx] = action_text
        previous_end = end

    unlabeled_count = sum(task == coarse_task_text for task in tasks)
    return tasks, unlabeled_count


def _build_episode_columns(
    *,
    state_arrays: dict[str, np.ndarray],
    action_arrays: dict[str, np.ndarray],
    depth_paths: list[Path],
    task_config: dict[str, Any],
    eef_type: str,
    save_depth: bool,
    tasks_per_frame: list[str],
    fps: int,
) -> dict[str, Any]:
    num_frames = len(tasks_per_frame)
    columns: dict[str, Any] = {
        "task": tasks_per_frame,
        "frame_index": list(range(num_frames)),
        "timestamp": [frame_idx / fps for frame_idx in range(num_frames)],
    }

    if eef_type == "gripper":
        state_value, action_value = _derive_gripper_state_and_actions(state_arrays, action_arrays)
        if len(state_value) != num_frames or len(action_value) != num_frames:
            raise ValueError("Derived state/action length mismatch with labels.")
        columns["observation.state"] = state_value
        columns["actions"] = action_value
    else:
        for key, value in state_arrays.items():
            if len(value) != num_frames:
                raise ValueError(f"State length mismatch for {key}: {len(value)} vs {num_frames}.")
            columns[key] = value
        for key, value in action_arrays.items():
            action_value = value
            if action_value.size == 0:
                shape = task_config["actions"][key.replace("actions.", "")]["shape"]
                action_value = np.zeros((num_frames, *shape), dtype=np.float32)
            if len(action_value) != num_frames:
                raise ValueError(f"Action length mismatch for {key}: {len(action_value)} vs {num_frames}.")
            columns[key] = action_value

    if save_depth:
        columns["observation.images.head_depth"] = depth_paths
    return columns


class AgiBotWorldDataset(DirectVideoLeRobotDataset):

    def consolidate(self, *, run_compute_stats: bool = True) -> None:
        del run_compute_stats
        self.finalize()


def _column_to_numpy(column: Any) -> np.ndarray:
    array = column.combine_chunks() if hasattr(column, "combine_chunks") else column
    if hasattr(array, "to_numpy"):
        try:
            return np.asarray(array.to_numpy(zero_copy_only=False))
        except TypeError:
            return np.asarray(array.to_numpy())
    return np.asarray(array.to_pylist())


def _column_to_matrix(column: Any) -> np.ndarray:
    values = _column_to_numpy(column)
    if values.dtype == object:
        values = np.asarray(values.tolist(), dtype=np.float32)
    return np.asarray(values, dtype=np.float32)


def _compute_norm_stats_from_dataset(dataset_root: Path) -> dict[str, dict[str, list[float]]]:
    accumulators = {
        "state": RunningNormStats(),
        "actions": RunningNormStats(),
    }
    available_keys = {
        "state": "observation.state",
        "actions": "actions",
    }
    seen: set[str] = set()

    for parquet_path in sorted(dataset_root.glob("data/chunk-*/file-*.parquet")):
        table = pq.read_table(parquet_path)
        for out_key, parquet_key in available_keys.items():
            if parquet_key not in table.schema.names:
                continue
            accumulators[out_key].update(_column_to_matrix(table.column(parquet_key)))
            seen.add(out_key)

    return {key: accumulators[key].to_dict() for key in sorted(seen)}


def _write_norm_stats(dataset_root: Path, *, run_compute_stats: bool) -> None:
    if not run_compute_stats:
        return
    norm_stats = _compute_norm_stats_from_dataset(dataset_root)
    if not norm_stats:
        return
    with open(dataset_root / "norm_stats.json", "w") as f:
        json.dump({"norm_stats": norm_stats}, f, indent=2)


def _write_dataset_info_labels(
    dataset_root: Path, *, eef_type: str, gripper_max_width_mm: tuple[float, float] | None = None
) -> None:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Dataset info.json not found at {info_path}")
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    info["embodiment"] = eef_type
    info["agibot_eef_type"] = eef_type
    if eef_type == "gripper":
        info["agibot_eef_conversion_version"] = 2
        info["agibot_quaternion_order"] = "wxyz"
        info["agibot_state_gripper_format"] = "closed_fraction"
        info["agibot_action_gripper_format"] = "closed_fraction"
        info["agibot_gripper_max_width_mm"] = list(gripper_max_width_mm) if gripper_max_width_mm is not None else None
    with info_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def _set_temp_environment(tmp_dir: Path) -> None:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmp_dir)
    os.environ["TMP"] = str(tmp_dir)
    os.environ["TEMP"] = str(tmp_dir)
    tempfile.tempdir = str(tmp_dir)


def _convert_episode(
    *,
    dataset: AgiBotWorldDataset,
    selection: EpisodeSelection,
    task_config: dict[str, Any],
    src_path: Path,
    task_id: str,
    eef_type: str,
    save_depth: bool,
    fps: int,
) -> dict[str, Any]:
    state_arrays, action_arrays, depth_paths, videos = _load_episode_arrays(
        selection.episode_id,
        src_path,
        task_id,
        task_config,
        save_depth=save_depth,
    )
    num_frames = len(next(iter(state_arrays.values())))
    tasks_per_frame, unlabeled_frames = _build_frame_labels(
        num_frames=num_frames,
        action_config=selection.action_config,
        coarse_task_text=selection.task_name or selection.init_scene_text,
    )
    columns = _build_episode_columns(
        state_arrays=state_arrays,
        action_arrays=action_arrays,
        depth_paths=depth_paths,
        task_config=task_config,
        eef_type=eef_type,
        save_depth=save_depth,
        tasks_per_frame=tasks_per_frame,
        fps=fps,
    )
    video_durations = _compute_episode_video_durations(
        videos,
        num_frames=num_frames,
        fps=fps,
        episode_id=selection.episode_id,
    )

    episode_index = dataset.meta.total_episodes
    for frame_idx in range(num_frames):
        frame = {"task": columns["task"][frame_idx]}
        frame["frame_index"] = columns["frame_index"][frame_idx]
        frame["timestamp"] = columns["timestamp"][frame_idx]
        for key, values in columns.items():
            if key in {"task", "frame_index", "timestamp"}:
                continue
            frame[key] = values[frame_idx]
        dataset.add_frame(frame)

    dataset.save_episode(
        videos=videos,
        video_durations=video_durations,
        probe_video_durations=False,
    )
    return {
        "task_name": selection.task_name,
        "init_scene_text": selection.init_scene_text,
        "episode_id": selection.episode_id,
        "episode_index": episode_index,
        "frame_count": num_frames,
        "episode_tasks": list(dict.fromkeys(tasks_per_frame)),
        "unlabeled_frames": unlabeled_frames,
    }


def _convert_task_dataset(
    task: TaskSpec,
    *,
    src_path: Path,
    output_root: Path,
    repo_id: str,
    task_config: dict[str, Any],
    features: dict[str, dict[str, Any]],
    eef_type: str,
    save_depth: bool,
    run_compute_stats: bool,
    episode_offset: int,
    episodes_per_task: int | None,
    fps: int,
) -> dict[str, Any]:
    selections = _select_episodes(
        _load_task_info(task.json_file),
        episode_offset=episode_offset,
        episodes_per_task=episodes_per_task,
    )

    if output_root.exists():
        shutil.rmtree(output_root)

    dataset = AgiBotWorldDataset.create(
        repo_id=repo_id,
        root=output_root,
        fps=fps,
        robot_type="a2d",
        features=features,
    )

    episode_summaries: list[dict[str, Any]] = []
    for selection in tqdm(selections, desc=f"Converting {task.task_stem}"):
        episode_summaries.append(
            _convert_episode(
                dataset=dataset,
                selection=selection,
                task_config=task_config,
                src_path=src_path,
                task_id=task.task_id,
                eef_type=eef_type,
                save_depth=save_depth,
                fps=fps,
            )
        )
        gc.collect()

    dataset.consolidate(run_compute_stats=run_compute_stats)
    _write_dataset_info_labels(output_root, eef_type=eef_type)
    _write_norm_stats(output_root, run_compute_stats=run_compute_stats)
    return {
        "task_stem": task.task_stem,
        "task_id": task.task_id,
        "repo_id": repo_id,
        "output_path": str(output_root),
        "episodes_available": len(_load_task_info(task.json_file)),
        "episodes_selected": len(selections),
        "episodes_written": len(episode_summaries),
        "frames_written": sum(summary["frame_count"] for summary in episode_summaries),
        "unlabeled_frames": sum(summary["unlabeled_frames"] for summary in episode_summaries),
        "episode_summaries": episode_summaries,
    }
