# Copyright 2026 PLaW-VLA authors. SPDX-License-Identifier: Apache-2.0
"""Shared EgoDex raw-data helpers for direct LeRobot v3 builders."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import dataclasses
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
from typing import Any

from lerobot.datasets.video_utils import get_video_duration_in_s
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: F401
from lerobot.utils.constants import HF_LEROBOT_HOME
import numpy as np
from tqdm import tqdm

from openpi.datasets.common.lerobot_v3 import DirectVideoLeRobotDataset
from openpi.datasets.common.lerobot_v3 import MergeLinkMode
from openpi.shared import normalize

REPO_NAME = "egodex"
FPS = 30
IMAGE_SIZE = (256, 256)
DEFAULT_OUTPUT_PATH = Path("data/pretrain/egodex")
DEFAULT_NUM_WORKERS = max(1, os.cpu_count() or 1)
STATE_DIM = 48
ACTION_DIM = 48
ACTION_CHUNK_SIZE = 16
ACTION_UPSAMPLE_RATE = 3
DATASET_IMAGE_WRITER_THREADS = 0
DATASET_IMAGE_WRITER_PROCESSES = 0
_SPLIT_IDS = {"train": 0, "test": 1, "extra": 2}
_SUPPORTED_SPLIT_DIRS = {"part1", "part2", "part3", "part4", "part5", "test", "extra"}
_FINGERTIP_JOINTS = {
    "left": (
        "leftThumbTip",
        "leftIndexFingerTip",
        "leftMiddleFingerTip",
        "leftRingFingerTip",
        "leftLittleFingerTip",
    ),
    "right": (
        "rightThumbTip",
        "rightIndexFingerTip",
        "rightMiddleFingerTip",
        "rightRingFingerTip",
        "rightLittleFingerTip",
    ),
}


@dataclasses.dataclass(frozen=True)
class EpisodeSpec:
    split: str
    part_name: str
    part_id: int
    task_name: str
    task_id: int
    file_index: int
    mp4_path: Path
    hdf5_path: Path


@dataclasses.dataclass(frozen=True)
class JointSchema:
    joint_names: tuple[str, ...]


class EgoDexDataset(DirectVideoLeRobotDataset):
    """LeRobot v3 dataset that stores EgoDex RGB streams as copied MP4 files."""


def construct_48d_action_from_transforms(transforms_group: Any, frame_idx: int) -> np.ndarray:
    """Construct the Psi0/H-RDT 48 DoF EgoDex hand action vector for one frame."""
    action_vector: list[float] = []

    for hand_side in ("left", "right"):
        hand_transform = np.asarray(transforms_group[f"{hand_side}Hand"][frame_idx], dtype=np.float32)

        # Wrist position in the ARKit/world frame.
        action_vector.extend(hand_transform[:3, 3].tolist())

        # Wrist rotation as 6D representation: first two columns of the rotation matrix.
        rotation_matrix = hand_transform[:3, :3]
        rotation_6d = np.concatenate([rotation_matrix[:, 0], rotation_matrix[:, 1]])
        action_vector.extend(rotation_6d.tolist())

        # Fingertip positions in the ARKit/world frame.
        for fingertip in _FINGERTIP_JOINTS[hand_side]:
            fingertip_transform = np.asarray(transforms_group[fingertip][frame_idx], dtype=np.float32)
            action_vector.extend(fingertip_transform[:3, 3].tolist())

    return np.asarray(action_vector, dtype=np.float32)


def construct_48d_actions_from_transforms(transforms_group: Any, num_frames: int) -> np.ndarray:
    """Construct the Psi0/H-RDT 48 DoF EgoDex hand action vectors for all frames."""
    action_components: list[np.ndarray] = []

    for hand_side in ("left", "right"):
        hand_transforms = np.asarray(transforms_group[f"{hand_side}Hand"][:num_frames], dtype=np.float32)
        action_components.append(hand_transforms[:, :3, 3])
        action_components.append(hand_transforms[:, :3, :2].transpose(0, 2, 1).reshape(num_frames, 6))

        for fingertip in _FINGERTIP_JOINTS[hand_side]:
            fingertip_transforms = np.asarray(transforms_group[fingertip][:num_frames], dtype=np.float32)
            action_components.append(fingertip_transforms[:, :3, 3])

    return np.concatenate(action_components, axis=1).astype(np.float32, copy=False)


def d9_to_mat44_batch(nine_d: np.ndarray) -> np.ndarray:
    """Convert Psi0 9D wrist representations into homogeneous transforms."""
    position = nine_d[..., :3]
    rot_col0 = nine_d[..., 3:6]
    rot_col1 = nine_d[..., 6:9]

    col0 = rot_col0 / (np.linalg.norm(rot_col0, axis=-1, keepdims=True) + 1e-8)
    col1 = rot_col1 - np.sum(rot_col1 * col0, axis=-1, keepdims=True) * col0
    col1 = col1 / (np.linalg.norm(col1, axis=-1, keepdims=True) + 1e-8)
    col2 = np.cross(col0, col1)

    mat44 = np.broadcast_to(np.eye(4, dtype=nine_d.dtype), nine_d.shape[:-1] + (4, 4)).copy()
    mat44[..., :3, 0] = col0
    mat44[..., :3, 1] = col1
    mat44[..., :3, 2] = col2
    mat44[..., :3, 3] = position
    return mat44


def d9_to_mat44(nine_d: np.ndarray) -> np.ndarray:
    """Convert Psi0 9D wrist representation back to a 4x4 transform."""
    return d9_to_mat44_batch(nine_d)


def delta_rpy_from_tfs(tfs: np.ndarray) -> np.ndarray:
    """Compute Psi0 delta roll/pitch/yaw between consecutive transforms."""
    from scipy.spatial.transform import Rotation

    relative_rotations = tfs[..., 1:, :3, :3] @ np.swapaxes(tfs[..., :-1, :3, :3], -1, -2)
    flat_rotations = relative_rotations.reshape(-1, 3, 3)
    flat_deltas = Rotation.from_matrix(flat_rotations).as_euler("xyz", degrees=False).astype(np.float32)
    return flat_deltas.reshape(relative_rotations.shape[:-2] + (3,))


def points_to_camera(points_3d: np.ndarray, cam_ext: np.ndarray) -> np.ndarray:
    """Convert world-frame points into the current camera frame."""
    points_h = np.concatenate(
        [points_3d, np.ones(points_3d.shape[:-1] + (1,), dtype=points_3d.dtype)],
        axis=-1,
    )
    cam_inv = np.linalg.inv(cam_ext)
    if cam_inv.ndim == 2:
        return (points_h @ cam_inv.T)[..., :3]
    return np.einsum("...ij,...kj->...ki", cam_inv, points_h)[..., :3]


def convert_to_camera_frame(tfs: np.ndarray, cam_ext: np.ndarray) -> np.ndarray:
    """Convert world-frame transforms into the current camera frame."""
    cam_inv = np.linalg.inv(cam_ext)
    if cam_inv.ndim == 2:
        return cam_inv[None] @ tfs
    return cam_inv[..., None, :, :] @ tfs


def convert_to_delta_actions(actions: np.ndarray, chunk_size: int, cam_ext: np.ndarray) -> np.ndarray:
    """Match Psi0's conversion from absolute 48D actions to delta 48D actions."""
    left_wrist = d9_to_mat44_batch(actions[..., :9])
    right_wrist = d9_to_mat44_batch(actions[..., 24:33])
    left_hand_finger_tips = np.stack(
        [actions[..., 9:12], actions[..., 12:15], actions[..., 15:18], actions[..., 18:21], actions[..., 21:24]],
        axis=-2,
    )
    right_hand_finger_tips = np.stack(
        [actions[..., 33:36], actions[..., 36:39], actions[..., 39:42], actions[..., 42:45], actions[..., 45:48]],
        axis=-2,
    )

    left_wrist_tfs_in_cam = convert_to_camera_frame(left_wrist, cam_ext)
    right_wrist_tfs_in_cam = convert_to_camera_frame(right_wrist, cam_ext)

    delta_left_wrist_xyz = left_wrist_tfs_in_cam[..., 1:, :3, 3] - left_wrist_tfs_in_cam[..., :-1, :3, 3]
    delta_left_wrist_rpy = delta_rpy_from_tfs(left_wrist_tfs_in_cam)

    delta_right_wrist_xyz = right_wrist_tfs_in_cam[..., 1:, :3, 3] - right_wrist_tfs_in_cam[..., :-1, :3, 3]
    delta_right_wrist_rpy = delta_rpy_from_tfs(right_wrist_tfs_in_cam)

    left_points = points_to_camera(
        left_hand_finger_tips.reshape(left_hand_finger_tips.shape[:-3] + (chunk_size * 5, 3)),
        cam_ext,
    ).reshape(left_hand_finger_tips.shape)
    delta_left_fingers = left_points[..., 1:, :, :] - left_points[..., :-1, :, :]

    right_points = points_to_camera(
        right_hand_finger_tips.reshape(right_hand_finger_tips.shape[:-3] + (chunk_size * 5, 3)),
        cam_ext,
    ).reshape(right_hand_finger_tips.shape)
    delta_right_fingers = right_points[..., 1:, :, :] - right_points[..., :-1, :, :]

    return np.concatenate(
        [
            delta_left_wrist_xyz,
            delta_left_wrist_rpy,
            np.zeros_like(delta_left_wrist_rpy),
            delta_left_fingers.reshape(delta_left_fingers.shape[:-2] + (-1,)),
            delta_right_wrist_xyz,
            delta_right_wrist_rpy,
            np.zeros_like(delta_right_wrist_rpy),
            delta_right_fingers.reshape(delta_right_fingers.shape[:-2] + (-1,)),
        ],
        axis=-1,
    ).astype(np.float32)




def build_psi0_states_and_actions(
    absolute_actions: np.ndarray,
    max_index: int,
    camera_extrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build current states and current absolute actions for all output frames."""
    del camera_extrinsics
    num_output_frames = max_index + 1
    states = absolute_actions[:num_output_frames].astype(np.float32, copy=False)
    actions = absolute_actions[:num_output_frames].astype(np.float32, copy=False)
    return states, actions


def _decode_attr(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    return str(value).strip()


def _parse_int_attr(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def choose_prompt(attrs: Any, fallback: str) -> tuple[str, int]:
    desc1 = _decode_attr(attrs.get("llm_description"))
    desc2 = _decode_attr(attrs.get("llm_description2"))
    which = _parse_int_attr(attrs.get("which_llm_description"), default=0)

    if which == 2 and desc2:
        return desc2, 2
    if which == 1 and desc1:
        return desc1, 1
    if desc1:
        return desc1, 1
    if desc2:
        return desc2, 2
    return fallback, 0


def parse_part(part_name: str) -> tuple[str, int]:
    if part_name.startswith("part") and part_name[4:].isdigit():
        return "train", int(part_name[4:])
    if part_name == "test":
        return "test", 0
    if part_name == "extra":
        return "extra", 0
    raise ValueError(f"Unsupported EgoDex split directory: {part_name}")


def resolve_data_root(data_dir: Path) -> Path:
    if any((data_dir / d).is_dir() for d in _SUPPORTED_SPLIT_DIRS):
        return data_dir
    if (data_dir / "v1").exists():
        nested = data_dir / "v1"
        if any((nested / d).is_dir() for d in _SUPPORTED_SPLIT_DIRS):
            return nested
    return data_dir


def _resolve_scan_dirs(data_root: Path, subdirs: list[str] | None) -> list[Path]:
    if not subdirs:
        return [d for d in data_root.iterdir() if d.is_dir() and d.name in _SUPPORTED_SPLIT_DIRS]

    resolved_dirs: list[Path] = []
    seen: set[Path] = set()
    for subdir in subdirs:
        subdir_path = Path(subdir).expanduser()
        if not subdir_path.is_absolute():
            subdir_path = data_root / subdir_path
        resolved = subdir_path.resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"Requested EgoDex subdir does not exist or is not a directory: {subdir}")
        if resolved not in seen:
            resolved_dirs.append(resolved)
            seen.add(resolved)
    return resolved_dirs


def _find_split_dir(scan_dir: Path, data_root: Path) -> Path | None:
    try:
        scan_dir.relative_to(data_root)
    except ValueError as exc:
        raise ValueError(f"Requested subdir {scan_dir} is outside the resolved data root {data_root}.") from exc

    current = scan_dir
    while True:
        if current.name in _SUPPORTED_SPLIT_DIRS:
            return current
        if current == data_root:
            return None
        current = current.parent


def _collect_task_dirs(scan_dir: Path, split_dir: Path) -> list[Path]:
    task_root = split_dir / split_dir.name if (split_dir / split_dir.name).is_dir() else split_dir

    if scan_dir in (split_dir, task_root):
        return [p for p in sorted(task_root.iterdir()) if p.is_dir()]
    if scan_dir.parent == task_root:
        return [scan_dir]

    task_dirs: list[Path] = []
    for candidate in sorted(p for p in scan_dir.rglob("*") if p.is_dir()):
        has_mp4 = next(candidate.glob("*.mp4"), None) is not None
        if not has_mp4:
            continue
        has_hdf5 = next(candidate.glob("*.hdf5"), None) is not None
        if has_hdf5:
            task_dirs.append(candidate)
    return task_dirs


def discover_task_names(data_root: Path, *, subdirs: list[str] | None = None) -> list[str]:
    scan_dirs = _resolve_scan_dirs(data_root, subdirs)
    if not scan_dirs:
        raise FileNotFoundError(
            f"No EgoDex split directories found in {data_root}. "
            "Expected part1..part5/test/extra."
        )

    task_names: set[str] = set()
    seen_task_dirs: set[Path] = set()
    for scan_dir in sorted(scan_dirs, key=lambda p: p.as_posix()):
        split_dir = _find_split_dir(scan_dir, data_root)
        if split_dir is None:
            raise ValueError(
                f"Requested subdir {scan_dir} is not inside a supported EgoDex split directory. "
                "Expected a path under part1..part5/test/extra."
            )
        for candidate_task_dir in _collect_task_dirs(scan_dir, split_dir):
            task_dir = candidate_task_dir.resolve()
            if task_dir in seen_task_dirs:
                continue
            seen_task_dirs.add(task_dir)
            task_names.add(task_dir.name)

    return sorted(task_names)


def discover_episode_pairs(
    data_root: Path,
    *,
    subdirs: list[str] | None = None,
    task_names: list[str] | None = None,
) -> list[EpisodeSpec]:
    scan_dirs = _resolve_scan_dirs(data_root, subdirs)
    if not scan_dirs:
        raise FileNotFoundError(
            f"No EgoDex split directories found in {data_root}. "
            "Expected part1..part5/test/extra."
        )

    requested_task_names = set(task_names or [])
    discovered_task_names: set[str] = set()
    pair_rows: list[tuple[str, int, str, Path, Path]] = []
    seen_task_dirs: set[Path] = set()

    for scan_dir in sorted(scan_dirs, key=lambda p: p.as_posix()):
        split_dir = _find_split_dir(scan_dir, data_root)
        if split_dir is None:
            raise ValueError(
                f"Requested subdir {scan_dir} is not inside a supported EgoDex split directory. "
                "Expected a path under part1..part5/test/extra."
            )
        split, part_id = parse_part(split_dir.name)
        for candidate_task_dir in _collect_task_dirs(scan_dir, split_dir):
            task_dir = candidate_task_dir.resolve()
            if task_dir in seen_task_dirs:
                continue
            seen_task_dirs.add(task_dir)
            if requested_task_names and task_dir.name not in requested_task_names:
                continue
            mp4_by_stem = {p.stem: p for p in task_dir.glob("*.mp4")}
            hdf5_by_stem = {p.stem: p for p in task_dir.glob("*.hdf5")}
            for stem in sorted(set(mp4_by_stem).intersection(hdf5_by_stem), key=lambda x: int(x) if x.isdigit() else x):
                discovered_task_names.add(task_dir.name)
                pair_rows.append((split, part_id, task_dir.name, mp4_by_stem[stem], hdf5_by_stem[stem]))

    task_to_id = {name: i for i, name in enumerate(sorted(discovered_task_names))}

    episodes: list[EpisodeSpec] = []
    for split, part_id, task_name, mp4_path, hdf5_path in pair_rows:
        episodes.append(
            EpisodeSpec(
                split=split,
                part_name=(f"part{part_id}" if split == "train" else split),
                part_id=part_id,
                task_name=task_name,
                task_id=task_to_id[task_name],
                file_index=int(mp4_path.stem) if mp4_path.stem.isdigit() else -1,
                mp4_path=mp4_path,
                hdf5_path=hdf5_path,
            )
        )

    return episodes


def group_episodes_by_task(episodes: list[EpisodeSpec]) -> dict[str, list[EpisodeSpec]]:
    grouped: dict[str, list[EpisodeSpec]] = defaultdict(list)
    for episode in episodes:
        grouped[episode.task_name].append(episode)
    return {task_name: grouped[task_name] for task_name in sorted(grouped)}


def select_task_names(
    grouped_episodes: dict[str, list[EpisodeSpec]],
    *,
    task_names: list[str] | None,
    start_task_name: str | None,
    max_tasks: int | None,
) -> list[str]:
    selected = sorted(grouped_episodes)
    if task_names:
        requested = set(task_names)
        selected = [task_name for task_name in selected if task_name in requested]
    if start_task_name is not None:
        try:
            start_index = selected.index(start_task_name)
        except ValueError as exc:
            raise ValueError(f"Task {start_task_name!r} is not present in the selected task set.") from exc
        selected = selected[start_index:]
    if max_tasks is not None:
        selected = selected[:max_tasks]
    return selected


def infer_joint_schema(first_hdf5_path: Path) -> JointSchema:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        raise ImportError(
            "h5py is required for EgoDex conversion. Install it in your environment first."
        ) from exc

    with h5py.File(first_hdf5_path, "r") as f:
        if "transforms" not in f:
            raise KeyError(f"Missing 'transforms' group in {first_hdf5_path}")
        joint_names = tuple(sorted(f["transforms"].keys()))
        if not joint_names:
            raise ValueError(f"No joints found under transforms in {first_hdf5_path}")
    return JointSchema(joint_names=joint_names)


def get_video_reader(video_path: Path):
    try:
        import decord
    except ImportError:
        return OpenCVVideoReader(video_path)
    else:
        decord.bridge.set_bridge("native")
        return decord.VideoReader(str(video_path))


class OpenCVVideoReader:
    """Sequential fallback reader when decord is unavailable."""

    def __init__(self, video_path: Path) -> None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - runtime dependency guard
            raise ImportError(
                "Either decord or opencv-python is required for EgoDex conversion."
            ) from exc

        self._cv2 = cv2
        self._cap = cv2.VideoCapture(str(video_path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video {video_path}")
        self._frame_count = max(0, int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    def __len__(self) -> int:
        return self._frame_count

    def iter_frames(self, *, limit: int):
        for _ in range(limit):
            ok, frame = self._cap.read()
            if not ok:
                break
            yield self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        self._cap.release()


def iter_video_frames(reader, *, limit: int, batch_size: int = 256):
    if hasattr(reader, "iter_frames"):
        yield from reader.iter_frames(limit=limit)
        return

    total = len(reader)
    total = min(total, limit)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        indices = list(range(start, end))
        batch = reader.get_batch(indices).asnumpy()
        yield from batch


def _frame_to_numpy(frame: Any) -> np.ndarray:
    if hasattr(frame, "asnumpy"):
        return frame.asnumpy()
    return np.asarray(frame)


def get_video_reader_shape(reader) -> tuple[int, int, int]:
    if hasattr(reader, "__getitem__"):
        try:
            return tuple(_frame_to_numpy(reader[0]).shape)
        except Exception:
            pass

    first_frame = next(iter_video_frames(reader, limit=1), None)
    if first_frame is None:
        raise ValueError("Video contains no readable frames.")

    frame_shape = tuple(np.asarray(first_frame).shape)
    if len(frame_shape) != 3:
        raise ValueError(f"Expected video frame shape (H, W, C), found {frame_shape}.")
    return frame_shape


def infer_video_shape(video_path: Path) -> tuple[int, int, int]:
    reader = get_video_reader(video_path)
    try:
        if len(reader) == 0:
            raise ValueError(f"Video {video_path} contains no frames.")
        return get_video_reader_shape(reader)
    finally:
        close_video_reader(reader)


def close_video_reader(reader) -> None:
    close = getattr(reader, "close", None)
    if callable(close):
        close()


def create_dataset(
    repo_name: str,
    video_shape: tuple[int, int, int],
    joint_count: int,
    *,
    output_path: Path | None = None,
    keep_extra_fields: bool = False,
) -> EgoDexDataset:
    return EgoDexDataset.create(
        repo_id=repo_name,
        root=output_path,
        robot_type="egodex",
        fps=FPS,
        features=build_dataset_features(video_shape, joint_count, keep_extra_fields=keep_extra_fields),
        image_writer_threads=DATASET_IMAGE_WRITER_THREADS,
        image_writer_processes=DATASET_IMAGE_WRITER_PROCESSES,
    )


def resolve_episode_commit_batch_size(requested: int | None, *, workers: int, episode_count: int) -> int:
    if episode_count <= 0:
        return 1
    if requested is None:
        requested = min(max(8, workers), 16)
    return max(1, min(requested, episode_count))


def _column_to_matrix(column: Any) -> np.ndarray:
    array = column.combine_chunks() if hasattr(column, "combine_chunks") else column
    if hasattr(array, "to_numpy"):
        try:
            values = array.to_numpy(zero_copy_only=False)
        except TypeError:
            values = array.to_numpy()
    else:
        values = array.to_pylist()

    matrix = np.asarray(values)
    if matrix.dtype == object:
        matrix = np.asarray(matrix.tolist(), dtype=np.float32)
    return np.asarray(matrix, dtype=np.float32)


def write_norm_stats(dataset_root: Path, *, run_compute_stats: bool) -> None:
    if not run_compute_stats:
        return

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        raise ImportError("pyarrow is required to compute EgoDex norm stats.") from exc

    stats = {
        "state": normalize.RunningStats(),
        "actions": normalize.RunningStats(),
    }
    seen: set[str] = set()

    for parquet_path in sorted(dataset_root.glob("data/chunk-*/file-*.parquet")):
        table = pq.read_table(parquet_path)
        if "observation.state" in table.schema.names:
            stats["state"].update(_column_to_matrix(table.column("observation.state")))
            seen.add("state")
        if "actions" in table.schema.names:
            stats["actions"].update(_column_to_matrix(table.column("actions")))
            seen.add("actions")

    if not seen:
        return

    normalize.save(dataset_root, {key: stats[key].get_statistics() for key in sorted(seen)})


def build_task_manifest(task_name: str, task_episodes: list[EpisodeSpec], joint_names: tuple[str, ...]) -> dict[str, Any]:
    return {
        "task_name": task_name,
        "action_dim": ACTION_DIM,
        "state_shape": [STATE_DIM],
        "action_shape": [ACTION_DIM],
        "action_format": "absolute_per_frame",
        "recommended_action_chunk_size": ACTION_CHUNK_SIZE,
        "recommended_action_offsets": [ACTION_UPSAMPLE_RATE * (i + 1) for i in range(ACTION_CHUNK_SIZE)],
        "action_reference_frame": "world_frame",
        "action_components_per_hand": {
            "wrist_position": 3,
            "wrist_rotation_6d": 6,
            "fingertips_xyz": 15,
        },
        "joint_names": list(joint_names),
        "split_id_map": _SPLIT_IDS,
        "episode_count": len(task_episodes),
        "split_counts": {
            split_name: sum(1 for episode in task_episodes if episode.split == split_name)
            for split_name in sorted(_SPLIT_IDS)
        },
        "part_ids": sorted({episode.part_id for episode in task_episodes}),
    }


@dataclasses.dataclass(frozen=True)
class TaskConvertJob:
    repo_id: str
    task_name: str
    task_episodes: list[EpisodeSpec]
    schema: JointSchema
    video_shape: tuple[int, int, int]
    output_root: Path
    run_compute_stats: bool
    push_to_hub: bool
    resume: bool
    keep_extra_fields: bool


def build_dataset_features(
    video_shape: tuple[int, int, int],
    joint_count: int,
    *,
    keep_extra_fields: bool = False,
) -> dict[str, dict[str, Any]]:
    h, w, c = video_shape
    features = {
        "observation.images.top": {
            "dtype": "video",
            "shape": (h, w, c),
            "names": ["height", "width", "rgb"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": ["state"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": None,
        },
        "egodex.camera_extrinsic": {
            "dtype": "float32",
            "shape": (4, 4),
            "names": None,
        },
    }
    if keep_extra_fields:
        features.update(
            {
                "egodex.camera_intrinsic": {
                    "dtype": "float32",
                    "shape": (3, 3),
                    "names": None,
                },
                "egodex.transforms": {
                    "dtype": "float32",
                    "shape": (joint_count, 4, 4),
                    "names": None,
                },
                "egodex.confidences": {
                    "dtype": "float32",
                    "shape": (joint_count,),
                    "names": None,
                },
                "egodex.confidence_mask": {
                    "dtype": "float32",
                    "shape": (joint_count,),
                    "names": None,
                },
            }
        )
    return features


def build_episode_columns(
    *,
    prompt: str,
    num_output_frames: int,
    states: np.ndarray,
    actions: np.ndarray,
    camera_extrinsics: np.ndarray,
    extra_fields: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    columns = {
        "size": num_output_frames,
        "task": [prompt] * num_output_frames,
        "frame_index": np.arange(num_output_frames, dtype=np.int64),
        "timestamp": np.arange(num_output_frames, dtype=np.float32) / float(FPS),
        "observation.state": states,
        "actions": actions,
        "egodex.camera_extrinsic": camera_extrinsics.astype(np.float32, copy=False),
    }
    if extra_fields:
        columns.update(extra_fields)
    return columns


def fill_episode_buffer(episode_data: dict[str, Any], columns: dict[str, Any]) -> dict[str, Any]:
    episode_data["size"] = int(columns["size"])
    episode_data["task"] = list(columns["task"])
    for key, value in columns.items():
        if key in {"size", "task"}:
            continue
        episode_data[key] = value
    return episode_data


def save_prepared_episode_batch(
    dataset: EgoDexDataset,
    payload_batch: list[dict[str, Any]],
    *,
    link_mode: MergeLinkMode = "auto",
    passthrough_video_workers: int | None = None,
    probe_video_durations: bool = False,
    on_episode_saved: Callable[[int, int], None] | None = None,
) -> None:
    if not payload_batch:
        return

    start_episode_index = dataset.meta.total_episodes
    episode_batch = [
        fill_episode_buffer(
            dataset.create_episode_buffer(start_episode_index + offset),
            dict(payload["episode_columns"]),
        )
        for offset, payload in enumerate(payload_batch)
    ]
    dataset.save_episode_batch(
        episode_batch,
        videos=[{"observation.images.top": Path(payload["video_path"])} for payload in payload_batch],
        video_durations=[
            {"observation.images.top": float(payload["video_duration_s"])}
            for payload in payload_batch
        ],
        link_mode=link_mode,
        passthrough_video_workers=passthrough_video_workers,
        probe_video_durations=probe_video_durations,
        on_episode_saved=on_episode_saved,
    )


def prepare_episode_payload(
    episode: EpisodeSpec,
    *,
    schema: JointSchema,
    video_shape: tuple[int, int, int],
    keep_extra_fields: bool = False,
) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        raise ImportError("h5py is required for EgoDex conversion. Install it in your environment first.") from exc

    reader = None
    try:
        try:
            reader = get_video_reader(episode.mp4_path)
        except Exception as exc:
            return {"status": "skipped", "task_name": episode.task_name, "reason": str(exc)}

        if len(reader) == 0:
            return {"status": "skipped", "task_name": episode.task_name, "reason": "empty_video"}

        source_frame_count = len(reader)
        source_video_duration_s = float(get_video_duration_in_s(episode.mp4_path))
        actual_video_shape = get_video_reader_shape(reader)
        if tuple(actual_video_shape) != tuple(video_shape):
            return {
                "status": "skipped",
                "task_name": episode.task_name,
                "reason": f"video_shape_mismatch:{actual_video_shape}",
            }

        with h5py.File(episode.hdf5_path, "r") as f:
            transforms_group = f["transforms"]
            joint_names = tuple(sorted(transforms_group.keys()))
            if joint_names != schema.joint_names:
                return {"status": "skipped", "task_name": episode.task_name, "reason": "joint_schema_mismatch"}

            n_pose = transforms_group[schema.joint_names[0]].shape[0]
            n_frames = min(source_frame_count, n_pose)
            max_index = n_frames - 2
            if max_index < 0:
                return {"status": "skipped", "task_name": episode.task_name, "reason": "too_short"}

            num_output_frames = max_index + 1
            absolute_actions = construct_48d_actions_from_transforms(transforms_group, n_frames)
            camera_extrinsics = np.asarray(transforms_group["camera"][:num_output_frames], dtype=np.float32)
            states, actions = build_psi0_states_and_actions(
                absolute_actions,
                max_index,
                camera_extrinsics,
            )
            extra_fields = None
            if keep_extra_fields:
                all_transforms = np.stack(
                    [np.asarray(transforms_group[name][:num_output_frames], dtype=np.float32) for name in schema.joint_names],
                    axis=1,
                )
                joint_count = len(schema.joint_names)
                confidences_group = f.get("confidences")
                all_confidences = np.zeros((num_output_frames, joint_count), dtype=np.float32)
                all_confidence_masks = np.zeros((num_output_frames, joint_count), dtype=np.float32)
                if confidences_group is not None:
                    for joint_idx, name in enumerate(schema.joint_names):
                        if name in confidences_group:
                            all_confidences[:, joint_idx] = np.asarray(
                                confidences_group[name][:num_output_frames],
                                dtype=np.float32,
                            )
                            all_confidence_masks[:, joint_idx] = 1.0
                intrinsic = np.asarray(f["camera"]["intrinsic"], dtype=np.float32)
                extra_fields = {
                    "egodex.camera_intrinsic": np.broadcast_to(
                        intrinsic.astype(np.float32, copy=False),
                        (num_output_frames, 3, 3),
                    ).copy(),
                    "egodex.transforms": all_transforms.astype(np.float32, copy=False),
                    "egodex.confidences": all_confidences.astype(np.float32, copy=False),
                    "egodex.confidence_mask": all_confidence_masks.astype(np.float32, copy=False),
                }
            prompt, _ = choose_prompt(f.attrs, fallback=episode.task_name.replace("_", " "))
    finally:
        if reader is not None:
            close_video_reader(reader)

    return {
        "status": "converted",
        "task_name": episode.task_name,
        "video_path": str(episode.mp4_path),
        "video_duration_s": source_video_duration_s,
        "episode_columns": build_episode_columns(
            prompt=prompt,
            num_output_frames=num_output_frames,
            states=states,
            actions=actions,
            camera_extrinsics=camera_extrinsics,
            extra_fields=extra_fields,
        ),
        "frame_count": num_output_frames,
    }


def load_committed_episode_count(output_root: Path) -> int | None:
    info_path = output_root / "meta" / "info.json"
    if not info_path.exists():
        return None

    info = json.loads(info_path.read_text())
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
) -> EgoDexDataset:
    return EgoDexDataset(repo_id=repo_id, root=output_root)


def validate_resume_dataset(
    dataset: EgoDexDataset,
    *,
    video_shape: tuple[int, int, int],
    joint_count: int,
    keep_extra_fields: bool,
) -> None:
    if dataset.meta.fps != FPS:
        raise ValueError(f"Cannot resume {dataset.root}: expected fps={FPS}, found {dataset.meta.fps}.")
    if dataset.meta.robot_type != "egodex":
        raise ValueError(
            f"Cannot resume {dataset.root}: expected robot_type='egodex', found {dataset.meta.robot_type!r}."
        )

    expected_features = build_dataset_features(video_shape, joint_count, keep_extra_fields=keep_extra_fields)
    optional_extra_keys = {
        "egodex.camera_intrinsic",
        "egodex.transforms",
        "egodex.confidences",
        "egodex.confidence_mask",
    }
    actual_has_extra_fields = any(key in dataset.features for key in optional_extra_keys)
    if actual_has_extra_fields != keep_extra_fields:
        raise ValueError(
            f"Cannot resume {dataset.root}: existing dataset keep_extra_fields={actual_has_extra_fields}, "
            f"but requested keep_extra_fields={keep_extra_fields}."
        )
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


def convert_task_dataset(
    *,
    repo_id: str,
    task_name: str,
    task_episodes: list[EpisodeSpec],
    schema: JointSchema,
    video_shape: tuple[int, int, int],
    output_root: Path,
    run_compute_stats: bool,
    push_to_hub: bool,
    resume: bool,
    keep_extra_fields: bool = False,
    episode_commit_batch_size: int | None = None,
) -> dict[str, Any]:
    joint_count = len(schema.joint_names)
    committed_episodes = 0
    episodes_to_convert = task_episodes
    rebuilt_empty_resume = False

    if output_root.exists():
        if resume:
            if not output_root.is_dir():
                raise ValueError(f"Cannot resume {task_name!r}: {output_root} exists and is not a directory.")

            committed_episodes = load_committed_episode_count(output_root)
            if committed_episodes is None:
                raise ValueError(
                    f"Cannot resume {task_name!r}: {output_root} exists but does not contain meta/info.json."
                )
            if committed_episodes > len(task_episodes):
                raise ValueError(
                    f"Cannot resume {task_name!r}: existing dataset has {committed_episodes} committed episodes, "
                    f"but the current selection only contains {len(task_episodes)} episodes. "
                    "Re-run with the same task filters or remove the existing output directory."
                )

            if committed_episodes == 0:
                shutil.rmtree(output_root)
                dataset = create_dataset(
                    repo_id,
                    video_shape,
                    joint_count,
                    output_path=output_root,
                    keep_extra_fields=keep_extra_fields,
                )
                rebuilt_empty_resume = True
            else:
                dataset = open_dataset_for_append(repo_id, output_root)
                validate_resume_dataset(
                    dataset,
                    video_shape=video_shape,
                    joint_count=joint_count,
                    keep_extra_fields=keep_extra_fields,
                )
                episodes_to_convert = task_episodes[committed_episodes:]
                print(
                    f"Resuming {task_name}: found {committed_episodes} committed episodes; "
                    f"{len(episodes_to_convert)} remaining."
                )
        else:
            if output_root.is_dir():
                shutil.rmtree(output_root)
            else:
                output_root.unlink()
            dataset = create_dataset(
                repo_id,
                video_shape,
                joint_count,
                output_path=output_root,
                keep_extra_fields=keep_extra_fields,
            )
    else:
        dataset = create_dataset(
            repo_id,
            video_shape,
            joint_count,
            output_path=output_root,
            keep_extra_fields=keep_extra_fields,
        )

    if rebuilt_empty_resume:
        print(f"Resume requested for {task_name}, but no committed episodes were found. Rebuilding task from scratch.")

    converted = 0
    skipped = 0
    frames_written = 0
    commit_batch_size = resolve_episode_commit_batch_size(
        episode_commit_batch_size,
        workers=DEFAULT_NUM_WORKERS,
        episode_count=len(episodes_to_convert),
    )
    ready_payloads: list[dict[str, Any]] = []

    for episode in tqdm(episodes_to_convert, desc=f"Converting {task_name}"):
        result = prepare_episode_payload(
            episode,
            schema=schema,
            video_shape=video_shape,
            keep_extra_fields=keep_extra_fields,
        )
        if result["status"] != "converted":
            reason = str(result.get("reason", "unknown"))
            if reason.startswith("video_shape_mismatch:"):
                print(
                    "Warning: skipping episode with different video shape "
                    f"{episode.mp4_path} ({reason.split(':', 1)[1]} != {video_shape})"
                )
            elif reason == "joint_schema_mismatch":
                print(f"Warning: skipping episode with different joint schema {episode.hdf5_path}")
            elif reason not in {"empty_video", "too_short"}:
                print(f"Warning: failed to prepare episode {episode.mp4_path}: {reason}")
            skipped += 1
            continue

        ready_payloads.append(result)
        while len(ready_payloads) >= commit_batch_size:
            batch = ready_payloads[:commit_batch_size]
            del ready_payloads[:commit_batch_size]
            save_prepared_episode_batch(dataset, batch)
            converted += len(batch)
            frames_written += sum(int(item["frame_count"]) for item in batch)

    if ready_payloads:
        save_prepared_episode_batch(dataset, ready_payloads)
        converted += len(ready_payloads)
        frames_written += sum(int(item["frame_count"]) for item in ready_payloads)

    dataset.finalize()
    write_norm_stats(
        output_root,
        run_compute_stats=run_compute_stats and (converted > 0 or not (output_root / "norm_stats.json").exists()),
    )
    manifest = build_task_manifest(task_name, task_episodes, schema.joint_names)
    (output_root / "egodex_metadata.json").write_text(json.dumps(manifest, indent=2))

    print(
        f"Converted {converted} episodes; "
        f"reused {committed_episodes} existing episodes; skipped {skipped}"
    )
    print(f"Saved to {output_root}")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["egodex", "egocentric", "dexterous-manipulation"],
            private=False,
            push_videos=True,
            license="cc-by-nc-nd-4.0",
        )

    return {
        "task_name": task_name,
        "repo_id": repo_id,
        "output_path": str(output_root),
        "episodes_written": converted,
        "episodes_already_present": committed_episodes,
        "episodes_skipped": skipped,
        "frames_written": frames_written,
    }


def convert_task_dataset_job(job: TaskConvertJob) -> dict[str, Any]:
    return convert_task_dataset(
        repo_id=job.repo_id,
        task_name=job.task_name,
        task_episodes=job.task_episodes,
        schema=job.schema,
        video_shape=job.video_shape,
        output_root=job.output_root,
        run_compute_stats=job.run_compute_stats,
        push_to_hub=job.push_to_hub,
        resume=job.resume,
        keep_extra_fields=job.keep_extra_fields,
    )


def main(
    data_dir: str,
    *,
    repo_name: str = REPO_NAME,
    output_path: str = str(DEFAULT_OUTPUT_PATH),
    subdirs: list[str] | None = None,
    task_names: list[str] | None = None,
    start_task_name: str | None = None,
    max_tasks: int | None = None,
    episode_offset: int = 0,
    episodes_per_task: int | None = None,
    image_size: tuple[int, int] = IMAGE_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    skip_stats: bool = False,
    summary_json: str | None = None,
    push_to_hub: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """Convert raw EgoDex episodes into one local dataset per task."""
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1.")
    if push_to_hub and num_workers > 1:
        raise ValueError("push_to_hub with num_workers > 1 is not supported.")
    if image_size != IMAGE_SIZE:
        print("Ignoring --image_size: EgoDex conversion preserves raw MP4s and training resizes later.")

    data_path = resolve_data_root(Path(data_dir).expanduser().resolve())
    output_base = Path(output_path).expanduser().resolve() if output_path else HF_LEROBOT_HOME / repo_name
    output_base.mkdir(parents=True, exist_ok=True)

    episodes = discover_episode_pairs(data_path, subdirs=subdirs)
    if not episodes:
        raise ValueError(f"No valid paired mp4+hdf5 episodes found under {data_path}")

    grouped_episodes = group_episodes_by_task(episodes)
    selected_task_names = select_task_names(
        grouped_episodes,
        task_names=task_names,
        start_task_name=start_task_name,
        max_tasks=max_tasks,
    )
    if not selected_task_names:
        raise ValueError("No EgoDex tasks matched the requested selection.")

    schema = infer_joint_schema(episodes[0].hdf5_path)
    jobs: list[TaskConvertJob] = []

    for task_name in selected_task_names:
        task_episodes = grouped_episodes[task_name]
        if episode_offset:
            task_episodes = task_episodes[episode_offset:]
        if episodes_per_task is not None:
            task_episodes = task_episodes[:episodes_per_task]
        if not task_episodes:
            continue

        repo_id = f"{repo_name}/{task_name}"
        task_output_root = output_base / task_name
        video_shape = infer_video_shape(task_episodes[0].mp4_path)
        jobs.append(
            TaskConvertJob(
                repo_id=repo_id,
                task_name=task_name,
                task_episodes=task_episodes,
                schema=schema,
                video_shape=video_shape,
                output_root=task_output_root,
                run_compute_stats=not skip_stats,
                push_to_hub=push_to_hub,
                resume=resume,
            )
        )

    task_results: list[dict[str, Any]] = []
    if not jobs:
        raise ValueError("No EgoDex task datasets remain after applying episode/task filters.")
    effective_workers = min(num_workers, len(jobs), os.cpu_count() or 1)

    if effective_workers == 1:
        for job in jobs:
            print(f"\nWriting {job.repo_id} -> {job.output_root}")
            task_results.append(convert_task_dataset_job(job))
    else:
        print(f"Running {len(jobs)} EgoDex tasks with {effective_workers} worker processes.")
        with ProcessPoolExecutor(
            max_workers=effective_workers,
            mp_context=mp.get_context("spawn"),
        ) as executor:
            futures = {}
            for job in jobs:
                print(f"Queueing {job.repo_id} -> {job.output_root}")
                futures[executor.submit(convert_task_dataset_job, job)] = job.task_name

            for future in as_completed(futures):
                task_name = futures[future]
                try:
                    task_results.append(future.result())
                except Exception as exc:
                    raise RuntimeError(f"Parallel EgoDex conversion failed for task {task_name!r}.") from exc

        task_results.sort(key=lambda item: item["task_name"])

    summary = {
        "data_dir": str(data_path),
        "output_path": str(output_base),
        "repo_name": repo_name,
        "selected_subdirs": subdirs,
        "selected_tasks": selected_task_names,
        "num_workers": effective_workers,
        "episode_offset": episode_offset,
        "episodes_per_task": episodes_per_task,
        "resume": resume,
        "datasets": task_results,
        "totals": {
            "datasets_written": len(task_results),
            "episodes_written": sum(task["episodes_written"] for task in task_results),
            "episodes_already_present": sum(task["episodes_already_present"] for task in task_results),
            "episodes_skipped": sum(task["episodes_skipped"] for task in task_results),
            "frames_written": sum(task["frames_written"] for task in task_results),
        },
    }

    summary_path = Path(summary_json).expanduser().resolve() if summary_json else (output_base / "conversion_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote summary to {summary_path}")
    return summary
