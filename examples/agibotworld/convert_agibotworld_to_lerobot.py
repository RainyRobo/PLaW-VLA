#!/usr/bin/env python3
"""Convert raw AgiBot-World data into embodiment-level LeRobot v3 datasets."""
# ruff: noqa: E402, SLF001

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
import dataclasses
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import h5py
import pyarrow as pa
import pyarrow.parquet as pq

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")

import numpy as np
import tyro

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
SRC_ROOT = REPO_ROOT / "src"
PYTHON_PATHS = (THIS_DIR, REPO_ROOT, SRC_ROOT)


def _ensure_import_paths() -> None:
    for path in PYTHON_PATHS:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_ensure_import_paths()

from openpi.datasets.common import hub as hub_push
from openpi.datasets.common import progress as progress_display
from openpi.datasets.common import ray_runtime
from openpi.datasets.common import staging
from openpi.datasets.common.async_commit import OrderedAsyncBatchCommitter
from openpi.datasets.common.async_commit import pop_contiguous_batch
from openpi.datasets.common.async_commit import resolve_max_inflight_episodes
from openpi.datasets.common.conversion_checkpoint import EpisodeCheckpointStore
from openpi.datasets.common.lerobot_v3 import MergeLinkMode
from openpi.datasets.common.video_concat import concatenate_video_files
from openpi.datasets.specs import agibotworld as convert_agibotworld

ray, _HAS_RAY = ray_runtime.import_optional_ray()

CONSOLE = progress_display.get_console()
SUPPORTED_EEF_TYPES = ("gripper", "dexhand")
PREPARE_RETRY_ATTEMPTS = 2


@dataclasses.dataclass(frozen=True)
class Args:
    output_dir: Path
    src_path: Path = Path("data/raw/agibotworld")
    staging_root: Path | None = None
    tmp_dir: Path | None = None
    ray_temp_root: Path | None = None
    cleanup_tmp_on_success: bool = False
    eef_type: str = "all"
    task_ids: tuple[str, ...] = ()
    task_id_file: Path | None = None
    start_task_id: str | None = None
    max_tasks: int | None = None
    episode_offset: int = 0
    episodes_per_task: int | None = None
    keep_extra_fields: bool = False
    resume: bool = False
    overwrite: bool = False
    cleanup_resolved_failures: bool = False
    skip_stats: bool = True
    conversion_num_workers: int | None = None
    episode_commit_batch_size: int | None = None
    max_inflight_episodes: int | None = None
    video_link_mode: MergeLinkMode = "copy"
    push_to_hub: bool = False
    hub_owner: str | None = None
    summary_json: Path | None = None


def _source_episode_key(task_id: str, episode_id: int) -> str:
    return f"{task_id}:{episode_id}"


def _resolve_eef_types(requested: str) -> tuple[str, ...]:
    normalized = requested.strip().lower()
    if normalized == "all":
        return SUPPORTED_EEF_TYPES
    if normalized not in SUPPORTED_EEF_TYPES:
        raise ValueError(f"eef_type must be one of {SUPPORTED_EEF_TYPES} or 'all', got {requested!r}.")
    return (normalized,)


def _resolve_conversion_workers(requested: int | None, job_count: int) -> int:
    if job_count <= 0:
        return 1
    cpu_count = os.cpu_count() or 1
    if requested is None:
        requested = min(cpu_count, job_count)
    return max(1, min(requested, cpu_count, job_count))


def _resolve_ray_prepare_cpus() -> float:
    raw = os.environ.get("OPENPI_RAY_PREPARE_CPUS", "0.5")
    try:
        value = float(raw)
    except ValueError:
        value = 0.5
    return max(0.1, value)


RAY_PREPARE_CPUS = _resolve_ray_prepare_cpus()


def _resolve_episode_commit_batch_size(requested: int | None, *, workers: int, episode_count: int) -> int:
    if episode_count <= 0:
        return 1
    if requested is None:
        requested = min(max(4, workers), 8)
    return max(1, min(requested, episode_count))


def _load_requested_task_ids(args: Args) -> list[str]:
    requested_task_ids = [convert_agibotworld._normalize_task_id(task_id) for task_id in args.task_ids]
    if args.task_id_file is not None:
        requested_task_ids.extend(convert_agibotworld._load_task_id_file(args.task_id_file))
    return list(dict.fromkeys(requested_task_ids))


def _infer_eef_type_from_task(src_path: Path, task: convert_agibotworld.TaskSpec) -> str:
    task_info = convert_agibotworld._load_task_info(task.json_file)
    if not task_info:
        raise ValueError(f"Task {task.task_stem} does not contain any episodes.")
    first_episode_id = int(task_info[0]["episode_id"])
    h5_path = src_path / "proprio_stats" / task.task_id / str(first_episode_id) / "proprio_stats.h5"
    with h5py.File(h5_path, "r") as f:
        effector_dim = int(f["state/effector/position"].shape[-1])
    if effector_dim == 2:
        return "gripper"
    if effector_dim == 12:
        return "dexhand"
    raise ValueError(f"Unsupported effector dimension {effector_dim} for {task.task_stem}.")


def _select_tasks(
    src_path: Path,
    *,
    eef_type: str,
    explicit_task_ids: list[str],
    start_task_id: str | None,
    max_tasks: int | None,
) -> list[convert_agibotworld.TaskSpec]:
    tasks = convert_agibotworld._get_all_tasks(src_path)
    typed_tasks = [(task, _infer_eef_type_from_task(src_path, task)) for task in tasks]
    tasks = [task for task, inferred_type in typed_tasks if inferred_type == eef_type]

    if explicit_task_ids:
        requested = set(explicit_task_ids)
        tasks = [task for task in tasks if task.task_stem in requested]

    if start_task_id is not None:
        normalized_start_task_id = convert_agibotworld._normalize_task_id(start_task_id)
        start_idx = next(
            (idx for idx, task in enumerate(tasks) if task.task_stem == normalized_start_task_id),
            None,
        )
        if start_idx is None:
            raise ValueError(f"Start task id {normalized_start_task_id!r} is not present in the selected task set.")
        tasks = tasks[start_idx:]

    if max_tasks is not None:
        tasks = tasks[:max_tasks]
    return tasks


def _output_task_config(eef_type: str) -> tuple[dict[str, Any], dict[str, str]]:
    raw_task_config = convert_agibotworld.AGIBOTWORLD_TASK_CONFIGS[eef_type]
    if eef_type == "dexhand":
        # Dex-hand episodes do not provide the narrow FOV hand color streams that
        # gripper episodes have. We keep a shared canonical three-camera interface
        # by routing the available fisheye hand views into hand_left/right.
        image_specs = {
            "head": dict(raw_task_config["images"]["head"]),
            "hand_left": dict(raw_task_config["images"]["hand_left_fisheye"]),
            "hand_right": dict(raw_task_config["images"]["hand_right_fisheye"]),
        }
        image_sources = {
            "head": "head",
            "hand_left": "hand_left_fisheye",
            "hand_right": "hand_right_fisheye",
        }
    else:
        image_specs = {
            "head": dict(raw_task_config["images"]["head"]),
            "hand_left": dict(raw_task_config["images"]["hand_left"]),
            "hand_right": dict(raw_task_config["images"]["hand_right"]),
        }
        image_sources = {key: key for key in image_specs}

    return {
        "images": image_specs,
        "states": dict(raw_task_config["states"]),
        "actions": dict(raw_task_config["actions"]),
    }, image_sources


def _extra_state_field_name(raw_key: str) -> str:
    return f"agibot.observation_states.{raw_key}"


def _extra_action_field_name(raw_key: str) -> str:
    return f"agibot.actions.{raw_key}"


def _build_extra_field_specs(
    eef_type: str,
    *,
    canonical_image_sources: dict[str, str],
) -> tuple[dict[str, dict[str, Any]], dict[str, str], bool]:
    raw_task_config = convert_agibotworld.AGIBOTWORLD_TASK_CONFIGS[eef_type]
    extra_features: dict[str, dict[str, Any]] = {}
    extra_video_sources: dict[str, str] = {}

    canonical_output_keys = set(canonical_image_sources)
    for raw_key, spec in raw_task_config["images"].items():
        if raw_key == "head_depth":
            extra_features["agibot.images.head_depth"] = dict(spec)
            continue
        if raw_key in canonical_output_keys:
            continue
        extra_features[f"agibot.images.{raw_key}"] = dict(spec)
        extra_video_sources[f"agibot.images.{raw_key}"] = f"observation.images.{raw_key}"

    for raw_key, spec in raw_task_config["states"].items():
        extra_features[_extra_state_field_name(raw_key)] = dict(spec)
    for raw_key, spec in raw_task_config["actions"].items():
        extra_features[_extra_action_field_name(raw_key)] = dict(spec)
    extra_features["agibot.fine_skill"] = {"dtype": "string", "shape": (1,), "names": None}
    return extra_features, extra_video_sources, "head_depth" in raw_task_config["images"]


def _build_features(eef_type: str, fps: int, *, keep_extra_fields: bool) -> dict[str, dict[str, Any]]:
    task_config, image_sources = _output_task_config(eef_type)
    base_features = {
        f"observation.images.{key}": dict(value)
        for key, value in task_config["images"].items()
    }
    if eef_type == "gripper":
        state_shape = (convert_agibotworld.GRIPPER_CANONICAL_STATE_DIM,)
        action_shape = (convert_agibotworld.GRIPPER_CANONICAL_ACTION_DIM,)
        vector_names = list(convert_agibotworld.GRIPPER_CANONICAL_VECTOR_NAMES)
    else:
        state_shape = (sum(int(np.prod(spec["shape"])) for spec in task_config["states"].values()),)
        action_shape = (sum(int(np.prod(spec["shape"])) for spec in task_config["actions"].values()),)
        vector_names = None
    base_features["observation.state"] = {
        "dtype": "float32",
        "shape": state_shape,
        "names": vector_names,
    }
    base_features["actions"] = {
        "dtype": "float32",
        "shape": action_shape,
        "names": vector_names,
    }
    features = convert_agibotworld._add_common_features(base_features)
    if keep_extra_fields:
        extra_features, _extra_video_sources, _has_depth = _build_extra_field_specs(eef_type, canonical_image_sources=image_sources)
        features.update(extra_features)
    return convert_agibotworld._video_info_features(features, fps)


def _flatten_vectors(
    arrays: dict[str, np.ndarray],
    specs: dict[str, dict[str, Any]],
    *,
    prefix: str,
    num_frames: int,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    for key, spec in specs.items():
        values = arrays[f"{prefix}.{key}"]
        if values.size == 0:
            parts.append(np.zeros((num_frames, int(np.prod(spec["shape"]))), dtype=np.float32))
            continue
        values = np.asarray(values, dtype=np.float32)
        if values.ndim == 1:
            values = values[:, np.newaxis]
        parts.append(values.reshape(values.shape[0], -1))
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def _build_skill_labels(num_frames: int, action_config: list[dict[str, Any]]) -> list[str]:
    skills = [""] * num_frames
    for span in sorted(action_config, key=lambda item: (int(item["start_frame"]), int(item["end_frame"]))):
        start = int(span["start_frame"])
        end = int(span["end_frame"])
        skill = str(span.get("skill", "")).strip()
        for frame_idx in range(start, end):
            skills[frame_idx] = skill
    return skills


def _prepare_episode_payload(job: dict[str, Any]) -> dict[str, Any]:
    _ensure_import_paths()

    from openpi.datasets.specs import agibotworld as convert_inner

    src_path = Path(job["src_path"])
    task_id = str(job["task_id"])
    task_stem = str(job["task_stem"])
    eef_type = str(job["eef_type"])
    fps = int(job["fps"])
    image_sources = dict(job["image_sources"])
    keep_extra_fields = bool(job["keep_extra_fields"])
    source_episode_key = str(job["source_episode_key"])

    last_error: str | None = None
    for attempt in range(1, PREPARE_RETRY_ATTEMPTS + 1):
        try:
            selection = convert_inner.EpisodeSelection(
                episode_id=int(job["episode_id"]),
                task_name=str(job["task_name"]),
                init_scene_text=str(job["init_scene_text"]),
                action_config=list(job["action_config"]),
            )

            task_config, _ = _output_task_config(eef_type)
            raw_task_config = convert_inner.AGIBOTWORLD_TASK_CONFIGS[eef_type]
            state_arrays, action_arrays, _depth_paths, raw_videos = convert_inner._load_episode_arrays(
                selection.episode_id,
                src_path,
                task_id,
                raw_task_config,
                save_depth=keep_extra_fields,
            )
            num_frames = len(next(iter(state_arrays.values())))
            tasks_per_frame, unlabeled_frames = convert_inner._build_frame_labels(
                num_frames=num_frames,
                action_config=selection.action_config,
                coarse_task_text=selection.task_name or selection.init_scene_text,
            )
            skills_per_frame = _build_skill_labels(num_frames, selection.action_config) if keep_extra_fields else []
            videos = {
                f"observation.images.{output_key}": raw_videos[f"observation.images.{source_key}"]
                for output_key, source_key in image_sources.items()
            }
            episode_columns: dict[str, Any] = {
                "task": tasks_per_frame,
                "frame_index": np.arange(num_frames, dtype=np.int64),
                "timestamp": np.arange(num_frames, dtype=np.float32) / float(fps),
            }
            if eef_type == "gripper":
                state_value, action_value = convert_inner._derive_gripper_state_and_actions(state_arrays, action_arrays)
                episode_columns["observation.state"] = state_value
                episode_columns["actions"] = action_value
            else:
                episode_columns["observation.state"] = _flatten_vectors(
                    state_arrays,
                    task_config["states"],
                    prefix="observation.states",
                    num_frames=num_frames,
                )
                episode_columns["actions"] = _flatten_vectors(
                    action_arrays,
                    task_config["actions"],
                    prefix="actions",
                    num_frames=num_frames,
                )
            if keep_extra_fields:
                _, extra_video_sources, has_depth = _build_extra_field_specs(eef_type, canonical_image_sources=image_sources)
                for raw_key in raw_task_config["states"]:
                    values = np.asarray(state_arrays[f"observation.states.{raw_key}"], dtype=np.float32)
                    if values.ndim == 1:
                        values = values[:, np.newaxis]
                    episode_columns[_extra_state_field_name(raw_key)] = values
                for raw_key, spec in raw_task_config["actions"].items():
                    values = np.asarray(action_arrays[f"actions.{raw_key}"], dtype=np.float32)
                    if values.size == 0:
                        values = np.zeros((num_frames, *spec["shape"]), dtype=np.float32)
                    if values.ndim == 1:
                        values = values[:, np.newaxis]
                    episode_columns[_extra_action_field_name(raw_key)] = values
                episode_columns["agibot.fine_skill"] = skills_per_frame
                if has_depth:
                    episode_columns["agibot.images.head_depth"] = _depth_paths
                for output_key, raw_video_key in extra_video_sources.items():
                    videos[output_key] = raw_videos[raw_video_key]
            video_durations = convert_inner._compute_episode_video_durations(
                {key: Path(value) for key, value in videos.items()},
                num_frames=num_frames,
                fps=fps,
                episode_id=selection.episode_id,
            )
            return {
                "status": "converted",
                "job_index": int(job["job_index"]),
                "task_stem": task_stem,
                "task_id": task_id,
                "task_name": selection.task_name,
                "episode_id": selection.episode_id,
                "source_episode_key": source_episode_key,
                "frame_count": num_frames,
                "unlabeled_frames": unlabeled_frames,
                "episode_columns": episode_columns,
                "videos": {key: str(value) for key, value in videos.items()},
                "video_durations": video_durations,
            }
        except Exception as exc:
            last_error = f"attempt {attempt}/{PREPARE_RETRY_ATTEMPTS}: {exc}"

    return {
        "status": "failed",
        "job_index": int(job["job_index"]),
        "task_stem": task_stem,
        "task_id": task_id,
        "task_name": str(job["task_name"]),
        "episode_id": int(job["episode_id"]),
        "source_episode_key": source_episode_key,
        "error": last_error or "unknown prepare failure",
    }


@ray.remote(num_cpus=RAY_PREPARE_CPUS)
def _prepare_episode_remote(job: dict[str, Any]) -> dict[str, Any]:
    return _prepare_episode_payload(job)


def _build_episode_buffer(
    dataset: convert_agibotworld.AgiBotWorldDataset,
    payload: dict[str, Any],
    *,
    episode_index: int,
) -> dict[str, Any]:
    columns = dict(payload["episode_columns"])
    episode_data = dataset.create_episode_buffer(episode_index)
    episode_data["size"] = int(payload["frame_count"])
    episode_data["task"] = list(columns.pop("task"))
    for key, value in columns.items():
        episode_data[key] = value
    return episode_data


def _save_prepared_episode_batch(
    dataset: convert_agibotworld.AgiBotWorldDataset,
    payload_batch: list[dict[str, Any]],
    *,
    link_mode: MergeLinkMode = "auto",
    on_episode_saved: Callable[[int, int], None] | None = None,
) -> None:
    start_episode_index = dataset.meta.total_episodes
    episode_batch = [
        _build_episode_buffer(dataset, payload, episode_index=start_episode_index + offset)
        for offset, payload in enumerate(payload_batch)
    ]
    dataset.save_episode_batch(
        episode_batch,
        videos=[{key: Path(value) for key, value in dict(payload["videos"]).items()} for payload in payload_batch],
        video_durations=[{key: float(value) for key, value in dict(payload["video_durations"]).items()} for payload in payload_batch],
        link_mode=link_mode,
        probe_video_durations=False,
        on_episode_saved=on_episode_saved,
    )


@contextmanager
def _scoped_temp_environment(convert_inner: Any, tmp_dir: Path):
    previous_values = {key: os.environ.get(key) for key in ("TMPDIR", "TMP", "TEMP")}
    previous_tempdir = tempfile.tempdir
    tmp_dir.mkdir(parents=True, exist_ok=True)
    convert_inner._set_temp_environment(tmp_dir)
    try:
        yield
    finally:
        for key, value in previous_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        tempfile.tempdir = previous_tempdir


def _chunk_file_indices(path: Path) -> tuple[int, int]:
    return int(path.parent.name.split("-")[-1]), int(path.stem.split("-")[-1])


def _data_file_path(output_root: Path, *, chunk_index: int, file_index: int) -> Path:
    return output_root / f"data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"


def _video_file_path(output_root: Path, *, video_key: str, chunk_index: int, file_index: int) -> Path:
    return output_root / f"videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"


def _read_agibot_episode_metadata_prefix(output_root: Path) -> tuple[list[dict[str, Any]], Path | None]:
    metadata_root = output_root / "meta" / "episodes"
    if not metadata_root.exists():
        return [], None

    rows: list[dict[str, Any]] = []
    first_bad_path: Path | None = None
    columns = [
        "episode_index",
        "length",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
        "dataset_to_index",
        "videos/observation.images.head/chunk_index",
        "videos/observation.images.head/file_index",
        "videos/observation.images.head/from_timestamp",
        "videos/observation.images.head/to_timestamp",
        "videos/observation.images.hand_left/chunk_index",
        "videos/observation.images.hand_left/file_index",
        "videos/observation.images.hand_left/from_timestamp",
        "videos/observation.images.hand_left/to_timestamp",
        "videos/observation.images.hand_right/chunk_index",
        "videos/observation.images.hand_right/file_index",
        "videos/observation.images.hand_right/from_timestamp",
        "videos/observation.images.hand_right/to_timestamp",
        "meta/episodes/chunk_index",
        "meta/episodes/file_index",
    ]
    for path in sorted(metadata_root.rglob("file-*.parquet")):
        try:
            rows.extend(pq.read_table(path, columns=columns).to_pylist())
        except Exception:
            first_bad_path = path
            break
    return rows, first_bad_path


def _first_unreadable_data_row(output_root: Path, rows: list[dict[str, Any]]) -> tuple[int, dict[str, Any], Path] | None:
    checked: dict[Path, bool] = {}
    for idx, row in enumerate(rows):
        data_path = _data_file_path(
            output_root,
            chunk_index=int(row["data/chunk_index"]),
            file_index=int(row["data/file_index"]),
        )
        ok = checked.get(data_path)
        if ok is None:
            ok = data_path.exists()
            if ok:
                try:
                    pq.ParquetFile(data_path)
                except Exception:
                    ok = False
            checked[data_path] = ok
        if not ok:
            return idx, row, data_path
    return None


def _referenced_video_path(row: dict[str, Any], output_root: Path, video_key: str) -> Path:
    return _video_file_path(
        output_root,
        video_key=video_key,
        chunk_index=int(row[f"videos/{video_key}/chunk_index"]),
        file_index=int(row[f"videos/{video_key}/file_index"]),
    )


def _rebuild_missing_video_shards(
    convert_inner: Any,
    *,
    output_root: Path,
    rows: list[dict[str, Any]],
    episode_jobs: list[dict[str, Any]],
    image_sources: dict[str, str],
) -> list[Path]:
    jobs_by_episode_index = {int(job_item["job_index"]): job_item for job_item in episode_jobs}
    missing_groups: dict[tuple[str, Path], list[int]] = {}
    for row in rows:
        episode_index = int(row["episode_index"])
        for output_key in image_sources:
            video_key = f"observation.images.{output_key}"
            shard_path = _referenced_video_path(row, output_root, video_key)
            if not shard_path.exists():
                missing_groups.setdefault((video_key, shard_path), []).append(episode_index)

    rebuilt_paths: list[Path] = []
    for (video_key, shard_path), episode_indices in sorted(missing_groups.items(), key=lambda item: str(item[0][1])):
        source_paths: list[Path] = []
        output_key = video_key.removeprefix("observation.images.")
        source_key = image_sources[output_key]
        for episode_index in episode_indices:
            job_item = jobs_by_episode_index.get(episode_index)
            if job_item is None:
                raise ValueError(
                    f"Cannot repair {shard_path}: missing selected episode metadata for episode_index={episode_index}. "
                    "Re-run with the same task filters."
                )
            ob_dir = Path(job_item["src_path"]) / "observations" / str(job_item["task_id"]) / str(job_item["episode_id"])
            source_path = convert_inner._video_path_for_key(ob_dir, source_key)
            if not source_path.exists():
                raise FileNotFoundError(f"Cannot repair {shard_path}: missing source video {source_path}")
            source_paths.append(source_path)
        CONSOLE.print(f"[yellow]Rebuilding missing video shard:[/yellow] {shard_path}")
        concatenate_video_files(source_paths, shard_path)
        rebuilt_paths.append(shard_path)
    return rebuilt_paths


def _trim_data_file_to_boundary(output_root: Path, last_row: dict[str, Any]) -> bool:
    data_path = _data_file_path(
        output_root,
        chunk_index=int(last_row["data/chunk_index"]),
        file_index=int(last_row["data/file_index"]),
    )
    table = pq.read_table(data_path)
    if "index" not in table.schema.names:
        return False

    boundary = int(last_row["dataset_to_index"])
    index_values = np.asarray(table.column("index").combine_chunks().to_numpy(zero_copy_only=False), dtype=np.int64)
    keep_mask = index_values < boundary
    if keep_mask.all():
        return False

    trimmed = table.filter(pa.array(keep_mask.tolist()))
    tmp_path = data_path.with_suffix(".repair.parquet")
    pq.write_table(trimmed, tmp_path)
    tmp_path.replace(data_path)
    return True


def _delete_numbered_files_after(root: Path, *, suffix: str, max_chunk_index: int, max_file_index: int) -> list[Path]:
    deleted: list[Path] = []
    for path in sorted(root.rglob(f"file-*.{suffix}")):
        chunk_index, file_index = _chunk_file_indices(path)
        if (chunk_index, file_index) > (max_chunk_index, max_file_index):
            path.unlink()
            deleted.append(path)
    return deleted


def _rewrite_agibot_info_counts(output_root: Path, *, total_episodes: int, total_frames: int) -> None:
    info_path = output_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["total_episodes"] = int(total_episodes)
    info["total_frames"] = int(total_frames)
    info["splits"] = {"train": f"0:{int(total_episodes)}"}
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")


def _repair_agibot_resume_output_if_needed(
    convert_inner: Any,
    *,
    output_root: Path,
    episode_jobs: list[dict[str, Any]],
    image_sources: dict[str, str],
) -> dict[str, Any] | None:
    rows, first_bad_meta_path = _read_agibot_episode_metadata_prefix(output_root)
    if not rows:
        if first_bad_meta_path is not None:
            raise ValueError(
                f"Cannot resume {output_root}: no readable episode metadata remain, first bad file is {first_bad_meta_path}. "
                "Remove the broken output root and rebuild with --overwrite."
            )
        return None

    data_issue = _first_unreadable_data_row(output_root, rows)
    if data_issue is not None:
        boundary_rows = rows[: data_issue[0]]
    else:
        boundary_rows = rows
    if not boundary_rows:
        raise ValueError(
            f"Cannot resume {output_root}: the first readable episode metadata row already points to unreadable data. "
            "Remove the broken output root and rebuild with --overwrite."
        )

    rebuilt_video_paths = _rebuild_missing_video_shards(
        convert_inner,
        output_root=output_root,
        rows=boundary_rows,
        episode_jobs=episode_jobs,
        image_sources=image_sources,
    )

    last_row = boundary_rows[-1]
    boundary_episode_count = int(last_row["episode_index"]) + 1
    boundary_frame_count = int(last_row["dataset_to_index"])
    if boundary_episode_count > len(episode_jobs):
        raise ValueError(
            f"Cannot resume {output_root}: existing output already contains {boundary_episode_count} episodes, "
            f"but the current selection only resolves to {len(episode_jobs)} episodes. "
            "Re-run with the same task filters used for the original conversion."
        )
    info = json.loads((output_root / "meta" / "info.json").read_text(encoding="utf-8"))
    checkpoint = EpisodeCheckpointStore(output_root, namespace="openpi_agibot")
    completed_records = checkpoint.load_completed_records()

    extra_data_paths = _delete_numbered_files_after(
        output_root / "data",
        suffix="parquet",
        max_chunk_index=int(last_row["data/chunk_index"]),
        max_file_index=int(last_row["data/file_index"]),
    )
    trimmed_last_data = _trim_data_file_to_boundary(output_root, last_row)

    extra_video_paths: list[Path] = []
    for video_key in (
        "observation.images.head",
        "observation.images.hand_left",
        "observation.images.hand_right",
    ):
        extra_video_paths.extend(
            _delete_numbered_files_after(
                output_root / "videos" / video_key,
                suffix="mp4",
                max_chunk_index=int(last_row[f"videos/{video_key}/chunk_index"]),
                max_file_index=int(last_row[f"videos/{video_key}/file_index"]),
            )
        )

    extra_meta_paths = _delete_numbered_files_after(
        output_root / "meta" / "episodes",
        suffix="parquet",
        max_chunk_index=int(last_row["meta/episodes/chunk_index"]),
        max_file_index=int(last_row["meta/episodes/file_index"]),
    )

    needs_repair = any(
        (
            first_bad_meta_path is not None,
            data_issue is not None,
            rebuilt_video_paths,
            extra_data_paths,
            extra_video_paths,
            extra_meta_paths,
            trimmed_last_data,
            int(info.get("total_episodes", 0)) != boundary_episode_count,
            int(info.get("total_frames", 0)) != boundary_frame_count,
            len(completed_records) != boundary_episode_count,
        )
    )
    if not needs_repair:
        return None

    _rewrite_agibot_info_counts(
        output_root,
        total_episodes=boundary_episode_count,
        total_frames=boundary_frame_count,
    )
    checkpoint.rewrite_completed_records(
        [
            {
                "key": str(job_item["source_episode_key"]),
                "episode_index": int(job_item["job_index"]),
            }
            for job_item in episode_jobs[:boundary_episode_count]
        ]
    )

    for stale_path in (output_root / "meta" / "stats.json", output_root / "norm_stats.json"):
        if stale_path.exists():
            stale_path.unlink()

    summary = {
        "boundary_episode_count": boundary_episode_count,
        "boundary_frame_count": boundary_frame_count,
        "first_bad_meta_path": str(first_bad_meta_path) if first_bad_meta_path is not None else None,
        "first_bad_data_path": str(data_issue[2]) if data_issue is not None else None,
        "rebuilt_video_shards": [str(path) for path in rebuilt_video_paths],
        "deleted_data_files": [str(path) for path in extra_data_paths],
        "deleted_video_files": [str(path) for path in extra_video_paths],
        "deleted_meta_files": [str(path) for path in extra_meta_paths],
        "trimmed_last_data": trimmed_last_data,
    }
    CONSOLE.print(
        progress_display.render_key_value_panel(
            "AgiBot Resume Repair",
            [
                ("Output Dir", output_root),
                ("Recovered Episodes", boundary_episode_count),
                ("Recovered Frames", boundary_frame_count),
                ("Rebuilt Video Shards", len(rebuilt_video_paths)),
                ("Deleted Data Files", len(extra_data_paths)),
                ("Deleted Video Files", len(extra_video_paths)),
                ("Deleted Meta Files", len(extra_meta_paths)),
                ("Trimmed Last Data", trimmed_last_data),
            ],
        )
    )
    return summary


def _build_embodiment_dataset(job: dict[str, Any], *, workers: int) -> dict[str, Any]:
    _ensure_import_paths()

    from openpi.datasets.specs import agibotworld as convert_inner

    src_path = Path(job["src_path"])
    output_root = Path(job["output_root"])
    tmp_dir = Path(job["tmp_dir"])
    eef_type = str(job["eef_type"])
    overwrite = bool(job["overwrite"])
    resume = bool(job.get("resume", False))
    cleanup_resolved_failures = bool(job.get("cleanup_resolved_failures", False))
    fps = int(job["fps"])
    image_sources = dict(job["image_sources"])
    keep_extra_fields = bool(job["keep_extra_fields"])
    episode_commit_batch_size = int(job["episode_commit_batch_size"])
    max_inflight_episodes = int(job["max_inflight_episodes"])
    video_link_mode = str(job["video_link_mode"])
    selected_tasks = [
        convert_inner.TaskSpec(
            json_file=Path(task["json_file"]),
            task_stem=str(task["task_stem"]),
            task_id=str(task["task_id"]),
        )
        for task in job["selected_tasks"]
    ]

    if (overwrite or resume) and tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

    info_path = output_root / "meta" / "info.json"
    if output_root.exists():
        if resume:
            if not info_path.exists():
                raise FileNotFoundError(f"Cannot resume without existing dataset metadata at {info_path}")
        elif overwrite or not info_path.exists():
            shutil.rmtree(output_root)

    temp_env = None
    try:
        temp_env = _scoped_temp_environment(convert_inner, tmp_dir)
        temp_env.__enter__()
        task_totals: dict[str, dict[str, Any]] = {}
        episode_jobs: list[dict[str, Any]] = []
        task_names_in_order: list[str] = []
        job_index = 0
        for task in selected_tasks:
            selections = convert_inner._select_episodes(
                convert_inner._load_task_info(task.json_file),
                episode_offset=int(job["episode_offset"]),
                episodes_per_task=job["episodes_per_task"],
            )
            task_totals[task.task_stem] = {
                "task_stem": task.task_stem,
                "task_id": task.task_id,
                "episodes_selected": len(selections),
                "episodes_written": 0,
                "episodes_failed": 0,
                "frames_written": 0,
            }
            for selection in selections:
                task_names_in_order.append(selection.task_name)
                episode_jobs.append(
                    {
                        "job_index": job_index,
                        "src_path": str(src_path),
                        "task_id": task.task_id,
                        "task_stem": task.task_stem,
                        "episode_id": selection.episode_id,
                        "task_name": selection.task_name,
                        "init_scene_text": selection.init_scene_text,
                        "action_config": selection.action_config,
                        "eef_type": eef_type,
                        "fps": fps,
                        "image_sources": image_sources,
                        "keep_extra_fields": keep_extra_fields,
                        "source_episode_key": _source_episode_key(task.task_id, selection.episode_id),
                    }
                )
                job_index += 1

        repair_summary: dict[str, Any] | None = None
        if resume and info_path.exists():
            repair_summary = _repair_agibot_resume_output_if_needed(
                convert_inner,
                output_root=output_root,
                episode_jobs=episode_jobs,
                image_sources=image_sources,
            )

        if info_path.exists():
            if resume:
                existing_info = json.loads(info_path.read_text(encoding="utf-8"))
                requested_episodes = [0] if int(existing_info.get("total_episodes", 0)) > 0 else None
                dataset = convert_inner.AgiBotWorldDataset(
                    str(job["repo_id"]),
                    root=output_root,
                    episodes=requested_episodes,
                )
            else:
                dataset = convert_inner.AgiBotWorldDataset(str(job["repo_id"]), root=output_root)
        else:
            dataset = convert_inner.AgiBotWorldDataset.create(
                repo_id=str(job["repo_id"]),
                root=output_root,
                fps=fps,
                robot_type="a2d",
                features=dict(job["features"]),
            )
        checkpoint = EpisodeCheckpointStore(output_root, namespace="openpi_agibot")
        completed_keys = checkpoint.load_completed()
        episode_jobs = [job_item for job_item in episode_jobs if str(job_item["source_episode_key"]) not in completed_keys]
        for next_job_index, job_item in enumerate(episode_jobs):
            job_item["job_index"] = next_job_index
        dataset.meta.save_episode_tasks(list(dict.fromkeys(task_names_in_order)))

        episode_count = 0
        frame_count = 0
        unlabeled_frames = 0
        failed_count = 0
        failure_records: list[dict[str, Any]] = []

        def _record_failed_payload(payload: dict[str, Any], *, error: str, stage: str) -> None:
            nonlocal failed_count
            task_summary = task_totals[str(payload["task_stem"])]
            task_summary["episodes_failed"] += 1
            failed_count += 1
            failure_record = {
                "source_episode_key": str(payload["source_episode_key"]),
                "task_stem": str(payload["task_stem"]),
                "task_id": str(payload["task_id"]),
                "episode_id": int(payload["episode_id"]),
                "stage": stage,
                "error": str(error),
            }
            failure_records.append(failure_record)
            checkpoint.append_failure(
                str(payload["source_episode_key"]),
                str(error),
                task_stem=str(payload["task_stem"]),
                task_id=str(payload["task_id"]),
                episode_id=int(payload["episode_id"]),
                stage=stage,
            )

        def _record_saved_payload(payload: dict[str, Any]) -> None:
            nonlocal episode_count, frame_count, unlabeled_frames
            task_summary = task_totals[str(payload["task_stem"])]
            task_summary["episodes_written"] += 1
            task_summary["frames_written"] += int(payload["frame_count"])
            episode_count += 1
            frame_count += int(payload["frame_count"])
            unlabeled_frames += int(payload["unlabeled_frames"])

        def _persist_saved_payloads(saved_payloads: list[tuple[dict[str, Any], int]]) -> None:
            checkpoint_records: list[dict[str, Any]] = []
            for payload, episode_index in saved_payloads:
                _record_saved_payload(payload)
                source_episode_key = str(payload["source_episode_key"])
                if source_episode_key in completed_keys:
                    continue
                checkpoint_records.append(
                    {
                        "key": source_episode_key,
                        "episode_index": int(episode_index),
                    }
                )
            if checkpoint_records:
                checkpoint.append_completed_many(checkpoint_records)
                completed_keys.update(str(record["key"]) for record in checkpoint_records)

        def _commit_payload_batch(payload_batch: list[dict[str, Any]]) -> None:
            if not payload_batch:
                return
            saved_offsets: set[int] = set()
            saved_payloads: list[tuple[dict[str, Any], int]] = []

            def _on_episode_saved(batch_offset: int, episode_index: int) -> None:
                saved_offsets.add(batch_offset)
                saved_payloads.append((payload_batch[batch_offset], episode_index))

            try:
                _save_prepared_episode_batch(
                    dataset,
                    payload_batch,
                    link_mode=video_link_mode,
                    on_episode_saved=_on_episode_saved,
                )
                _persist_saved_payloads(saved_payloads)
                return
            except Exception as exc:
                if saved_payloads:
                    _persist_saved_payloads(saved_payloads)
                if len(payload_batch) == 1:
                    if not saved_offsets:
                        _record_failed_payload(payload_batch[0], error=str(exc), stage="commit")
                    return

            for batch_offset, payload in enumerate(payload_batch):
                if batch_offset in saved_offsets:
                    continue
                try:
                    single_saved_payloads: list[tuple[dict[str, Any], int]] = []
                    _save_prepared_episode_batch(
                        dataset,
                        [payload],
                        link_mode=video_link_mode,
                        on_episode_saved=lambda _offset, episode_index, payload=payload: single_saved_payloads.append(
                            (payload, episode_index)
                        ),
                    )
                    _persist_saved_payloads(single_saved_payloads)
                except Exception as item_exc:
                    _record_failed_payload(payload, error=str(item_exc), stage="commit")

        use_ray = workers > 1 and len(episode_jobs) > 1 and ray.is_initialized()
        with progress_display.create_progress(console=CONSOLE) as progress:
            prepare_task_id = progress.add_task(f"[cyan]Preparing {eef_type} episodes", total=len(episode_jobs))
            finalize_task_id = progress.add_task(f"[green]Finalizing {eef_type} episodes", total=len(episode_jobs))
            committer = OrderedAsyncBatchCommitter(
                _commit_payload_batch,
                max_pending_batches=2,
            )
            try:
                if use_ray:
                    ref_to_index: dict[ray.ObjectRef, int] = {}
                    job_cursor = 0
                    wait_num_returns = max(1, min(8, max_inflight_episodes))
                    while job_cursor < len(episode_jobs) and len(ref_to_index) < max_inflight_episodes:
                        episode_job = episode_jobs[job_cursor]
                        ref_to_index[_prepare_episode_remote.remote(episode_job)] = int(episode_job["job_index"])
                        job_cursor += 1

                    buffered_payloads: dict[int, dict[str, Any]] = {}
                    next_commit_index = 0
                    while ref_to_index:
                        done, _ = ray.wait(
                            list(ref_to_index),
                            num_returns=min(len(ref_to_index), wait_num_returns),
                        )
                        for ref in done:
                            result = ray.get(ref)
                            buffered_payloads[ref_to_index.pop(ref)] = result
                            progress.advance(prepare_task_id)
                            if job_cursor < len(episode_jobs):
                                episode_job = episode_jobs[job_cursor]
                                ref_to_index[_prepare_episode_remote.remote(episode_job)] = int(episode_job["job_index"])
                                job_cursor += 1
                        while True:
                            batch, next_commit_index = pop_contiguous_batch(
                                buffered_payloads,
                                next_index=next_commit_index,
                                batch_size=episode_commit_batch_size,
                            )
                            if not batch:
                                break
                            failed_batch = [item for item in batch if item["status"] != "converted"]
                            if failed_batch:
                                for item in failed_batch:
                                    _record_failed_payload(item, error=str(item["error"]), stage="prepare")
                                progress.advance(finalize_task_id, advance=len(failed_batch))
                                batch = [item for item in batch if item["status"] == "converted"]
                            if batch:
                                progress.advance(finalize_task_id, advance=committer.submit(batch))
                        progress.advance(finalize_task_id, advance=committer.drain_completed())
                else:
                    ready_payloads: list[dict[str, Any]] = []
                    for episode_job in episode_jobs:
                        result = _prepare_episode_payload(episode_job)
                        progress.advance(prepare_task_id)
                        if result["status"] != "converted":
                            _record_failed_payload(result, error=str(result["error"]), stage="prepare")
                            progress.advance(finalize_task_id)
                            continue
                        ready_payloads.append(result)
                        while len(ready_payloads) >= episode_commit_batch_size:
                            batch = ready_payloads[:episode_commit_batch_size]
                            del ready_payloads[:episode_commit_batch_size]
                            progress.advance(finalize_task_id, advance=committer.submit(batch))
                        progress.advance(finalize_task_id, advance=committer.drain_completed())
                    if ready_payloads:
                        progress.advance(finalize_task_id, advance=committer.submit(ready_payloads))
                progress.advance(finalize_task_id, advance=committer.close())
                committer = None
            finally:
                if committer is not None:
                    committer.close()

        dataset.consolidate(run_compute_stats=not bool(job["skip_stats"]))
        cleaned_failure_records = checkpoint.prune_failures_for_completed(completed_keys) if cleanup_resolved_failures else 0
        convert_inner._write_dataset_info_labels(output_root, eef_type=eef_type)
        convert_inner._write_norm_stats(output_root, run_compute_stats=not bool(job["skip_stats"]))
        return {
            "eef_type": eef_type,
            "repo_id": str(job["repo_id"]),
            "output_root": str(output_root),
            "status": "converted",
            "tasks": [task_totals[key] for key in sorted(task_totals)],
            "episodes_written": episode_count,
            "episodes_failed": failed_count,
            "frames_written": frame_count,
            "unlabeled_frames": unlabeled_frames,
            "failures": failure_records,
            "checkpoint_path": str(checkpoint.completed_path),
            "failure_path": str(checkpoint.failed_path),
            "cleaned_failure_records": cleaned_failure_records,
            "resume_repair": repair_summary,
        }
    finally:
        if temp_env is not None:
            temp_env.__exit__(None, None, None)
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _run_embodiment_job(job: dict[str, Any], *, workers: int) -> dict[str, Any]:
    current_job = dict(job)
    last_error: str | None = None
    for _attempt in range(2):
        try:
            return _build_embodiment_dataset(current_job, workers=workers)
        except Exception as exc:
            last_error = str(exc)
            current_job["overwrite"] = False
            current_job["resume"] = True
    return {
        "eef_type": str(job["eef_type"]),
        "repo_id": str(job["repo_id"]),
        "output_root": str(job["output_root"]),
        "status": "failed",
        "tasks": [],
        "episodes_written": 0,
        "episodes_failed": 0,
        "frames_written": 0,
        "unlabeled_frames": 0,
        "error": last_error or "unknown embodiment failure",
        "failures": [],
    }


def main(args: Args) -> None:
    if args.push_to_hub and not args.hub_owner:
        raise ValueError("hub_owner is required when push_to_hub is enabled.")

    src = args.src_path.expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"Source path does not exist: {src}")

    output_root = args.output_dir.expanduser().resolve()
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive.")
    if args.staging_root is not None:
        CONSOLE.print("[yellow]Ignoring --staging-root: shard mode writes directly to --output-dir.")
    if args.video_link_mode != "copy":
        CONSOLE.print("[yellow]Ignoring --video-link-mode: conversion now always writes real shard files.")
    build_root = output_root
    using_staging = False
    if output_root.exists() and not (args.overwrite or args.resume):
        raise FileExistsError(f"Output already exists: {output_root}")
    if build_root.exists() and args.overwrite:
        staging.remove_tree(build_root)
    build_root.mkdir(parents=True, exist_ok=True)
    ray_temp_dir = ray_runtime.resolve_ray_temp_dir(
        label="agibot-v3",
        unique_key=str(build_root),
        requested_root=args.ray_temp_root,
    )
    ray_runtime.configure_process_temp_dir(ray_temp_dir)
    temp_root = args.tmp_dir.expanduser().resolve() if args.tmp_dir else (ray_temp_dir / "local")
    temp_root.mkdir(parents=True, exist_ok=True)

    requested_task_ids = _load_requested_task_ids(args)
    jobs: list[dict[str, Any]] = []
    total_selected_tasks = 0
    total_selected_episodes = 0
    fps = 30
    for eef_type in _resolve_eef_types(args.eef_type):
        selected_tasks = _select_tasks(
            src,
            eef_type=eef_type,
            explicit_task_ids=requested_task_ids,
            start_task_id=args.start_task_id,
            max_tasks=args.max_tasks,
        )
        if not selected_tasks:
            continue
        selected_episode_count = sum(
            len(
                convert_agibotworld._select_episodes(
                    convert_agibotworld._load_task_info(task.json_file),
                    episode_offset=args.episode_offset,
                    episodes_per_task=args.episodes_per_task,
                )
            )
            for task in selected_tasks
        )
        _, image_sources = _output_task_config(eef_type)
        features = _build_features(eef_type, fps, keep_extra_fields=args.keep_extra_fields)
        total_selected_tasks += len(selected_tasks)
        total_selected_episodes += selected_episode_count
        jobs.append(
            {
                "src_path": str(src),
                "output_root": str(build_root / f"agibot_{eef_type}"),
                "tmp_dir": str(temp_root / eef_type),
                "eef_type": eef_type,
                "repo_id": f"agibot_{eef_type}",
                "features": features,
                "image_sources": image_sources,
                "selected_tasks": [
                    {"json_file": str(task.json_file), "task_stem": task.task_stem, "task_id": task.task_id}
                    for task in selected_tasks
                ],
                "overwrite": args.overwrite,
                "skip_stats": args.skip_stats,
                "episode_offset": args.episode_offset,
                "episodes_per_task": args.episodes_per_task,
                "fps": fps,
                "selected_task_count": len(selected_tasks),
                "selected_episode_count": selected_episode_count,
                "keep_extra_fields": args.keep_extra_fields,
                "resume": args.resume,
                "cleanup_resolved_failures": args.cleanup_resolved_failures,
                "video_link_mode": args.video_link_mode,
            }
        )

    if not jobs:
        raise ValueError("No AgiBot tasks matched the requested selection.")

    workers = _resolve_conversion_workers(args.conversion_num_workers, total_selected_episodes)
    if not _HAS_RAY and workers > 1 and total_selected_episodes > 1:
        CONSOLE.print("[yellow]ray is not installed; falling back to serial conversion (workers=1).[/yellow]")
        workers = 1
    episode_commit_batch_size = _resolve_episode_commit_batch_size(
        args.episode_commit_batch_size,
        workers=workers,
        episode_count=total_selected_episodes,
    )
    for job in jobs:
        job["episode_commit_batch_size"] = episode_commit_batch_size
        job["max_inflight_episodes"] = resolve_max_inflight_episodes(
            args.max_inflight_episodes,
            workers=workers,
            episode_count=int(job["selected_episode_count"]),
            prepare_task_cpus=RAY_PREPARE_CPUS,
        )
    CONSOLE.print(
        progress_display.render_key_value_panel(
            "AgiBotWorld v3 Conversion",
            [
                ("Source", src),
                ("Output Dir", output_root),
                ("Build Root", build_root if using_staging else "in-place"),
                ("Embodiments", ", ".join(job["eef_type"] for job in jobs)),
                ("Selected Tasks", total_selected_tasks),
                ("Selected Episodes", total_selected_episodes),
                ("Keep Extra Fields", args.keep_extra_fields),
                ("Ray Workers", workers),
                ("Ray Prepare Task CPUs", RAY_PREPARE_CPUS),
                ("Max In-Flight Episodes", max(int(job["max_inflight_episodes"]) for job in jobs)),
                ("Episode Commit Batch", episode_commit_batch_size),
                ("Ray Temp", ray_temp_dir),
            ],
        )
    )

    if workers > 1 and total_selected_episodes > 1:
        ray_runtime.init_ray(num_workers=workers, temp_dir=ray_temp_dir, python_paths=PYTHON_PATHS)

    results = []
    try:
        for job in jobs:
            results.append(_run_embodiment_job(job, workers=workers))
    finally:
        if ray.is_initialized():
            ray.shutdown()

    if temp_root.is_dir():
        shutil.rmtree(temp_root, ignore_errors=True)

    results.sort(key=lambda item: str(item["eef_type"]))
    hub_results: list[dict[str, Any]] = []
    if args.push_to_hub:
        with progress_display.create_progress(console=CONSOLE) as progress:
            push_task_id = progress.add_task("[yellow]Pushing datasets to Hub", total=len(results))
            for result in results:
                if result["status"] != "converted":
                    hub_results.append(
                        {
                            "eef_type": result["eef_type"],
                            "status": "skipped",
                            "reason": f"dataset status is {result['status']}",
                        }
                    )
                    progress.advance(push_task_id)
                    continue
                repo_id = hub_push.build_repo_id(args.hub_owner or "", "agibotworld-v3", str(result["eef_type"]))
                try:
                    hub_push.push_dataset_root_to_hub(
                        Path(result["output_root"]),
                        repo_id,
                        tags=tuple(dict.fromkeys(("agibotworld", str(result["eef_type"])))),
                        license="cc-by-nc-sa-4.0",
                        push_videos=True,
                    )
                    hub_results.append({"eef_type": result["eef_type"], "repo_id": repo_id, "status": "pushed"})
                except Exception as exc:
                    hub_results.append(
                        {"eef_type": result["eef_type"], "repo_id": repo_id, "status": "failed", "error": str(exc)}
                    )
                progress.advance(push_task_id)
    summary = {
        "src_path": str(src),
        "output_root": str(output_root),
        "build_root": str(build_root),
        "results": results,
        "hub_push": hub_results,
        "keep_extra_fields": args.keep_extra_fields,
        "max_inflight_episodes": max(int(job["max_inflight_episodes"]) for job in jobs),
        "episode_commit_batch_size": episode_commit_batch_size,
        "totals": {
            "datasets_written": sum(1 for item in results if item["status"] == "converted"),
            "episodes_written": sum(int(item["episodes_written"]) for item in results),
            "frames_written": sum(int(item["frames_written"]) for item in results),
            "cleaned_failure_records": sum(int(item.get("cleaned_failure_records", 0)) for item in results),
        },
    }
    summary_path = args.summary_json.expanduser().resolve() if args.summary_json else (output_root / "build_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.cleanup_tmp_on_success:
        cleaned = ray_runtime.cleanup_temp_paths([ray_temp_dir])
        if cleaned:
            CONSOLE.print(f"[green]Cleaned temporary paths:[/green] {', '.join(str(path) for path in cleaned)}")
    CONSOLE.print(
        progress_display.render_summary_table(
            "AgiBotWorld v3 Summary",
            [
                ("Datasets written", summary["totals"]["datasets_written"]),
                ("Episodes written", summary["totals"]["episodes_written"]),
                ("Frames written", summary["totals"]["frames_written"]),
                ("Resolved failures cleaned", summary["totals"]["cleaned_failure_records"]),
                ("Summary path", summary_path),
            ],
        )
    )

if __name__ == "__main__":
    main(tyro.cli(Args))
