# Copyright 2026 PLaW-VLA authors. SPDX-License-Identifier: Apache-2.0
# LeRobot-derived writer portions: Copyright Hugging Face contributors (Apache-2.0).
# Modified for PLaW-VLA; see NOTICE.
from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from dataclasses import dataclass, field
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any

from datasets import Dataset
import jsonlines
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.io_utils import cast_stats_to_numpy
from lerobot.datasets.io_utils import get_file_size_in_mb
from lerobot.datasets.io_utils import load_info
from lerobot.datasets.io_utils import write_episodes
from lerobot.datasets.io_utils import write_info
from lerobot.datasets.io_utils import write_stats
from lerobot.datasets.io_utils import write_tasks
from lerobot.datasets.utils import DEFAULT_CHUNK_SIZE
from lerobot.datasets.utils import DEFAULT_DATA_FILE_SIZE_IN_MB
from lerobot.datasets.utils import DEFAULT_DATA_PATH
from lerobot.datasets.utils import DEFAULT_VIDEO_FILE_SIZE_IN_MB
from lerobot.datasets.utils import DEFAULT_VIDEO_PATH
from lerobot.datasets.utils import LEGACY_EPISODES_PATH
from lerobot.datasets.utils import LEGACY_EPISODES_STATS_PATH
from lerobot.datasets.utils import LEGACY_TASKS_PATH
from lerobot.datasets.utils import flatten_dict
from lerobot.datasets.utils import update_chunk_file_indices
from lerobot.datasets.video_utils import get_video_duration_in_s
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from openpi.datasets.common import staging
from openpi.datasets.common.lerobot_v3 import MergeLinkMode
from openpi.datasets.common.video_concat import concatenate_video_files
from openpi.shared import normalize

V21 = "v2.1"
V30 = "v3.0"
FinalizeProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class V21EpisodeBundle:
    source_root: Path
    source_episode_index: int
    source_data_path: Path
    data_size_mb: float
    num_frames: int
    episode: dict[str, Any]
    stats: dict[str, Any]
    video_paths: dict[str, Path]
    video_sizes_mb: dict[str, float]
    video_durations_s: dict[str, float]


@dataclass(frozen=True)
class VectorElementRemap:
    column_name: str
    element_index: int
    mapping: tuple[tuple[float, float], ...]
    atol: float = 1e-6


@dataclass(frozen=True)
class V21DatasetBundle:
    root: Path
    info: dict[str, Any]
    tasks: dict[int, str]
    episodes: tuple[V21EpisodeBundle, ...]
    video_keys: tuple[str, ...]
    max_data_file_mb: int
    max_video_file_mb: int
    column_renames: dict[str, str] = field(default_factory=dict)
    vector_element_remaps: tuple[VectorElementRemap, ...] = field(default_factory=tuple)

    @property
    def total_frames(self) -> int:
        return sum(episode.num_frames for episode in self.episodes)


@dataclass(frozen=True)
class OutputEpisode:
    output_episode_index: int
    source: V21EpisodeBundle
    task_index_remap: dict[int, int]
    episode_metadata: dict[str, Any]


def _load_jsonlines(path: Path) -> list[dict[str, Any]]:
    with jsonlines.open(path, "r") as reader:
        return list(reader)


def _validate_v21_root(root: Path) -> dict[str, Any]:
    info = load_info(root)
    version = info.get("codebase_version")
    if version != V21:
        raise ValueError(f"Expected a LeRobot {V21} dataset at {root}, found {version!r}.")
    return info


def _legacy_tasks(root: Path) -> tuple[dict[int, str], dict[str, int]]:
    rows = _load_jsonlines(root / LEGACY_TASKS_PATH)
    mapping = {int(row["task_index"]): str(row["task"]) for row in rows}
    inverse = {task: task_index for task_index, task in mapping.items()}
    return mapping, inverse


def _legacy_episodes(root: Path) -> dict[int, dict[str, Any]]:
    rows = _load_jsonlines(root / LEGACY_EPISODES_PATH)
    return {int(row["episode_index"]): row for row in rows}


def _legacy_episode_stats(root: Path) -> dict[int, dict[str, Any]]:
    rows = _load_jsonlines(root / LEGACY_EPISODES_STATS_PATH)
    return {int(row["episode_index"]): cast_stats_to_numpy(row["stats"]) for row in rows}


def _video_keys(info: dict[str, Any]) -> list[str]:
    return sorted(
        key
        for key, feature in info.get("features", {}).items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    )


def _iter_v21_episode_files(root: Path) -> list[tuple[int, Path]]:
    return [
        (int(parquet_path.stem.split("_")[-1]), parquet_path)
        for parquet_path in sorted((root / "data").glob("chunk-*/episode_*.parquet"))
    ]


def _build_v3_info(
    info: dict[str, Any],
    *,
    data_budget_mb: int,
    video_budget_mb: int,
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
) -> dict[str, Any]:
    updated = dict(info)
    updated["codebase_version"] = V30
    updated["data_path"] = DEFAULT_DATA_PATH
    updated["video_path"] = DEFAULT_VIDEO_PATH if updated.get("video_path") is not None else None
    updated["fps"] = int(updated["fps"])
    updated["total_episodes"] = total_episodes
    updated["total_frames"] = total_frames
    updated["total_tasks"] = total_tasks
    updated["data_files_size_in_mb"] = data_budget_mb
    updated["video_files_size_in_mb"] = video_budget_mb
    updated.pop("total_chunks", None)
    updated.pop("total_videos", None)
    for feature in updated.get("features", {}).values():
        if feature.get("dtype") != "video":
            feature["fps"] = updated["fps"]
    return updated


def _write_task_mapping(tasks: dict[int, str], out_root: Path) -> None:
    task_indices = tasks.keys()
    task_strings = tasks.values()
    task_df = pd.DataFrame({"task_index": task_indices}, index=pd.Index(task_strings, name="task"))
    write_tasks(task_df, out_root)


def _write_episode_rows(
    out_root: Path,
    *,
    episode_rows: list[dict[str, Any]],
    episode_stats: list[dict[str, Any]],
) -> None:
    if len(episode_rows) != len(episode_stats):
        raise ValueError(
            f"Expected the same number of episode rows and stats entries, got {len(episode_rows)} and {len(episode_stats)}."
        )

    def _rows():
        for row, stats in zip(episode_rows, episode_stats, strict=True):
            yield {
                **row,
                **flatten_dict({"stats": stats}),
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
            }

    cache_root = staging.default_staging_base() / "hf-datasets"
    cache_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="openpi-hf-datasets-", dir=cache_root) as cache_dir:
        write_episodes(Dataset.from_generator(_rows, cache_dir=cache_dir), out_root)
    write_stats(aggregate_stats(episode_stats), out_root)


def _build_video_lookup(root: Path, video_keys: tuple[str, ...]) -> dict[str, dict[int, Path]]:
    lookup: dict[str, dict[int, Path]] = {video_key: {} for video_key in video_keys}
    for video_key in video_keys:
        for source_path in sorted((root / "videos").glob(f"chunk-*/{video_key}/episode_*.mp4")):
            episode_index = int(source_path.stem.split("_")[-1])
            lookup[video_key][episode_index] = source_path
    return lookup


def load_v21_dataset_bundle(root: Path) -> V21DatasetBundle:
    root = root.expanduser().resolve()
    info = _validate_v21_root(root)
    tasks, _ = _legacy_tasks(root)
    legacy_episodes = _legacy_episodes(root)
    legacy_stats = _legacy_episode_stats(root)
    video_keys = tuple(_video_keys(info))
    video_lookup = _build_video_lookup(root, video_keys)

    episode_bundles: list[V21EpisodeBundle] = []
    max_data_file_mb = DEFAULT_DATA_FILE_SIZE_IN_MB
    max_video_file_mb = DEFAULT_VIDEO_FILE_SIZE_IN_MB
    for episode_index, source_data_path in _iter_v21_episode_files(root):
        if episode_index not in legacy_episodes:
            raise ValueError(f"Episode {episode_index} is missing from {root / LEGACY_EPISODES_PATH}.")
        if episode_index not in legacy_stats:
            raise ValueError(f"Episode {episode_index} is missing from {root / LEGACY_EPISODES_STATS_PATH}.")

        num_frames = pq.read_metadata(source_data_path).num_rows
        data_size_mb = get_file_size_in_mb(source_data_path)
        max_data_file_mb = max(max_data_file_mb, math.ceil(data_size_mb))

        episode_video_paths: dict[str, Path] = {}
        episode_video_sizes: dict[str, float] = {}
        episode_video_durations: dict[str, float] = {}
        for video_key in video_keys:
            video_path = video_lookup[video_key].get(episode_index)
            if video_path is None:
                raise FileNotFoundError(f"Missing source video for {video_key} episode {episode_index} under {root}.")
            episode_video_paths[video_key] = video_path
            episode_video_sizes[video_key] = get_file_size_in_mb(video_path)
            episode_video_durations[video_key] = float(get_video_duration_in_s(video_path))
            max_video_file_mb = max(max_video_file_mb, math.ceil(episode_video_sizes[video_key]))

        episode_bundles.append(
            V21EpisodeBundle(
                source_root=root,
                source_episode_index=episode_index,
                source_data_path=source_data_path,
                data_size_mb=data_size_mb,
                num_frames=num_frames,
                episode=dict(legacy_episodes[episode_index]),
                stats=legacy_stats[episode_index],
                video_paths=episode_video_paths,
                video_sizes_mb=episode_video_sizes,
                video_durations_s=episode_video_durations,
            )
        )

    return V21DatasetBundle(
        root=root,
        info=info,
        tasks=tasks,
        episodes=tuple(episode_bundles),
        video_keys=video_keys,
        max_data_file_mb=max_data_file_mb,
        max_video_file_mb=max_video_file_mb,
    )


def _rewrite_table_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    column_index = table.schema.get_field_index(name)
    if column_index < 0:
        return table
    field = table.schema.field(column_index)
    array_values = values.tolist() if isinstance(values, np.ndarray) and values.ndim > 1 else values
    return table.set_column(column_index, field, pa.array(array_values, type=field.type))


def apply_vector_element_remaps_to_table(
    table: pa.Table,
    remaps: tuple[VectorElementRemap, ...] | None = None,
) -> pa.Table:
    if not remaps:
        return table

    for remap in remaps:
        column_index = table.schema.get_field_index(remap.column_name)
        if column_index < 0:
            continue

        values = _column_to_numpy(table.column(remap.column_name))
        if values.dtype == object:
            values = np.asarray(values.tolist())
        if values.ndim < 2:
            raise ValueError(
                f"Vector element remap requires a vector-valued column, got {remap.column_name!r} with shape {values.shape}."
            )
        if remap.element_index < 0 or remap.element_index >= values.shape[-1]:
            raise ValueError(
                f"Vector element remap index {remap.element_index} is out of bounds for "
                f"{remap.column_name!r} with shape {values.shape}."
            )

        updated = np.array(values, copy=True)
        source_values = np.array(updated[..., remap.element_index], copy=True)
        remapped_values = np.array(source_values, copy=True)
        matched = np.zeros(source_values.shape, dtype=bool)
        for source_value, target_value in remap.mapping:
            current = np.isclose(source_values, source_value, rtol=0.0, atol=remap.atol)
            remapped_values[current] = target_value
            matched |= current

        if not np.all(matched):
            unmatched_values = np.unique(source_values[~matched]).tolist()
            raise ValueError(
                f"Vector element remap for {remap.column_name!r}[{remap.element_index}] did not match "
                f"values {unmatched_values}."
            )

        updated[..., remap.element_index] = remapped_values.astype(updated.dtype, copy=False)
        table = _rewrite_table_column(table, remap.column_name, updated)

    return table


def _is_list_like_arrow_type(field_type: pa.DataType) -> bool:
    return (
        pa.types.is_list(field_type)
        or pa.types.is_large_list(field_type)
        or pa.types.is_fixed_size_list(field_type)
    )


def _arrow_list_leaf_type(field_type: pa.DataType) -> pa.DataType:
    current = field_type
    while _is_list_like_arrow_type(current):
        current = current.value_type
    return current


def _arrow_type_from_shape(leaf_type: pa.DataType, shape: tuple[int, ...]) -> pa.DataType:
    if not shape:
        return leaf_type
    return pa.list_(_arrow_type_from_shape(leaf_type, shape[1:]), int(shape[0]))


def _normalize_table_to_feature_schema(
    table: pa.Table,
    *,
    feature_specs: dict[str, Any],
) -> pa.Table:
    for name in table.schema.names:
        if name not in feature_specs:
            continue
        field_index = table.schema.get_field_index(name)
        if field_index < 0:
            continue

        current_field = table.schema.field(field_index)
        if not _is_list_like_arrow_type(current_field.type):
            continue

        shape = tuple(int(dim) for dim in feature_specs[name].get("shape") or ())
        if not shape:
            continue

        target_type = _arrow_type_from_shape(_arrow_list_leaf_type(current_field.type), shape)
        if current_field.type == target_type:
            continue

        values = table.column(name).combine_chunks().to_pylist()
        table = table.set_column(
            field_index,
            pa.field(name, target_type, nullable=current_field.nullable, metadata=current_field.metadata),
            pa.array(values, type=target_type),
        )

    return table


def _column_to_numpy(column: pa.ChunkedArray) -> np.ndarray:
    array = column.combine_chunks() if hasattr(column, "combine_chunks") else column
    if hasattr(array, "to_numpy"):
        try:
            return np.asarray(array.to_numpy(zero_copy_only=False))
        except TypeError:
            return np.asarray(array.to_numpy())
    return np.asarray(array.to_pylist())


def _column_to_matrix(column: pa.ChunkedArray) -> np.ndarray:
    values = _column_to_numpy(column)
    if values.dtype == object:
        values = np.asarray(values.tolist(), dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    return values


def is_v3_dataset_root(dataset_root: Path) -> bool:
    return (dataset_root / "meta" / "info.json").exists() and any((dataset_root / "data").glob("chunk-*/file-*.parquet"))


def compute_output_norm_stats(dataset_root: Path) -> tuple[dict[str, normalize.NormStats], str | None]:
    stats = {
        "state": normalize.RunningStats(),
        "actions": normalize.RunningStats(),
    }
    seen: set[str] = set()
    action_key_used: str | None = None

    for parquet_path in sorted(dataset_root.glob("data/chunk-*/file-*.parquet")):
        table = pq.read_table(parquet_path)
        if "observation.state" in table.schema.names:
            stats["state"].update(_column_to_matrix(table.column("observation.state")))
            seen.add("state")
        if "actions" in table.schema.names:
            stats["actions"].update(_column_to_matrix(table.column("actions")))
            seen.add("actions")
            action_key_used = "actions"
        elif "action" in table.schema.names:
            stats["actions"].update(_column_to_matrix(table.column("action")))
            seen.add("actions")
            action_key_used = "action"

    return {key: stats[key].get_statistics() for key in sorted(seen)}, action_key_used


def write_output_norm_stats(dataset_root: Path, *, run_compute_stats: bool) -> dict[str, Any] | None:
    if not run_compute_stats:
        return None
    norm_stats, action_key_used = compute_output_norm_stats(dataset_root)
    if not norm_stats:
        return None
    normalize.save(dataset_root, norm_stats)
    return {
        "keys": sorted(norm_stats),
        "action_key": action_key_used,
    }


def _rewrite_episode_table(
    source_path: Path,
    *,
    output_episode_index: int,
    dataset_from_index: int,
    task_index_remap: dict[int, int],
    feature_specs: dict[str, Any] | None = None,
    column_renames: dict[str, str] | None = None,
    vector_element_remaps: tuple[VectorElementRemap, ...] | None = None,
) -> pa.Table:
    table = pq.read_table(source_path)
    num_rows = table.num_rows
    if num_rows <= 0:
        raise ValueError(f"Episode parquet is empty: {source_path}")

    table = apply_vector_element_remaps_to_table(table, vector_element_remaps)

    if column_renames:
        renamed_columns = [column_renames.get(name, name) for name in table.schema.names]
        if renamed_columns != list(table.schema.names):
            table = table.rename_columns(renamed_columns)

    if feature_specs:
        table = _normalize_table_to_feature_schema(table, feature_specs=feature_specs)

    table = _rewrite_table_column(
        table,
        "episode_index",
        np.full(num_rows, output_episode_index, dtype=np.int64),
    )
    table = _rewrite_table_column(
        table,
        "index",
        np.arange(dataset_from_index, dataset_from_index + num_rows, dtype=np.int64),
    )
    if "task_index" in table.schema.names:
        old_task_index = _column_to_numpy(table.column("task_index")).astype(np.int64, copy=False)
        remapped_task_index = np.asarray(
            [task_index_remap[int(task_index)] for task_index in old_task_index],
            dtype=np.int64,
        )
        table = _rewrite_table_column(table, "task_index", remapped_task_index)

    return table


def _advance_progress(
    callback: FinalizeProgressCallback | None,
    *,
    completed: int,
    total: int,
    advance: int = 1,
) -> int:
    completed += advance
    if callback is not None:
        callback(completed, total)
    return completed


def _build_output_episodes(
    bundles: list[V21DatasetBundle],
    *,
    merged_tasks: dict[int, str],
    task_index_maps: dict[Path, dict[int, int]],
) -> list[OutputEpisode]:
    output_episodes: list[OutputEpisode] = []
    for bundle in bundles:
        task_index_remap = task_index_maps[bundle.root]
        for source_episode in bundle.episodes:
            metadata = dict(source_episode.episode)
            metadata["episode_index"] = len(output_episodes)
            if "task_index" in metadata:
                metadata["task_index"] = task_index_remap[int(metadata["task_index"])]
            output_episodes.append(
                OutputEpisode(
                    output_episode_index=len(output_episodes),
                    source=source_episode,
                    task_index_remap=task_index_remap,
                    episode_metadata=metadata,
                )
            )
    if not merged_tasks:
        raise ValueError("Expected at least one task in the converted dataset.")
    return output_episodes


def _write_sharded_data_files(
    episodes: list[OutputEpisode],
    out_root: Path,
    *,
    data_file_size_in_mb: int,
    source_feature_maps: dict[Path, dict[str, Any]],
    source_column_renames: dict[Path, dict[str, str]],
    source_vector_element_remaps: dict[Path, tuple[VectorElementRemap, ...]],
    on_episode_processed: Callable[[], None] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    chunk_idx = 0
    file_idx = 0
    size_in_mb = 0.0
    frame_offset = 0
    pending_tables: list[pa.Table] = []
    per_episode_metadata: list[dict[str, Any]] = []

    def flush_pending() -> None:
        if not pending_tables:
            return
        destination_path = out_root / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        output_table = pending_tables[0] if len(pending_tables) == 1 else pa.concat_tables(pending_tables)
        pq.write_table(output_table, destination_path)
        pending_tables.clear()

    for episode in episodes:
        if size_in_mb + episode.source.data_size_mb >= data_file_size_in_mb and pending_tables:
            flush_pending()
            chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, DEFAULT_CHUNK_SIZE)
            size_in_mb = 0.0

        per_episode_metadata.append(
            {
                "episode_index": episode.output_episode_index,
                "data/chunk_index": chunk_idx,
                "data/file_index": file_idx,
                "dataset_from_index": frame_offset,
                "dataset_to_index": frame_offset + episode.source.num_frames,
            }
        )
        pending_tables.append(
            _rewrite_episode_table(
                episode.source.source_data_path,
                output_episode_index=episode.output_episode_index,
                dataset_from_index=frame_offset,
                task_index_remap=episode.task_index_remap,
                feature_specs=source_feature_maps.get(episode.source.source_root),
                column_renames=source_column_renames.get(episode.source.source_root),
                vector_element_remaps=source_vector_element_remaps.get(episode.source.source_root),
            )
        )
        size_in_mb += episode.source.data_size_mb
        frame_offset += episode.source.num_frames
        if on_episode_processed is not None:
            on_episode_processed()

    flush_pending()
    return per_episode_metadata, frame_offset


def _write_output_video_shard(output_path: Path, input_paths: list[Path]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(input_paths) == 1:
        shutil.copy2(input_paths[0], output_path)
        return
    concatenate_video_files(input_paths, output_path)


def _write_sharded_video_key_files(
    episodes: list[OutputEpisode],
    out_root: Path,
    *,
    video_key: str,
    video_file_size_in_mb: int,
) -> dict[int, dict[str, Any]]:
    per_episode: dict[int, dict[str, Any]] = {}
    chunk_idx = 0
    file_idx = 0
    size_in_mb = 0.0
    duration_s = 0.0
    pending_paths: list[Path] = []

    for episode in episodes:
        episode_size_mb = episode.source.video_sizes_mb[video_key]
        episode_duration_s = episode.source.video_durations_s[video_key]
        if size_in_mb + episode_size_mb >= video_file_size_in_mb and pending_paths:
            output_path = out_root / DEFAULT_VIDEO_PATH.format(
                video_key=video_key,
                chunk_index=chunk_idx,
                file_index=file_idx,
            )
            _write_output_video_shard(output_path, pending_paths)
            chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, DEFAULT_CHUNK_SIZE)
            size_in_mb = 0.0
            duration_s = 0.0
            pending_paths = []

        per_episode[episode.output_episode_index] = {
            f"videos/{video_key}/chunk_index": chunk_idx,
            f"videos/{video_key}/file_index": file_idx,
            f"videos/{video_key}/from_timestamp": duration_s,
            f"videos/{video_key}/to_timestamp": duration_s + episode_duration_s,
        }
        pending_paths.append(episode.source.video_paths[video_key])
        size_in_mb += episode_size_mb
        duration_s += episode_duration_s

    if pending_paths:
        output_path = out_root / DEFAULT_VIDEO_PATH.format(
            video_key=video_key,
            chunk_index=chunk_idx,
            file_index=file_idx,
        )
        _write_output_video_shard(output_path, pending_paths)

    return per_episode


def _write_sharded_video_files(
    episodes: list[OutputEpisode],
    out_root: Path,
    *,
    video_keys: tuple[str, ...],
    video_file_size_in_mb: int,
    worker_threads: int | None = None,
    on_episode_processed: Callable[[], None] | None = None,
) -> dict[int, dict[str, Any]]:
    if not video_keys:
        return {}

    per_episode: dict[int, dict[str, Any]] = {}
    if len(video_keys) == 1:
        key_metadata = _write_sharded_video_key_files(
            episodes,
            out_root,
            video_key=video_keys[0],
            video_file_size_in_mb=video_file_size_in_mb,
        )
        for episode_index, metadata in key_metadata.items():
            per_episode.setdefault(episode_index, {}).update(metadata)
        if on_episode_processed is not None:
            for _ in episodes:
                on_episode_processed()
        return per_episode

    max_workers = max(1, min(worker_threads or len(video_keys), len(video_keys), 8))
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="openpi-v21-video") as executor:
        futures = {
            executor.submit(
                _write_sharded_video_key_files,
                episodes,
                out_root,
                video_key=video_key,
                video_file_size_in_mb=video_file_size_in_mb,
            ): video_key
            for video_key in video_keys
        }
        for future in as_completed(futures):
            key_metadata = future.result()
            for episode_index, metadata in key_metadata.items():
                per_episode.setdefault(episode_index, {}).update(metadata)
            if on_episode_processed is not None:
                for _ in episodes:
                    on_episode_processed()

    return per_episode


def _build_episode_rows(
    episodes: list[OutputEpisode],
    *,
    data_metadata: list[dict[str, Any]],
    video_metadata: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data_by_episode = {int(item["episode_index"]): item for item in data_metadata}
    episode_rows: list[dict[str, Any]] = []
    episode_stats: list[dict[str, Any]] = []
    for episode in episodes:
        episode_rows.append(
            {
                **data_by_episode[episode.output_episode_index],
                **video_metadata.get(episode.output_episode_index, {}),
                **episode.episode_metadata,
            }
        )
        episode_stats.append(episode.source.stats)
    return episode_rows, episode_stats


def estimate_convert_finalize_steps(bundle: V21DatasetBundle) -> int:
    return len(bundle.episodes) * (len(bundle.video_keys) + 1) + 3


def estimate_merge_finalize_steps(bundles: list[V21DatasetBundle]) -> int:
    return sum(len(bundle.episodes) * (len(bundle.video_keys) + 1) for bundle in bundles) + 3


def _prepare_output_episodes_for_convert(bundle: V21DatasetBundle) -> tuple[list[OutputEpisode], dict[int, str]]:
    task_index_remap = {task_index: task_index for task_index in bundle.tasks}
    episodes = _build_output_episodes(
        [bundle],
        merged_tasks=bundle.tasks,
        task_index_maps={bundle.root: task_index_remap},
    )
    return episodes, bundle.tasks


def _validate_mergeable_bundles(bundles: list[V21DatasetBundle]) -> None:
    if not bundles:
        raise ValueError("Expected at least one v2.1 dataset bundle.")

    reference = bundles[0]
    for bundle in bundles[1:]:
        if bundle.video_keys != reference.video_keys:
            raise ValueError(
                f"Video key mismatch between {reference.root} and {bundle.root}: "
                f"{reference.video_keys} != {bundle.video_keys}"
            )
        if bundle.info.get("fps") != reference.info.get("fps"):
            raise ValueError(f"FPS mismatch between {reference.root} and {bundle.root}.")
        if bundle.info.get("features") != reference.info.get("features"):
            raise ValueError(f"Feature schema mismatch between {reference.root} and {bundle.root}.")


def _prepare_output_episodes_for_merge(
    bundles: list[V21DatasetBundle],
) -> tuple[list[OutputEpisode], dict[int, str]]:
    task_to_index: dict[str, int] = {}
    merged_tasks: dict[int, str] = {}
    task_index_maps: dict[Path, dict[int, int]] = {}
    for bundle in bundles:
        remap: dict[int, int] = {}
        for old_task_index in sorted(bundle.tasks):
            task = bundle.tasks[old_task_index]
            if task not in task_to_index:
                new_task_index = len(task_to_index)
                task_to_index[task] = new_task_index
                merged_tasks[new_task_index] = task
            remap[old_task_index] = task_to_index[task]
        task_index_maps[bundle.root] = remap

    episodes = _build_output_episodes(bundles, merged_tasks=merged_tasks, task_index_maps=task_index_maps)
    return episodes, merged_tasks


def _convert_output_episodes(
    episodes: list[OutputEpisode],
    *,
    info: dict[str, Any],
    tasks: dict[int, str],
    video_keys: tuple[str, ...],
    data_budget_mb: int,
    video_budget_mb: int,
    out_root: Path,
    source_feature_maps: dict[Path, dict[str, Any]],
    source_column_renames: dict[Path, dict[str, str]],
    source_vector_element_remaps: dict[Path, tuple[VectorElementRemap, ...]],
    conversion_num_workers: int | None,
    finalize_progress: FinalizeProgressCallback | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    finalize_total = len(episodes) * (len(video_keys) + 1) + 3
    finalize_completed = 0
    if finalize_progress is not None:
        finalize_progress(finalize_completed, finalize_total)

    def advance() -> None:
        nonlocal finalize_completed
        finalize_completed = _advance_progress(finalize_progress, completed=finalize_completed, total=finalize_total)

    data_metadata, total_frames = _write_sharded_data_files(
        episodes,
        out_root,
        data_file_size_in_mb=data_budget_mb,
        source_feature_maps=source_feature_maps,
        source_column_renames=source_column_renames,
        source_vector_element_remaps=source_vector_element_remaps,
        on_episode_processed=advance,
    )
    video_metadata = _write_sharded_video_files(
        episodes,
        out_root,
        video_keys=video_keys,
        video_file_size_in_mb=video_budget_mb,
        worker_threads=conversion_num_workers,
        on_episode_processed=advance,
    )
    v3_info = _build_v3_info(
        info,
        data_budget_mb=data_budget_mb,
        video_budget_mb=video_budget_mb,
        total_episodes=len(episodes),
        total_frames=total_frames,
        total_tasks=len(tasks),
    )
    write_info(v3_info, out_root)
    finalize_completed = _advance_progress(finalize_progress, completed=finalize_completed, total=finalize_total)
    _write_task_mapping(tasks, out_root)
    finalize_completed = _advance_progress(finalize_progress, completed=finalize_completed, total=finalize_total)
    episode_rows, episode_stats = _build_episode_rows(
        episodes,
        data_metadata=data_metadata,
        video_metadata=video_metadata,
    )
    _write_episode_rows(out_root, episode_rows=episode_rows, episode_stats=episode_stats)
    finalize_completed = _advance_progress(finalize_progress, completed=finalize_completed, total=finalize_total)
    return episode_rows, episode_stats, total_frames


def convert_dataset(
    input_root: Path,
    output_root: Path,
    *,
    link_mode: MergeLinkMode,
    overwrite: bool,
    conversion_num_workers: int | None = None,
    bundle: V21DatasetBundle | None = None,
    finalize_progress: FinalizeProgressCallback | None = None,
) -> dict[str, Any]:
    del link_mode
    bundle = bundle or load_v21_dataset_bundle(input_root)
    output_root = output_root.expanduser().resolve()

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root}")
        if output_root.is_file():
            output_root.unlink()
        else:
            shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)
    episodes, tasks = _prepare_output_episodes_for_convert(bundle)
    _, _, total_frames = _convert_output_episodes(
        episodes,
        info=bundle.info,
        tasks=tasks,
        video_keys=bundle.video_keys,
        data_budget_mb=bundle.max_data_file_mb,
        video_budget_mb=bundle.max_video_file_mb,
        out_root=output_root,
        source_feature_maps={bundle.root: dict(bundle.info.get("features", {}))},
        source_column_renames={bundle.root: dict(bundle.column_renames)},
        source_vector_element_remaps={bundle.root: tuple(bundle.vector_element_remaps)},
        conversion_num_workers=conversion_num_workers,
        finalize_progress=finalize_progress,
    )
    return {
        "input_root": str(bundle.root),
        "output_root": str(output_root),
        "episodes": len(episodes),
        "frames": total_frames,
        "video_keys": list(bundle.video_keys),
        "link_mode": "shard",
        "conversion_num_workers": max(1, conversion_num_workers or 1),
    }


def merge_datasets(
    input_roots: list[Path] | None,
    output_root: Path,
    *,
    link_mode: MergeLinkMode,
    overwrite: bool,
    conversion_num_workers: int | None = None,
    bundles: list[V21DatasetBundle] | None = None,
    finalize_progress: FinalizeProgressCallback | None = None,
) -> dict[str, Any]:
    del link_mode
    if bundles is None:
        if not input_roots:
            raise ValueError("input_roots must be provided when bundles is None.")
        bundles = [load_v21_dataset_bundle(path) for path in input_roots]
    elif not bundles:
        raise ValueError("Expected at least one dataset bundle to merge.")

    bundles = sorted(bundles, key=lambda bundle: bundle.root.as_posix())
    _validate_mergeable_bundles(bundles)

    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root}")
        if output_root.is_file():
            output_root.unlink()
        else:
            shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    episodes, merged_tasks = _prepare_output_episodes_for_merge(bundles)
    data_budget_mb = max(bundle.max_data_file_mb for bundle in bundles)
    video_budget_mb = max(bundle.max_video_file_mb for bundle in bundles)
    _, _, total_frames = _convert_output_episodes(
        episodes,
        info=bundles[0].info,
        tasks=merged_tasks,
        video_keys=bundles[0].video_keys,
        data_budget_mb=data_budget_mb,
        video_budget_mb=video_budget_mb,
        out_root=output_root,
        source_feature_maps={bundle.root: dict(bundle.info.get("features", {})) for bundle in bundles},
        source_column_renames={bundle.root: dict(bundle.column_renames) for bundle in bundles},
        source_vector_element_remaps={bundle.root: tuple(bundle.vector_element_remaps) for bundle in bundles},
        conversion_num_workers=conversion_num_workers,
        finalize_progress=finalize_progress,
    )

    return {
        "input_roots": [str(bundle.root) for bundle in bundles],
        "output_root": str(output_root),
        "child_datasets": len(bundles),
        "episodes": len(episodes),
        "frames": total_frames,
        "tasks": len(merged_tasks),
        "video_keys": list(bundles[0].video_keys),
        "link_mode": "shard",
        "conversion_num_workers": max(1, conversion_num_workers or 1),
    }
