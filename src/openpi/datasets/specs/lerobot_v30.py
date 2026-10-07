# Copyright 2026 PLaW-VLA authors. SPDX-License-Identifier: Apache-2.0
# LeRobot-derived writer portions: Copyright Hugging Face contributors (Apache-2.0).
# Modified for PLaW-VLA; see NOTICE.
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from dataclasses import dataclass
from pathlib import Path
import json
import shutil
import warnings
from typing import Any
from typing import Callable

import datasets
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.feature_utils import get_hf_features_from_features
from lerobot.datasets.io_utils import get_file_size_in_mb
from lerobot.datasets.io_utils import write_info
from lerobot.datasets.io_utils import write_stats
from lerobot.datasets.io_utils import write_tasks
from lerobot.datasets.utils import DEFAULT_CHUNK_SIZE
from lerobot.datasets.utils import DEFAULT_DATA_FILE_SIZE_IN_MB
from lerobot.datasets.utils import DEFAULT_DATA_PATH
from lerobot.datasets.utils import DEFAULT_EPISODES_PATH
from lerobot.datasets.utils import DEFAULT_TASKS_PATH
from lerobot.datasets.utils import DEFAULT_VIDEO_FILE_SIZE_IN_MB
from lerobot.datasets.utils import DEFAULT_VIDEO_PATH
from lerobot.datasets.utils import flatten_dict
from lerobot.datasets.utils import update_chunk_file_indices
from lerobot.datasets.video_utils import get_video_duration_in_s
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from openpi.datasets.common.lerobot_v3 import MergeLinkMode
from openpi.datasets.common.lerobot_v3 import link_or_copy_path
from openpi.datasets.common.video_concat import concatenate_video_files

V30 = "v3.0"


@dataclass(frozen=True)
class SourceVideoReference:
    path: Path
    from_timestamp_s: float
    to_timestamp_s: float


@dataclass(frozen=True)
class V30EpisodeBundle:
    source_root: Path
    source_episode_index: int
    source_data_path: Path
    source_dataset_from_index: int
    source_dataset_to_index: int
    source_videos: dict[str, SourceVideoReference]

    @property
    def num_frames(self) -> int:
        return self.source_dataset_to_index - self.source_dataset_from_index


@dataclass(frozen=True)
class V30DatasetBundle:
    root: Path
    info: dict[str, Any]
    task_lookup: dict[int, str]
    episodes: tuple[V30EpisodeBundle, ...]
    output_features: dict[str, dict[str, Any]]
    video_keys: tuple[str, ...]
    data_budget_mb: int
    video_budget_mb: int
    metadata: Any = None


@dataclass(frozen=True)
class OutputEpisode:
    output_episode_index: int
    source: V30EpisodeBundle
    source_task_lookup: dict[int, str]
    task_index_remap: dict[int, int]
    metadata: Any = None


@dataclass(frozen=True)
class PreparedOutputEpisode:
    output_episode_index: int
    episode_length: int
    episode_tasks: list[str]
    episode_stats: dict[str, dict[str, np.ndarray]]
    source_videos: dict[str, SourceVideoReference]


@dataclass(frozen=True)
class _SourceVideoShardGroup:
    source_path: Path
    source_size_mb: float
    source_duration_s: float
    episode_refs: tuple[tuple[int, float, float], ...]


@dataclass(frozen=True)
class _DataShardRange:
    path: Path
    start_index: int
    end_index: int


PrepareEpisodeFn = Callable[[pa.Table, OutputEpisode, datasets.Features, int], tuple[pa.Table, PreparedOutputEpisode]]
BuildOutputInfoFn = Callable[[dict[str, Any], dict[str, dict[str, Any]], int, int, int, int], dict[str, Any]]
SelectSourceColumnsFn = Callable[[OutputEpisode], list[str] | None]
ValidateBundlesFn = Callable[[list[V30DatasetBundle]], None]
FinalizeProgressCallback = Callable[[int, int], None]


def _normalize_feature_shapes(features: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    for key, value in features.items():
        feature = dict(value)
        shape = feature.get("shape")
        if shape is not None:
            feature["shape"] = tuple(shape)
        normalized[key] = feature
    return normalized


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


def _video_feature_info(feature: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    if isinstance(feature.get("video_info"), dict):
        return "video_info", dict(feature["video_info"])
    if isinstance(feature.get("info"), dict):
        return "info", dict(feature["info"])
    return None, {}


def _allowed_varying_video_shape_indices(names: Any, rank: int) -> set[int]:
    if isinstance(names, (list, tuple)):
        allowed = {
            index for index, name in enumerate(names) if str(name).strip().lower() in {"height", "width"}
        }
        if allowed:
            return allowed
    if rank == 3:
        return {0, 1}
    return set()


def _merge_video_feature_specs(
    feature_key: str,
    feature_specs: list[tuple[Path, dict[str, Any]]],
) -> dict[str, Any]:
    reference_root, reference = feature_specs[0]
    reference_without_shape = {
        key: value for key, value in reference.items() if key not in {"shape", "info", "video_info"}
    }
    for bundle_root, spec in feature_specs[1:]:
        candidate_without_shape = {
            key: value for key, value in spec.items() if key not in {"shape", "info", "video_info"}
        }
        if candidate_without_shape != reference_without_shape:
            raise ValueError(
                f"Video feature schema mismatch for {feature_key} between {reference_root} and {bundle_root}."
            )

    shapes = [tuple(spec.get("shape") or ()) for _, spec in feature_specs]
    if not shapes:
        raise ValueError(f"Expected at least one video feature spec for {feature_key}.")

    rank = len(shapes[0])
    if any(len(shape) != rank for shape in shapes[1:]):
        raise ValueError(
            f"Video feature rank mismatch for {feature_key} between {reference_root} and {feature_specs[1][0]}."
        )

    allowed_varying_indices = _allowed_varying_video_shape_indices(reference.get("names"), rank)
    merged_shape: list[int | None] = []
    for index in range(rank):
        dim_values = {shape[index] for shape in shapes}
        if len(dim_values) == 1:
            merged_shape.append(shapes[0][index])
            continue
        if index not in allowed_varying_indices:
            raise ValueError(
                f"Video feature shape mismatch for {feature_key} between {reference_root} and {feature_specs[1][0]}."
            )
        merged_shape.append(None)

    merged_feature = dict(reference_without_shape)
    merged_feature["shape"] = tuple(merged_shape)

    info_key = next(
        (
            key
            for key in ("video_info", "info")
            if any(isinstance(spec.get(key), dict) for _, spec in feature_specs)
        ),
        None,
    )
    info_by_bundle = [_video_feature_info(spec)[1] for _, spec in feature_specs]
    allowed_varying_info_keys = {"video.height", "video.width"}
    if info_by_bundle:
        candidate_keys: set[str] = set().union(*(info.keys() for info in info_by_bundle))
        for key in sorted(candidate_keys):
            values = {info[key] for info in info_by_bundle if key in info}
            if len(values) > 1 and key not in allowed_varying_info_keys:
                raise ValueError(
                    f"Video feature info mismatch for {feature_key} key {key!r} between {reference_root} "
                    f"and {feature_specs[1][0]}."
                )

        common_info: dict[str, Any] = {}
        common_keys = set(info_by_bundle[0])
        for info in info_by_bundle[1:]:
            common_keys &= set(info)
        for key in sorted(common_keys):
            value = info_by_bundle[0][key]
            if all(info[key] == value for info in info_by_bundle[1:]):
                common_info[key] = value
        if info_key is not None and common_info:
            merged_feature[info_key] = common_info

    return merged_feature


def _resolve_merged_output_features(bundles: list[V30DatasetBundle]) -> dict[str, dict[str, Any]]:
    if not bundles:
        raise ValueError("Expected at least one task-level v3 dataset bundle.")

    reference = bundles[0]
    reference_feature_keys = tuple(reference.output_features)
    reference_feature_key_set = set(reference_feature_keys)
    merged_features: dict[str, dict[str, Any]] = {}

    for bundle in bundles[1:]:
        if set(bundle.output_features) != reference_feature_key_set:
            raise ValueError(f"Output feature schema mismatch between {reference.root} and {bundle.root}.")

    for feature_key in reference_feature_keys:
        feature_specs = [(bundle.root, bundle.output_features[feature_key]) for bundle in bundles]
        reference_feature = feature_specs[0][1]
        if reference_feature.get("dtype") == "video":
            merged_features[feature_key] = _merge_video_feature_specs(feature_key, feature_specs)
            continue

        for bundle_root, spec in feature_specs[1:]:
            if spec != reference_feature:
                raise ValueError(f"Output feature schema mismatch between {reference.root} and {bundle_root}.")
        merged_features[feature_key] = dict(reference_feature)

    return _normalize_feature_shapes(merged_features)


def load_v30_info(root: Path) -> dict[str, Any]:
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    version = str(info.get("codebase_version") or "")
    if not version.startswith("v3"):
        raise ValueError(f"Expected a LeRobot v3 dataset at {root}, found {version!r}.")
    info["features"] = _normalize_feature_shapes(dict(info.get("features") or {}))
    return info


def load_task_lookup(root: Path) -> dict[int, str]:
    tasks = pd.read_parquet(root / DEFAULT_TASKS_PATH)
    tasks.index.name = "task"
    return {int(row["task_index"]): str(task) for task, row in tasks.iterrows()}


def is_task_level_v30_dataset_dir(path: Path) -> bool:
    path = path.expanduser().resolve()
    return (
        (path / "meta" / "info.json").is_file()
        and (path / DEFAULT_TASKS_PATH).is_file()
        and any((path / "data").glob("chunk-*/file-*.parquet"))
        and any((path / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    )


def _iter_episode_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for parquet_path in sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        rows.extend(pq.read_table(parquet_path).to_pylist())
    rows.sort(key=lambda row: int(row["episode_index"]))
    return rows


def _source_data_path(root: Path, info: dict[str, Any], row: dict[str, Any]) -> Path:
    return root / info["data_path"].format(
        chunk_index=int(row["data/chunk_index"]),
        file_index=int(row["data/file_index"]),
    )


def _build_data_shard_ranges(root: Path) -> tuple[_DataShardRange, ...]:
    ranges: list[_DataShardRange] = []
    start_index = 0
    for parquet_path in sorted((root / "data").glob("chunk-*/file-*.parquet")):
        num_rows = int(pq.read_metadata(parquet_path).num_rows)
        end_index = start_index + num_rows
        ranges.append(_DataShardRange(path=parquet_path, start_index=start_index, end_index=end_index))
        start_index = end_index
    return tuple(ranges)


def _resolve_source_data_path(
    root: Path,
    info: dict[str, Any],
    row: dict[str, Any],
    *,
    data_shard_ranges: tuple[_DataShardRange, ...] | None = None,
) -> Path:
    hinted_path = _source_data_path(root, info, row)
    if data_shard_ranges is None:
        return hinted_path

    dataset_from_index = int(row["dataset_from_index"])
    dataset_to_index = int(row["dataset_to_index"])
    if dataset_from_index == dataset_to_index:
        return hinted_path

    for shard in data_shard_ranges:
        if shard.path == hinted_path and dataset_from_index >= shard.start_index and dataset_to_index <= shard.end_index:
            return hinted_path

    for shard in data_shard_ranges:
        if dataset_from_index < shard.end_index:
            if dataset_to_index > shard.end_index:
                raise ValueError(
                    f"Episode {row['episode_index']} in {root} spans multiple data shards: "
                    f"[{dataset_from_index}, {dataset_to_index}) crosses {shard.path}."
                )
            return shard.path

    raise ValueError(
        f"Could not resolve data shard for episode {row['episode_index']} in {root}: "
        f"dataset range [{dataset_from_index}, {dataset_to_index}) and hinted path {hinted_path}."
    )


def _source_video_path(root: Path, info: dict[str, Any], raw_video_key: str, row: dict[str, Any]) -> Path:
    return root / info["video_path"].format(
        video_key=raw_video_key,
        chunk_index=int(row[f"videos/{raw_video_key}/chunk_index"]),
        file_index=int(row[f"videos/{raw_video_key}/file_index"]),
    )


def _video_keys(info: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        key
        for key, feature in sorted(info.get("features", {}).items())
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    )


def load_v30_dataset_bundle(
    root: Path,
    *,
    video_key_mapping: dict[str, str] | None = None,
    output_features: dict[str, dict[str, Any]] | None = None,
    metadata: Any = None,
) -> V30DatasetBundle:
    root = root.expanduser().resolve()
    if not is_task_level_v30_dataset_dir(root):
        raise FileNotFoundError(f"Not a task-level LeRobot v3 dataset root: {root}")

    info = load_v30_info(root)
    task_lookup = load_task_lookup(root)
    data_shard_ranges = _build_data_shard_ranges(root)
    raw_video_keys = _video_keys(info)
    mapping = video_key_mapping or {key: key for key in raw_video_keys}
    unknown_raw_keys = sorted(set(mapping) - set(raw_video_keys))
    if unknown_raw_keys:
        raise ValueError(f"Unknown raw video keys for {root}: {unknown_raw_keys}")

    episodes: list[V30EpisodeBundle] = []
    for row in _iter_episode_rows(root):
        source_videos = {
            output_video_key: SourceVideoReference(
                path=_source_video_path(root, info, raw_video_key, row),
                from_timestamp_s=float(row[f"videos/{raw_video_key}/from_timestamp"]),
                to_timestamp_s=float(row[f"videos/{raw_video_key}/to_timestamp"]),
            )
            for raw_video_key, output_video_key in mapping.items()
        }
        episodes.append(
            V30EpisodeBundle(
                source_root=root,
                source_episode_index=int(row["episode_index"]),
                source_data_path=_resolve_source_data_path(root, info, row, data_shard_ranges=data_shard_ranges),
                source_dataset_from_index=int(row["dataset_from_index"]),
                source_dataset_to_index=int(row["dataset_to_index"]),
                source_videos=source_videos,
            )
        )

    resolved_output_features = _normalize_feature_shapes(output_features or dict(info["features"]))
    resolved_video_keys = tuple(dict.fromkeys(mapping.get(raw_key, raw_key) for raw_key in raw_video_keys if raw_key in mapping))
    return V30DatasetBundle(
        root=root,
        info=info,
        task_lookup=task_lookup,
        episodes=tuple(episodes),
        output_features=resolved_output_features,
        video_keys=resolved_video_keys,
        data_budget_mb=int(info.get("data_files_size_in_mb") or DEFAULT_DATA_FILE_SIZE_IN_MB),
        video_budget_mb=int(info.get("video_files_size_in_mb") or DEFAULT_VIDEO_FILE_SIZE_IN_MB),
        metadata=metadata,
    )


def _validate_mergeable_bundles(bundles: list[V30DatasetBundle]) -> dict[str, dict[str, Any]]:
    if not bundles:
        raise ValueError("Expected at least one task-level v3 dataset bundle.")

    reference = bundles[0]
    for bundle in bundles[1:]:
        if int(bundle.info.get("fps") or 0) != int(reference.info.get("fps") or 0):
            raise ValueError(f"FPS mismatch between {reference.root} and {bundle.root}.")
        if bundle.video_keys != reference.video_keys:
            raise ValueError(f"Video key mismatch between {reference.root} and {bundle.root}.")

    return _resolve_merged_output_features(bundles)


def _prepare_output_episodes_for_merge(
    bundles: list[V30DatasetBundle],
) -> tuple[list[OutputEpisode], dict[int, str]]:
    task_to_index: dict[str, int] = {}
    merged_tasks: dict[int, str] = {}
    episodes: list[OutputEpisode] = []

    for bundle in bundles:
        remap: dict[int, int] = {}
        for source_task_index, task in sorted(bundle.task_lookup.items()):
            if task not in task_to_index:
                output_task_index = len(task_to_index)
                task_to_index[task] = output_task_index
                merged_tasks[output_task_index] = task
            remap[source_task_index] = task_to_index[task]

        for source_episode in bundle.episodes:
            if source_episode.num_frames <= 0:
                warnings.warn(
                    f"Skipping empty source episode {source_episode.source_episode_index} from {bundle.root}: "
                    f"dataset range [{source_episode.source_dataset_from_index}, "
                    f"{source_episode.source_dataset_to_index})",
                    stacklevel=2,
                )
                continue
            episodes.append(
                OutputEpisode(
                    output_episode_index=len(episodes),
                    source=source_episode,
                    source_task_lookup=bundle.task_lookup,
                    task_index_remap=remap,
                    metadata=bundle.metadata,
                )
            )

    return episodes, merged_tasks


def _serialize_metadata_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, list):
        return [_serialize_metadata_value(item) for item in value]
    if isinstance(value, tuple):
        return [_serialize_metadata_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize_metadata_value(item) for key, item in value.items()}
    return value


def _build_default_output_info(
    info: dict[str, Any],
    output_features: dict[str, dict[str, Any]],
    data_budget_mb: int,
    video_budget_mb: int,
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
) -> dict[str, Any]:
    updated = dict(info)
    updated["codebase_version"] = V30
    updated["features"] = output_features
    updated["data_path"] = DEFAULT_DATA_PATH
    updated["video_path"] = DEFAULT_VIDEO_PATH if any(ft["dtype"] == "video" for ft in output_features.values()) else None
    updated["fps"] = int(info["fps"])
    updated["total_episodes"] = total_episodes
    updated["total_frames"] = total_frames
    updated["total_tasks"] = total_tasks
    updated["splits"] = {"train": f"0:{total_episodes}"}
    updated["chunks_size"] = int(info.get("chunks_size") or DEFAULT_CHUNK_SIZE)
    updated["data_files_size_in_mb"] = data_budget_mb
    updated["video_files_size_in_mb"] = video_budget_mb
    updated.pop("total_chunks", None)
    updated.pop("total_videos", None)
    return updated


def _flush_pending_tables(pending_tables: list[pa.Table], destination: Path) -> None:
    if not pending_tables:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    table = pending_tables[0] if len(pending_tables) == 1 else pa.concat_tables(pending_tables)
    pq.write_table(table, destination)
    pending_tables.clear()


def _build_source_data_groups(episodes: list[OutputEpisode]) -> list[tuple[Path, list[OutputEpisode]]]:
    groups: list[tuple[Path, list[OutputEpisode]]] = []
    current_path: Path | None = None
    current_group: list[OutputEpisode] = []
    for episode in episodes:
        source_path = episode.source.source_data_path
        if current_path is None or source_path != current_path:
            if current_group:
                groups.append((current_path, current_group))
            current_path = source_path
            current_group = [episode]
        else:
            current_group.append(episode)
    if current_group and current_path is not None:
        groups.append((current_path, current_group))
    return groups


def _merge_output_stats(
    current: dict[str, dict[str, np.ndarray]] | None,
    batch: list[dict[str, dict[str, np.ndarray]]],
) -> dict[str, dict[str, np.ndarray]] | None:
    if not batch:
        return current
    if current is None:
        return aggregate_stats(batch)
    return aggregate_stats([current, *batch])


def _write_sharded_data_files(
    episodes: list[OutputEpisode],
    out_root: Path,
    *,
    output_hf_features: datasets.Features,
    data_file_size_in_mb: int,
    prepare_episode: PrepareEpisodeFn,
    select_source_columns: SelectSourceColumnsFn | None,
    on_episode_processed: Callable[[], None] | None = None,
) -> tuple[list[PreparedOutputEpisode], dict[int, dict[str, Any]], int, dict[str, dict[str, np.ndarray]] | None]:
    data_chunk_idx = 0
    data_file_idx = 0
    pending_size_mb = 0.0
    dataset_from_index = 0
    pending_tables: list[pa.Table] = []
    prepared_episodes: list[PreparedOutputEpisode] = []
    data_metadata_by_episode: dict[int, dict[str, Any]] = {}
    output_stats: dict[str, dict[str, np.ndarray]] | None = None
    pending_episode_stats: list[dict[str, dict[str, np.ndarray]]] = []

    def flush_pending() -> None:
        nonlocal data_chunk_idx, data_file_idx, pending_size_mb
        if not pending_tables:
            return
        destination = out_root / DEFAULT_DATA_PATH.format(chunk_index=data_chunk_idx, file_index=data_file_idx)
        _flush_pending_tables(pending_tables, destination)
        data_chunk_idx, data_file_idx = update_chunk_file_indices(data_chunk_idx, data_file_idx, DEFAULT_CHUNK_SIZE)
        pending_size_mb = 0.0

    for source_data_path, grouped_episodes in _build_source_data_groups(episodes):
        grouped_episodes = sorted(grouped_episodes, key=lambda item: item.source.source_dataset_from_index)
        first_dataset_from = grouped_episodes[0].source.source_dataset_from_index

        if select_source_columns is None:
            source_table = pq.read_table(source_data_path)
        else:
            schema_names = set(pq.read_schema(source_data_path).names)
            requested_columns = select_source_columns(grouped_episodes[0]) or []
            source_columns = [column_name for column_name in requested_columns if column_name in schema_names]
            source_table = pq.read_table(source_data_path, columns=source_columns or None)

        for episode in grouped_episodes:
            local_start = episode.source.source_dataset_from_index - first_dataset_from
            local_length = episode.source.num_frames
            episode_source_table = source_table.slice(local_start, local_length)
            if episode_source_table.num_rows != local_length:
                raise ValueError(
                    "Source episode slice length mismatch for "
                    f"{episode.source.source_root} episode {episode.source.source_episode_index}: "
                    f"expected {local_length} rows from [{episode.source.source_dataset_from_index}, "
                    f"{episode.source.source_dataset_to_index}), got {episode_source_table.num_rows} rows "
                    f"from {source_data_path}."
                )
            output_table, prepared = prepare_episode(
                episode_source_table,
                episode,
                output_hf_features,
                dataset_from_index,
            )
            output_table_size_mb = max(float(getattr(output_table, "nbytes", 0)) / (1024**2), 1e-6)

            if pending_tables and pending_size_mb + output_table_size_mb >= data_file_size_in_mb:
                flush_pending()

            current_chunk_idx = data_chunk_idx
            current_file_idx = data_file_idx
            pending_tables.append(output_table)
            pending_size_mb += output_table_size_mb

            data_metadata_by_episode[prepared.output_episode_index] = {
                "data/chunk_index": current_chunk_idx,
                "data/file_index": current_file_idx,
                "dataset_from_index": dataset_from_index,
                "dataset_to_index": dataset_from_index + prepared.episode_length,
            }
            prepared_episodes.append(prepared)
            pending_episode_stats.append(prepared.episode_stats)
            if len(pending_episode_stats) >= 64:
                output_stats = _merge_output_stats(output_stats, pending_episode_stats)
                pending_episode_stats = []
            if on_episode_processed is not None:
                on_episode_processed()

            dataset_from_index += prepared.episode_length

    flush_pending()
    output_stats = _merge_output_stats(output_stats, pending_episode_stats)
    return prepared_episodes, data_metadata_by_episode, dataset_from_index, output_stats


def _build_video_shard_groups(
    prepared_episodes: list[PreparedOutputEpisode],
    *,
    video_key: str,
    duration_cache: dict[Path, float],
    size_cache: dict[Path, float],
) -> list[_SourceVideoShardGroup]:
    groups: list[_SourceVideoShardGroup] = []
    current_source_path: Path | None = None
    current_episode_refs: list[tuple[int, float, float]] = []

    def flush_group() -> None:
        nonlocal current_source_path, current_episode_refs
        if current_source_path is None:
            return
        groups.append(
            _SourceVideoShardGroup(
                source_path=current_source_path,
                source_size_mb=size_cache.setdefault(current_source_path, get_file_size_in_mb(current_source_path)),
                source_duration_s=duration_cache.setdefault(
                    current_source_path,
                    float(get_video_duration_in_s(current_source_path)),
                ),
                episode_refs=tuple(current_episode_refs),
            )
        )
        current_source_path = None
        current_episode_refs = []

    for prepared in prepared_episodes:
        ref = prepared.source_videos[video_key]
        if current_source_path is None or ref.path != current_source_path:
            flush_group()
            current_source_path = ref.path
        current_episode_refs.append((prepared.output_episode_index, ref.from_timestamp_s, ref.to_timestamp_s))

    flush_group()
    return groups


def _write_output_video_shard(output_path: Path, input_paths: list[Path], *, link_mode: MergeLinkMode) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(input_paths) == 1:
        link_or_copy_path(input_paths[0], output_path, link_mode=link_mode)
        return
    concatenate_video_files(input_paths, output_path)


def _write_sharded_video_key_files(
    prepared_episodes: list[PreparedOutputEpisode],
    out_root: Path,
    *,
    video_key: str,
    video_file_size_in_mb: int,
    link_mode: MergeLinkMode,
    duration_cache: dict[Path, float],
    size_cache: dict[Path, float],
) -> dict[int, dict[str, Any]]:
    per_episode: dict[int, dict[str, Any]] = {}
    groups = _build_video_shard_groups(
        prepared_episodes,
        video_key=video_key,
        duration_cache=duration_cache,
        size_cache=size_cache,
    )

    video_chunk_idx = 0
    video_file_idx = 0
    pending_size_mb = 0.0
    pending_duration_s = 0.0
    pending_paths: list[Path] = []

    def flush_pending() -> None:
        nonlocal video_chunk_idx, video_file_idx, pending_size_mb, pending_duration_s, pending_paths
        if not pending_paths:
            return
        destination = out_root / DEFAULT_VIDEO_PATH.format(
            video_key=video_key,
            chunk_index=video_chunk_idx,
            file_index=video_file_idx,
        )
        _write_output_video_shard(destination, pending_paths, link_mode=link_mode)
        video_chunk_idx, video_file_idx = update_chunk_file_indices(video_chunk_idx, video_file_idx, DEFAULT_CHUNK_SIZE)
        pending_size_mb = 0.0
        pending_duration_s = 0.0
        pending_paths = []

    for group in groups:
        if pending_paths and pending_size_mb + group.source_size_mb >= video_file_size_in_mb:
            flush_pending()

        current_chunk_idx = video_chunk_idx
        current_file_idx = video_file_idx
        for episode_index, source_from, source_to in group.episode_refs:
            per_episode.setdefault(episode_index, {}).update(
                {
                    f"videos/{video_key}/chunk_index": current_chunk_idx,
                    f"videos/{video_key}/file_index": current_file_idx,
                    f"videos/{video_key}/from_timestamp": pending_duration_s + source_from,
                    f"videos/{video_key}/to_timestamp": pending_duration_s + source_to,
                }
            )

        pending_paths.append(group.source_path)
        pending_size_mb += group.source_size_mb
        pending_duration_s += group.source_duration_s

    flush_pending()
    return per_episode


def _write_sharded_video_files(
    prepared_episodes: list[PreparedOutputEpisode],
    out_root: Path,
    *,
    video_keys: tuple[str, ...],
    video_file_size_in_mb: int,
    link_mode: MergeLinkMode,
    worker_threads: int | None = None,
    on_episode_processed: Callable[[], None] | None = None,
) -> dict[int, dict[str, Any]]:
    if not video_keys:
        return {}

    duration_cache: dict[Path, float] = {}
    size_cache: dict[Path, float] = {}
    if len(video_keys) == 1:
        per_episode = _write_sharded_video_key_files(
            prepared_episodes,
            out_root,
            video_key=video_keys[0],
            video_file_size_in_mb=video_file_size_in_mb,
            link_mode=link_mode,
            duration_cache=duration_cache,
            size_cache=size_cache,
        )
        if on_episode_processed is not None:
            for _ in prepared_episodes:
                on_episode_processed()
        return per_episode

    per_episode: dict[int, dict[str, Any]] = {}
    max_workers = max(1, min(worker_threads or len(video_keys), len(video_keys), 8))
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="openpi-v30-video") as executor:
        futures = {
            executor.submit(
                _write_sharded_video_key_files,
                prepared_episodes,
                out_root,
                video_key=video_key,
                video_file_size_in_mb=video_file_size_in_mb,
                link_mode=link_mode,
                duration_cache=duration_cache,
                size_cache=size_cache,
            ): video_key
            for video_key in video_keys
        }
        for future in as_completed(futures):
            key_metadata = future.result()
            for episode_index, metadata in key_metadata.items():
                per_episode.setdefault(episode_index, {}).update(metadata)
            if on_episode_processed is not None:
                for _ in prepared_episodes:
                    on_episode_processed()
    return per_episode


def _write_episode_metadata_files(
    episode_rows: list[dict[str, Any]],
    out_root: Path,
    *,
    rows_per_file: int = 5000,
) -> None:
    meta_chunk_idx = 0
    meta_file_idx = 0
    for start in range(0, len(episode_rows), rows_per_file):
        batch_rows = []
        for row in episode_rows[start : start + rows_per_file]:
            serialized = {key: _serialize_metadata_value(value) for key, value in row.items()}
            serialized["meta/episodes/chunk_index"] = meta_chunk_idx
            serialized["meta/episodes/file_index"] = meta_file_idx
            batch_rows.append(serialized)

        destination = out_root / DEFAULT_EPISODES_PATH.format(chunk_index=meta_chunk_idx, file_index=meta_file_idx)
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(batch_rows), destination)
        meta_chunk_idx, meta_file_idx = update_chunk_file_indices(meta_chunk_idx, meta_file_idx, DEFAULT_CHUNK_SIZE)


def merge_datasets(
    bundles: list[V30DatasetBundle],
    output_root: Path,
    *,
    overwrite: bool,
    prepare_episode: PrepareEpisodeFn,
    link_mode: MergeLinkMode = "copy",
    video_worker_threads: int | None = None,
    build_output_info: BuildOutputInfoFn | None = None,
    select_source_columns: SelectSourceColumnsFn | None = None,
    validate_bundles: ValidateBundlesFn | None = None,
    finalize_progress: FinalizeProgressCallback | None = None,
) -> dict[str, Any]:
    if not bundles:
        raise ValueError("Expected at least one task-level v3 dataset bundle to merge.")

    bundles = sorted(bundles, key=lambda bundle: bundle.root.as_posix())
    output_features = _validate_mergeable_bundles(bundles)
    if validate_bundles is not None:
        validate_bundles(bundles)

    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        if not overwrite:
            return {
                "output_root": str(output_root),
                "status": "skipped",
                "child_datasets": len(bundles),
                "episodes": 0,
                "frames": 0,
                "tasks": 0,
                "video_keys": list(bundles[0].video_keys),
            }
        if output_root.is_file():
            output_root.unlink()
        else:
            shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    output_episodes, merged_tasks = _prepare_output_episodes_for_merge(bundles)
    finalize_total = len(output_episodes) * (len(bundles[0].video_keys) + 1) + 4
    finalize_completed = 0
    if finalize_progress is not None:
        finalize_progress(finalize_completed, finalize_total)

    def advance() -> None:
        nonlocal finalize_completed
        finalize_completed = _advance_progress(finalize_progress, completed=finalize_completed, total=finalize_total)

    output_hf_features = get_hf_features_from_features(output_features)
    data_budget_mb = max(bundle.data_budget_mb for bundle in bundles)
    video_budget_mb = max(bundle.video_budget_mb for bundle in bundles)

    prepared_episodes, data_metadata_by_episode, total_frames, output_stats = _write_sharded_data_files(
        output_episodes,
        output_root,
        output_hf_features=output_hf_features,
        data_file_size_in_mb=data_budget_mb,
        prepare_episode=prepare_episode,
        select_source_columns=select_source_columns,
        on_episode_processed=advance,
    )
    video_metadata_by_episode = _write_sharded_video_files(
        prepared_episodes,
        output_root,
        video_keys=bundles[0].video_keys,
        video_file_size_in_mb=video_budget_mb,
        link_mode=link_mode,
        worker_threads=video_worker_threads,
        on_episode_processed=advance,
    )

    info_builder = build_output_info or _build_default_output_info
    info = info_builder(
        bundles[0].info,
        output_features,
        data_budget_mb,
        video_budget_mb,
        len(prepared_episodes),
        total_frames,
        len(merged_tasks),
    )
    write_info(info, output_root)
    finalize_completed = _advance_progress(finalize_progress, completed=finalize_completed, total=finalize_total)

    ordered_task_strings = [merged_tasks[index] for index in sorted(merged_tasks)]
    tasks_df = pd.DataFrame(
        {"task_index": range(len(ordered_task_strings))},
        index=pd.Index(ordered_task_strings, name="task"),
    )
    write_tasks(tasks_df, output_root)
    finalize_completed = _advance_progress(finalize_progress, completed=finalize_completed, total=finalize_total)

    if output_stats is not None:
        write_stats(output_stats, output_root)
    finalize_completed = _advance_progress(finalize_progress, completed=finalize_completed, total=finalize_total)

    episode_rows = [
        {
            "episode_index": prepared.output_episode_index,
            "tasks": prepared.episode_tasks,
            "length": prepared.episode_length,
            **data_metadata_by_episode[prepared.output_episode_index],
            **video_metadata_by_episode.get(prepared.output_episode_index, {}),
            **flatten_dict({"stats": prepared.episode_stats}),
        }
        for prepared in prepared_episodes
    ]
    _write_episode_metadata_files(episode_rows, output_root)
    finalize_completed = _advance_progress(finalize_progress, completed=finalize_completed, total=finalize_total)

    return {
        "input_roots": [str(bundle.root) for bundle in bundles],
        "output_root": str(output_root),
        "status": "converted",
        "child_datasets": len(bundles),
        "episodes": len(prepared_episodes),
        "frames": total_frames,
        "tasks": len(merged_tasks),
        "video_keys": list(bundles[0].video_keys),
    }
