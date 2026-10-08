# Copyright 2026 PLaW-VLA authors. SPDX-License-Identifier: Apache-2.0
# LeRobot-derived writer portions: Copyright Hugging Face contributors (Apache-2.0).
# Modified for PLaW-VLA; see NOTICE.
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import jsonlines
from lerobot.datasets.compute_stats import compute_episode_stats
from lerobot.datasets.io_utils import get_file_size_in_mb
from lerobot.datasets.io_utils import load_info
from lerobot.datasets.io_utils import load_stats
from lerobot.datasets.io_utils import write_stats
from lerobot.datasets.utils import DEFAULT_DATA_FILE_SIZE_IN_MB
from lerobot.datasets.utils import DEFAULT_VIDEO_FILE_SIZE_IN_MB
from lerobot.datasets.utils import LEGACY_EPISODES_PATH
from lerobot.datasets.utils import LEGACY_TASKS_PATH
import numpy as np
import pyarrow.parquet as pq

from plawvla.datasets.common.lerobot_v3 import MergeLinkMode
from plawvla.datasets.specs import lerobot_v21

V20 = "v2.0"
INLINE_VISUAL_DTYPES = frozenset({"image", "video"})


def _load_jsonlines(path: Path) -> list[dict[str, Any]]:
    with jsonlines.open(path, "r") as reader:
        return list(reader)


def _validate_v20_root(root: Path) -> dict[str, Any]:
    info = load_info(root)
    version = info.get("codebase_version")
    if version != V20:
        raise ValueError(f"Expected a LeRobot {V20} dataset at {root}, found {version!r}.")
    return info


def _legacy_tasks(root: Path) -> dict[int, str]:
    rows = _load_jsonlines(root / LEGACY_TASKS_PATH)
    return {int(row["task_index"]): str(row["task"]) for row in rows}


def _legacy_episodes(root: Path) -> dict[int, dict[str, Any]]:
    rows = _load_jsonlines(root / LEGACY_EPISODES_PATH)
    return {int(row["episode_index"]): row for row in rows}


def _rename_top_level_keys(mapping: dict[str, Any], renames: dict[str, str]) -> dict[str, Any]:
    if not renames:
        return dict(mapping)
    return {renames.get(key, key): value for key, value in mapping.items()}


def _nonvisual_feature_specs(info: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        key: dict(feature)
        for key, feature in info.get("features", {}).items()
        if isinstance(feature, dict) and feature.get("dtype") not in INLINE_VISUAL_DTYPES | {"string"}
    }


def _iter_v20_episode_files(root: Path) -> list[tuple[int, Path]]:
    return [
        (int(parquet_path.stem.split("_")[-1]), parquet_path)
        for parquet_path in sorted((root / "data").glob("chunk-*/episode_*.parquet"))
    ]


def _column_to_dense_numpy(column: Any) -> np.ndarray:
    array = column.combine_chunks() if hasattr(column, "combine_chunks") else column
    if hasattr(array, "to_numpy"):
        try:
            values = np.asarray(array.to_numpy(zero_copy_only=False))
        except TypeError:
            values = np.asarray(array.to_numpy())
    else:
        values = np.asarray(array.to_pylist())
    if values.dtype == object:
        values = np.asarray(values.tolist())
    return values


def _normalize_stats_dtypes(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_stats_dtypes(child) for key, child in value.items()}
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.floating):
            return value.astype(np.float64, copy=False)
        if np.issubdtype(value.dtype, np.integer):
            return value.astype(np.int64, copy=False)
    return value


def _compute_nonvisual_episode_stats(
    source_path: Path,
    feature_specs: dict[str, dict[str, Any]],
    *,
    column_renames: dict[str, str],
    vector_element_remaps: tuple[lerobot_v21.VectorElementRemap, ...],
) -> dict[str, Any]:
    if not feature_specs:
        return {}

    table = pq.read_table(source_path)
    table = lerobot_v21.apply_vector_element_remaps_to_table(table, vector_element_remaps)
    if column_renames:
        renamed_columns = [column_renames.get(name, name) for name in table.schema.names]
        if renamed_columns != list(table.schema.names):
            table = table.rename_columns(renamed_columns)

    episode_data = {name: _column_to_dense_numpy(table.column(name)) for name in feature_specs if name in table.schema.names}
    if not episode_data:
        return {}
    stats = compute_episode_stats(episode_data, {name: feature_specs[name] for name in episode_data})
    return _normalize_stats_dtypes(stats)


def load_v20_dataset_bundle(
    root: Path,
    *,
    column_renames: dict[str, str] | None = None,
    vector_element_remaps: tuple[lerobot_v21.VectorElementRemap, ...] | None = None,
) -> lerobot_v21.V21DatasetBundle:
    root = root.expanduser().resolve()
    info = _validate_v20_root(root)
    column_renames = dict(column_renames or {})
    vector_element_remaps = tuple(vector_element_remaps or ())
    tasks = _legacy_tasks(root)
    legacy_episodes = _legacy_episodes(root)
    nonvisual_feature_specs = _rename_top_level_keys(_nonvisual_feature_specs(info), column_renames)

    normalized_info = dict(info)
    normalized_info["video_path"] = None
    normalized_info["features"] = _rename_top_level_keys(dict(info.get("features", {})), column_renames)

    episode_bundles: list[lerobot_v21.V21EpisodeBundle] = []
    max_data_file_mb = DEFAULT_DATA_FILE_SIZE_IN_MB
    for episode_index, source_data_path in _iter_v20_episode_files(root):
        if episode_index not in legacy_episodes:
            raise ValueError(f"Episode {episode_index} is missing from {root / LEGACY_EPISODES_PATH}.")

        num_frames = pq.read_metadata(source_data_path).num_rows
        data_size_mb = get_file_size_in_mb(source_data_path)
        max_data_file_mb = max(max_data_file_mb, math.ceil(data_size_mb))

        episode_bundles.append(
            lerobot_v21.V21EpisodeBundle(
                source_root=root,
                source_episode_index=episode_index,
                source_data_path=source_data_path,
                data_size_mb=data_size_mb,
                num_frames=num_frames,
                episode=dict(legacy_episodes[episode_index]),
                stats=_compute_nonvisual_episode_stats(
                    source_data_path,
                    nonvisual_feature_specs,
                    column_renames=column_renames,
                    vector_element_remaps=vector_element_remaps,
                ),
                video_paths={},
                video_sizes_mb={},
                video_durations_s={},
            )
        )

    return lerobot_v21.V21DatasetBundle(
        root=root,
        info=normalized_info,
        tasks=tasks,
        episodes=tuple(episode_bundles),
        video_keys=(),
        max_data_file_mb=max_data_file_mb,
        max_video_file_mb=DEFAULT_VIDEO_FILE_SIZE_IN_MB,
        column_renames=column_renames,
        vector_element_remaps=vector_element_remaps,
    )


def _preserve_source_visual_stats(source_root: Path, output_root: Path, *, column_renames: dict[str, str]) -> list[str]:
    source_stats = load_stats(source_root) or {}
    output_stats = load_stats(output_root) or {}
    if not source_stats and not output_stats:
        return []

    source_stats = _rename_top_level_keys(source_stats, column_renames)
    merged_stats = dict(source_stats)
    merged_stats.update(output_stats)
    write_stats(merged_stats, output_root)
    return sorted(source_stats)


def convert_dataset(
    input_root: Path,
    output_root: Path,
    *,
    link_mode: MergeLinkMode,
    overwrite: bool,
    column_renames: dict[str, str] | None = None,
    vector_element_remaps: tuple[lerobot_v21.VectorElementRemap, ...] | None = None,
    conversion_num_workers: int | None = None,
    bundle: lerobot_v21.V21DatasetBundle | None = None,
    finalize_progress: lerobot_v21.FinalizeProgressCallback | None = None,
) -> dict[str, Any]:
    bundle = bundle or load_v20_dataset_bundle(
        input_root,
        column_renames=column_renames,
        vector_element_remaps=vector_element_remaps,
    )
    summary = lerobot_v21.convert_dataset(
        input_root,
        output_root,
        link_mode=link_mode,
        overwrite=overwrite,
        conversion_num_workers=conversion_num_workers,
        bundle=bundle,
        finalize_progress=finalize_progress,
    )
    preserved_stats_keys = _preserve_source_visual_stats(
        bundle.root,
        output_root,
        column_renames=bundle.column_renames,
    )
    if preserved_stats_keys:
        summary["preserved_source_stats_keys"] = preserved_stats_keys
    summary["source_codebase_version"] = V20
    return summary
