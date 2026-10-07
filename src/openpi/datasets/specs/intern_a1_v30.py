# Copyright 2026 PLaW-VLA authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import datasets
from lerobot.datasets.compute_stats import compute_episode_stats
import numpy as np
import pyarrow as pa

from openpi.datasets.common.lerobot_v3 import MergeLinkMode
from openpi.datasets.specs import intern_a1 as common
from openpi.datasets.specs import lerobot_v30

V30 = lerobot_v30.V30
V30DatasetBundle = lerobot_v30.V30DatasetBundle


def _raw_to_canonical_video_keys(layout: common.InternA1Layout) -> dict[str, str]:
    mapping = {
        layout.head_camera_key: common.CANONICAL_CAMERA_KEYS[0],
        layout.left_camera_key: common.CANONICAL_CAMERA_KEYS[1],
    }
    if layout.right_camera_key is not None:
        mapping[layout.right_camera_key] = common.CANONICAL_CAMERA_KEYS[2]
    return mapping


def is_task_level_v30_dataset_dir(path: Path) -> bool:
    return lerobot_v30.is_task_level_v30_dataset_dir(path)


def discover_task_level_dataset_dirs(
    dataset_root: Path,
    *,
    embodiments: Iterable[str] | None = None,
    categories: Iterable[str] | None = None,
) -> dict[str, list[Path]]:
    normalized_embodiments = common.normalize_filters(embodiments, known_values=common.KNOWN_EMBODIMENTS)
    normalized_categories = common.normalize_filters(categories, known_values=common.KNOWN_CATEGORIES)

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
        if direct_info_paths and all(info_path.is_file() for info_path in direct_info_paths):
            candidate_info_paths = direct_info_paths
        else:
            candidate_info_paths = sorted(search_root.rglob("meta/info.json"))

        for info_path in candidate_info_paths:
            if not info_path.is_file():
                continue
            dataset_dir = info_path.parent.parent
            embodiment = common.infer_embodiment(dataset_dir)
            category = common.infer_category(dataset_dir)
            if embodiment is None or category is None:
                continue
            if embodiment not in normalized_embodiments or category not in normalized_categories:
                continue
            if not lerobot_v30.is_task_level_v30_dataset_dir(dataset_dir):
                continue
            grouped[embodiment].append(dataset_dir.resolve())

    return {embodiment: sorted(set(paths)) for embodiment, paths in grouped.items() if paths}


def load_v30_dataset_bundle(root: Path) -> V30DatasetBundle:
    info = lerobot_v30.load_v30_info(root.expanduser().resolve())
    layout = common.infer_layout_from_info(info, dataset_dir=root)
    output_features = common._build_output_features(info, layout)
    return lerobot_v30.load_v30_dataset_bundle(
        root,
        video_key_mapping=_raw_to_canonical_video_keys(layout),
        output_features=output_features,
        metadata={"layout": layout, "output_features": output_features, "fps": int(info["fps"])},
    )


def _select_source_columns(episode: lerobot_v30.OutputEpisode) -> list[str]:
    layout = episode.metadata["layout"]
    columns = [
        layout.left_state_key,
        layout.left_gripper_key,
        layout.left_action_key,
        layout.left_gripper_action_key,
        "task_index",
        "frame_index",
        "timestamp",
    ]
    if layout.right_state_key is not None and layout.right_gripper_key is not None:
        columns.extend(
            [
                layout.right_state_key,
                layout.right_gripper_key,
                layout.right_action_key,
                layout.right_gripper_action_key,
            ]
        )
    return columns


def _rewrite_table_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    column_index = table.schema.get_field_index(name)
    if column_index < 0:
        return table
    field = table.schema.field(column_index)
    return table.set_column(column_index, field, pa.array(values, type=field.type))


def _prepare_canonical_episode(
    source_table: pa.Table,
    episode: lerobot_v30.OutputEpisode,
    output_hf_features: datasets.Features,
    dataset_from_index: int,
) -> tuple[pa.Table, lerobot_v30.PreparedOutputEpisode]:
    layout = episode.metadata["layout"]
    output_features = episode.metadata["output_features"]
    state, state_mask, actions, actions_mask = common._convert_episode_arrays(source_table, layout)
    num_frames = state.shape[0]

    if "task_index" in source_table.schema.names:
        source_task_indices = common._column_to_vector(source_table.column("task_index")).astype(np.int64, copy=False)
    else:
        source_task_indices = np.zeros(num_frames, dtype=np.int64)
    frame_tasks = [episode.source_task_lookup[int(task_index)] for task_index in source_task_indices]
    output_task_indices = np.asarray(
        [episode.task_index_remap[int(task_index)] for task_index in source_task_indices],
        dtype=np.int64,
    )
    episode_tasks = list(dict.fromkeys(frame_tasks))

    if "frame_index" in source_table.schema.names:
        frame_indices = common._column_to_vector(source_table.column("frame_index")).astype(np.int64, copy=False)
    else:
        frame_indices = np.arange(num_frames, dtype=np.int64)

    if "timestamp" in source_table.schema.names:
        timestamps = common._column_to_vector(source_table.column("timestamp")).astype(np.float32, copy=False)
    else:
        timestamps = np.arange(num_frames, dtype=np.float32) / float(episode.metadata["fps"])

    episode_columns = {
        "task_index": output_task_indices,
        "episode_index": np.full(num_frames, episode.output_episode_index, dtype=np.int64),
        "frame_index": frame_indices.astype(np.int64, copy=False),
        "timestamp": timestamps.astype(np.float32, copy=False),
        "index": np.arange(dataset_from_index, dataset_from_index + num_frames, dtype=np.int64),
        "observation.state": state,
        "observation.state_mask": state_mask,
        "actions": actions,
        "actions_mask": actions_mask,
        "observation.image_mask": common._image_mask(num_frames, layout),
    }

    non_visual_features = {
        key: feature
        for key, feature in output_features.items()
        if feature["dtype"] not in {"video", "bool"}
    }
    non_visual_columns = {key: episode_columns[key] for key in non_visual_features}
    episode_stats = compute_episode_stats(non_visual_columns, non_visual_features) if non_visual_columns else {}

    output_dataset = datasets.Dataset.from_dict(episode_columns, features=output_hf_features, split="train")
    output_table = output_dataset.with_format("arrow")[:]
    output_table = _rewrite_table_column(output_table, "episode_index", episode_columns["episode_index"])
    output_table = _rewrite_table_column(output_table, "index", episode_columns["index"])
    output_table = _rewrite_table_column(output_table, "task_index", episode_columns["task_index"])

    prepared = lerobot_v30.PreparedOutputEpisode(
        output_episode_index=episode.output_episode_index,
        episode_length=num_frames,
        episode_tasks=episode_tasks,
        episode_stats=episode_stats,
        source_videos=episode.source.source_videos,
    )
    return output_table, prepared


def _build_output_info(
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
    updated["robot_type"] = "intern_a1"
    updated["features"] = output_features
    updated["data_path"] = lerobot_v30.DEFAULT_DATA_PATH
    updated["video_path"] = (
        lerobot_v30.DEFAULT_VIDEO_PATH if any(ft["dtype"] == "video" for ft in output_features.values()) else None
    )
    updated["fps"] = int(info["fps"])
    updated["total_episodes"] = total_episodes
    updated["total_frames"] = total_frames
    updated["total_tasks"] = total_tasks
    updated["splits"] = {"train": f"0:{total_episodes}"}
    updated["chunks_size"] = int(info.get("chunks_size") or lerobot_v30.DEFAULT_CHUNK_SIZE)
    updated["data_files_size_in_mb"] = data_budget_mb
    updated["video_files_size_in_mb"] = video_budget_mb
    updated.pop("total_chunks", None)
    updated.pop("total_videos", None)
    return updated


def _validate_bundles(bundles: list[V30DatasetBundle]) -> None:
    reference_layout = bundles[0].metadata["layout"]
    for bundle in bundles[1:]:
        layout = bundle.metadata["layout"]
        if layout.embodiment != reference_layout.embodiment:
            raise ValueError(
                f"Embodiment mismatch between {bundles[0].root} and {bundle.root}: "
                f"{reference_layout.embodiment} != {layout.embodiment}"
            )


def merge_datasets(
    bundles: list[V30DatasetBundle],
    output_root: Path,
    *,
    overwrite: bool,
    link_mode: MergeLinkMode = "copy",
    action_horizon: int = 50,
    write_output_norm_stats: bool = False,
    video_worker_threads: int | None = None,
    finalize_progress: lerobot_v30.FinalizeProgressCallback | None = None,
) -> dict[str, Any]:
    summary = lerobot_v30.merge_datasets(
        bundles,
        output_root,
        overwrite=overwrite,
        prepare_episode=_prepare_canonical_episode,
        link_mode=link_mode,
        video_worker_threads=video_worker_threads,
        build_output_info=_build_output_info,
        select_source_columns=_select_source_columns,
        validate_bundles=_validate_bundles,
        finalize_progress=finalize_progress,
    )

    if write_output_norm_stats and summary["status"] == "converted":
        common._write_norm_stats(
            Path(summary["output_root"]),
            action_horizon=action_horizon,
            stats_batch_size=32,
            run_compute_stats=True,
        )
    return summary
