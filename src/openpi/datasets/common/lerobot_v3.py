# Copyright 2026 PLaW-VLA authors.
# SPDX-License-Identifier: Apache-2.0 AND MIT
# LeRobot-derived writer portions: Copyright Hugging Face contributors (Apache-2.0).
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Dataset writer portions adapted from Any4LeRobot.
# Copyright (c) 2025 Qizhi Chen (MIT).
# Modified for PLaW-VLA; see NOTICE.
from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
import shutil
from typing import Any, Literal

import datasets
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.compute_stats import compute_episode_stats
from lerobot.datasets.feature_utils import validate_episode_buffer
from lerobot.datasets.io_utils import get_file_size_in_mb
from lerobot.datasets.io_utils import write_info
from lerobot.datasets.io_utils import write_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import flatten_dict
from lerobot.datasets.utils import update_chunk_file_indices
from lerobot.datasets.video_utils import get_video_duration_in_s
import numpy as np
import pyarrow.parquet as pq

from openpi.datasets.common.video_concat import concatenate_video_files

MergeLinkMode = Literal["symlink", "hardlink", "copy", "auto"]


@dataclass
class _PendingVideoShard:
    video_key: str
    chunk_index: int
    file_index: int
    output_path: Path
    pending_paths: list[Path] = field(default_factory=list)
    size_in_mb: float = 0.0
    duration_s: float = 0.0


def _clone_pending_video_shard(pending: _PendingVideoShard) -> _PendingVideoShard:
    return _PendingVideoShard(
        video_key=pending.video_key,
        chunk_index=pending.chunk_index,
        file_index=pending.file_index,
        output_path=pending.output_path,
        pending_paths=list(pending.pending_paths),
        size_in_mb=pending.size_in_mb,
        duration_s=pending.duration_s,
    )


def link_or_copy_path(source: Path, destination: Path, *, link_mode: MergeLinkMode = "copy") -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()

    if link_mode == "auto":
        try:
            destination.hardlink_to(source)
            return
        except OSError:
            pass
        try:
            destination.symlink_to(source)
            return
        except OSError:
            pass
        shutil.copy2(source, destination)
        return

    if link_mode == "symlink":
        destination.symlink_to(source)
        return
    if link_mode == "hardlink":
        try:
            destination.hardlink_to(source)
            return
        except OSError:
            pass

    shutil.copy2(source, destination)


def _ordered_video_episode_metadata(video_keys: list[str], raw_metadata: dict[str, Any]) -> dict[str, Any]:
    ordered_metadata: dict[str, Any] = {}
    for video_key in video_keys:
        for suffix in ("chunk_index", "file_index", "from_timestamp", "to_timestamp"):
            metadata_key = f"videos/{video_key}/{suffix}"
            ordered_metadata[metadata_key] = raw_metadata[metadata_key]
    return ordered_metadata


def _episode_value(episode: dict[str, Any], key: str) -> Any:
    value = episode[key]
    if isinstance(value, list | tuple):
        return value[0]
    if isinstance(value, np.ndarray):
        return value.reshape(-1)[0]
    return value


def _write_video_shard(
    *,
    existing_shard: Path | None,
    appended_episode_paths: list[Path],
    output_path: Path,
) -> bool:
    if not appended_episode_paths:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if existing_shard is None and len(appended_episode_paths) == 1:
        if output_path.exists() or output_path.is_symlink():
            output_path.unlink()
        shutil.copy2(appended_episode_paths[0], output_path)
        return True

    input_paths: list[Path | str] = []
    if existing_shard is not None:
        input_paths.append(existing_shard)
    input_paths.extend(appended_episode_paths)
    concatenate_video_files(input_paths, output_path)
    return True


def _video_duration_tolerance_s(fps: float | None) -> float:
    fps_value = float(fps or 0.0)
    if fps_value <= 0:
        return 0.1
    return max(0.1, 0.5 / fps_value)


def _validate_written_video_shard_duration(output_path: Path, *, expected_duration_s: float, fps: float | None) -> None:
    actual_duration_s = float(get_video_duration_in_s(output_path))
    tolerance_s = _video_duration_tolerance_s(fps)
    if abs(actual_duration_s - expected_duration_s) > tolerance_s:
        raise RuntimeError(
            f"Video shard duration mismatch for {output_path}: expected {expected_duration_s:.6f}s, "
            f"got {actual_duration_s:.6f}s."
        )


class DirectVideoLeRobotDataset(LeRobotDataset):
    """LeRobot v3 dataset that packs source episode videos into real MP4 shards."""

    def _save_episode_data(self, episode_buffer: dict[str, Any]) -> dict[str, Any]:
        """Write parquet rows without the image-embedding pass.

        Direct-video datasets keep videos outside parquet, so the upstream
        embed_images() dataset.map(...) step is wasted work here.
        """
        ep_dict = {key: episode_buffer[key] for key in self.hf_features}
        ep_dataset = datasets.Dataset.from_dict(ep_dict, features=self.hf_features, split="train")
        ep_num_frames = len(ep_dataset)

        if self.latest_episode is None:
            chunk_idx, file_idx = 0, 0
            global_frame_index = 0
            self._current_file_start_frame = 0
            if self.meta.episodes is not None and len(self.meta.episodes) > 0:
                latest_ep = self.meta.episodes[-1]
                global_frame_index = latest_ep["dataset_to_index"]
                chunk_idx = latest_ep["data/chunk_index"]
                file_idx = latest_ep["data/file_index"]
                chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, self.meta.chunks_size)
                self._current_file_start_frame = global_frame_index
        else:
            latest_ep = self.latest_episode
            chunk_idx = latest_ep["data/chunk_index"]
            file_idx = latest_ep["data/file_index"]
            global_frame_index = latest_ep["index"][-1] + 1

            latest_path = self.root / self.meta.data_path.format(chunk_index=chunk_idx, file_index=file_idx)
            latest_size_in_mb = get_file_size_in_mb(latest_path)

            frames_in_current_file = global_frame_index - self._current_file_start_frame
            av_size_per_frame = latest_size_in_mb / frames_in_current_file if frames_in_current_file > 0 else 0

            if (
                latest_size_in_mb + av_size_per_frame * ep_num_frames >= self.meta.data_files_size_in_mb
                or self._writer_closed_for_reading
            ):
                chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, self.meta.chunks_size)
                self._close_writer()
                self._writer_closed_for_reading = False
                self._current_file_start_frame = global_frame_index

        ep_dict["data/chunk_index"] = chunk_idx
        ep_dict["data/file_index"] = file_idx

        path = self.root / self.meta.data_path.format(chunk_index=chunk_idx, file_index=file_idx)
        path.parent.mkdir(parents=True, exist_ok=True)

        table = ep_dataset.with_format("arrow")[:]
        if not self.writer:
            self.writer = pq.ParquetWriter(
                path,
                schema=table.schema,
                compression="snappy",
                use_dictionary=True,
            )
        self.writer.write_table(table)

        metadata = {
            "data/chunk_index": chunk_idx,
            "data/file_index": file_idx,
            "dataset_from_index": global_frame_index,
            "dataset_to_index": global_frame_index + ep_num_frames,
        }

        self.latest_episode = {**ep_dict, **metadata}
        self._lazy_loading = True
        self._recorded_frames += ep_num_frames
        return metadata

    def _save_episode_metadata_batch(
        self,
        episode_infos: list[dict[str, Any]],
        *,
        per_episode_data_metadata: list[dict[str, Any]],
        per_episode_video_metadata: list[dict[str, Any]],
        on_episode_saved: Callable[[int, int], None] | None = None,
    ) -> None:
        if not episode_infos:
            return
        if len(per_episode_data_metadata) != len(episode_infos):
            raise ValueError("Expected data metadata for every episode in the batch.")
        if len(per_episode_video_metadata) != len(episode_infos):
            raise ValueError("Expected video metadata for every episode in the batch.")

        original_buffer_size = getattr(self.meta, "metadata_buffer_size", 1)
        self.meta.metadata_buffer_size = max(original_buffer_size, len(episode_infos))
        try:
            batch_stats: list[dict[str, dict[str, np.ndarray]]] = []
            for idx, episode_info in enumerate(episode_infos):
                episode_metadata = dict(per_episode_data_metadata[idx])
                if self.meta.video_keys:
                    episode_metadata.update(
                        _ordered_video_episode_metadata(self.meta.video_keys, per_episode_video_metadata[idx])
                    )

                episode_stats = dict(episode_info["episode_stats"])
                self.meta._save_episode_metadata(  # noqa: SLF001
                    {
                        "episode_index": int(episode_info["episode_index"]),
                        "tasks": list(episode_info["episode_tasks"]),
                        "length": int(episode_info["episode_length"]),
                        **episode_metadata,
                        **flatten_dict({"stats": episode_stats}),
                    }
                )
                if episode_stats:
                    batch_stats.append(episode_stats)

            self.meta._flush_metadata_buffer()  # noqa: SLF001

            self.meta.info["total_episodes"] += len(episode_infos)
            self.meta.info["total_frames"] += sum(int(item["episode_length"]) for item in episode_infos)
            self.meta.info["total_tasks"] = len(self.meta.tasks)
            self.meta.info["splits"] = {"train": f"0:{self.meta.info['total_episodes']}"}
            write_info(self.meta.info, self.root)

            if batch_stats:
                stats_inputs = [self.meta.stats, *batch_stats] if self.meta.stats is not None else batch_stats
                self.meta.stats = aggregate_stats(stats_inputs)
                write_stats(self.meta.stats, self.root)

            if on_episode_saved is not None:
                for idx, episode_info in enumerate(episode_infos):
                    on_episode_saved(idx, int(episode_info["episode_index"]))
        finally:
            self.meta.metadata_buffer_size = original_buffer_size

    def _prepare_episode_for_save(
        self,
        episode_buffer: dict[str, Any],
        *,
        total_episodes: int,
        total_frames: int,
    ) -> tuple[dict[str, Any], int, list[str], dict[str, dict]]:
        validate_episode_buffer(episode_buffer, total_episodes, self.features)

        prepared_buffer = dict(episode_buffer)
        episode_length = int(prepared_buffer.pop("size"))
        tasks = list(prepared_buffer.pop("task"))
        episode_tasks = list(dict.fromkeys(tasks))
        episode_index = int(prepared_buffer["episode_index"])

        prepared_buffer["index"] = list(range(total_frames, total_frames + episode_length))
        prepared_buffer["episode_index"] = [episode_index] * episode_length

        self.meta.save_episode_tasks(episode_tasks)
        prepared_buffer["task_index"] = [self.meta.get_task_index(task) for task in tasks]

        non_visual_buffer: dict[str, Any] = {}
        non_visual_features: dict[str, Any] = {}
        for key, ft in self.features.items():
            if key in {"index", "episode_index", "task_index"}:
                continue
            if ft["dtype"] in {"image", "video", "bool"}:
                continue
            value = prepared_buffer[key]
            if not hasattr(value, "shape"):
                value = np.stack(value)
                prepared_buffer[key] = value
            non_visual_buffer[key] = prepared_buffer[key]
            non_visual_features[key] = ft

        ep_stats = compute_episode_stats(non_visual_buffer, non_visual_features) if non_visual_buffer else {}
        return prepared_buffer, episode_length, episode_tasks, ep_stats

    def _resolve_video_duration_s(
        self,
        source_path: Path,
        *,
        known_duration_s: float | None = None,
        episode_length: int,
        probe_video_durations: bool,
    ) -> float:
        if known_duration_s is not None:
            return float(known_duration_s)
        if probe_video_durations:
            return float(get_video_duration_in_s(source_path))
        fps_value = float(getattr(self.meta, "fps", 0.0) or 0.0)
        if fps_value > 0:
            return float(episode_length) / fps_value
        return float(get_video_duration_in_s(source_path))

    def _video_output_path(self, video_key: str, chunk_index: int, file_index: int) -> Path:
        return self.root / self.meta.video_path.format(
            video_key=video_key,
            chunk_index=chunk_index,
            file_index=file_index,
        )

    def _initial_video_shard_indices(self, video_key: str) -> tuple[int, int]:
        latest_episode: dict[str, Any] | None = None
        if self.meta.latest_episode is not None and f"videos/{video_key}/chunk_index" in self.meta.latest_episode:
            latest_episode = self.meta.latest_episode
        elif self.meta.episodes is not None and len(self.meta.episodes) > 0:
            maybe_latest = self.meta.episodes[-1]
            if f"videos/{video_key}/chunk_index" in maybe_latest:
                latest_episode = maybe_latest

        if latest_episode is None:
            return 0, 0

        old_chunk = int(_episode_value(latest_episode, f"videos/{video_key}/chunk_index"))
        old_file = int(_episode_value(latest_episode, f"videos/{video_key}/file_index"))
        return update_chunk_file_indices(old_chunk, old_file, self.meta.chunks_size)

    def _get_pending_video_shard(self, video_key: str) -> _PendingVideoShard:
        if not hasattr(self, "_pending_video_shards"):
            self._pending_video_shards: dict[str, _PendingVideoShard] = {}
        pending = self._pending_video_shards.get(video_key)
        if pending is not None:
            return pending

        chunk_idx, file_idx = self._initial_video_shard_indices(video_key)
        pending = _PendingVideoShard(
            video_key=video_key,
            chunk_index=chunk_idx,
            file_index=file_idx,
            output_path=self._video_output_path(video_key, chunk_idx, file_idx),
        )
        self._pending_video_shards[video_key] = pending
        return pending

    def _snapshot_pending_video_shards(self) -> dict[str, _PendingVideoShard] | None:
        pending_states = getattr(self, "_pending_video_shards", None)
        if not pending_states:
            return None
        return {
            video_key: _clone_pending_video_shard(pending)
            for video_key, pending in pending_states.items()
        }

    def _restore_pending_video_shards(self, snapshot: dict[str, _PendingVideoShard] | None) -> None:
        if not snapshot:
            if hasattr(self, "_pending_video_shards"):
                delattr(self, "_pending_video_shards")
            return
        self._pending_video_shards = {
            video_key: _clone_pending_video_shard(pending)
            for video_key, pending in snapshot.items()
        }

    def _flush_pending_video_shard(self, pending: _PendingVideoShard) -> bool:
        if not pending.pending_paths:
            return False
        _write_video_shard(
            existing_shard=None,
            appended_episode_paths=pending.pending_paths,
            output_path=pending.output_path,
        )
        _validate_written_video_shard_duration(
            pending.output_path,
            expected_duration_s=pending.duration_s,
            fps=getattr(self.meta, "fps", None),
        )
        pending.pending_paths = []
        pending.size_in_mb = 0.0
        pending.duration_s = 0.0
        return True

    def _advance_pending_video_shard(self, pending: _PendingVideoShard) -> None:
        pending.chunk_index, pending.file_index = update_chunk_file_indices(
            pending.chunk_index,
            pending.file_index,
            self.meta.chunks_size,
        )
        pending.output_path = self._video_output_path(
            pending.video_key,
            pending.chunk_index,
            pending.file_index,
        )

    def _save_sharded_videos(
        self,
        episode_infos: list[dict[str, Any]],
        videos: list[dict[str, Path] | None],
        *,
        video_durations: list[dict[str, float] | None] | None,
        probe_video_durations: bool,
    ) -> list[dict[str, Any]]:
        per_episode_video_metadata: list[dict[str, Any]] = [{} for _ in episode_infos]
        if not self.meta.video_keys:
            return per_episode_video_metadata

        for video_key in self.meta.video_keys:
            pending = self._get_pending_video_shard(video_key)

            for episode_idx, episode_videos in enumerate(videos):
                if episode_videos is None:
                    raise ValueError("Missing videos for an episode in the batch.")
                source_path = Path(episode_videos[video_key])
                if not source_path.exists():
                    raise FileNotFoundError(f"Missing source video for {video_key}: {source_path}")

                episode_size_mb = get_file_size_in_mb(source_path)
                known_duration_s = None
                if video_durations is not None and video_durations[episode_idx] is not None:
                    known_duration_s = float(video_durations[episode_idx][video_key])
                episode_duration_s = self._resolve_video_duration_s(
                    source_path,
                    known_duration_s=known_duration_s,
                    episode_length=int(episode_infos[episode_idx]["episode_length"]),
                    probe_video_durations=probe_video_durations,
                )

                if pending.pending_paths and pending.size_in_mb + episode_size_mb >= self.meta.video_files_size_in_mb:
                    self._flush_pending_video_shard(pending)
                    self._advance_pending_video_shard(pending)

                per_episode_video_metadata[episode_idx].update(
                    {
                        f"videos/{video_key}/chunk_index": pending.chunk_index,
                        f"videos/{video_key}/file_index": pending.file_index,
                        f"videos/{video_key}/from_timestamp": pending.duration_s,
                        f"videos/{video_key}/to_timestamp": pending.duration_s + episode_duration_s,
                    }
                )
                pending.pending_paths.append(source_path)
                pending.size_in_mb += episode_size_mb
                pending.duration_s += episode_duration_s

        return per_episode_video_metadata

    def flush_pending_video_shards(self) -> None:
        pending_states = getattr(self, "_pending_video_shards", None)
        if not pending_states:
            return

        dirty_states = [pending for pending in pending_states.values() if pending.pending_paths]
        if not dirty_states:
            return

        if len(dirty_states) == 1:
            self._flush_pending_video_shard(dirty_states[0])
        else:
            with ThreadPoolExecutor(max_workers=len(dirty_states), thread_name_prefix="dataset-video-flush") as executor:
                futures = [executor.submit(self._flush_pending_video_shard, pending) for pending in dirty_states]
                for future in futures:
                    future.result()

        for video_key in self.meta.video_keys:
            if self.meta.features[video_key].get("info"):
                continue
            try:
                self.meta.update_video_info(video_key)
            except TypeError as exc:
                if "positional argument" not in str(exc):
                    raise
                self.meta.update_video_info()
        write_info(self.meta.info, self.meta.root)

    def save_episode_batch(
        self,
        episode_batch: list[dict[str, Any]],
        *,
        videos: list[dict[str, Path] | None] | None = None,
        video_durations: list[dict[str, float] | None] | None = None,
        link_mode: MergeLinkMode = "copy",
        passthrough_video_workers: int | None = None,
        probe_video_durations: bool = False,
        on_episode_saved: Callable[[int, int], None] | None = None,
    ) -> None:
        del link_mode, passthrough_video_workers
        if not episode_batch:
            return

        if self.meta.video_keys:
            if videos is None or len(videos) != len(episode_batch):
                raise ValueError("videos must be provided for every episode when the dataset has video features.")
        elif videos is not None and len(videos) != len(episode_batch):
            raise ValueError("videos must either be omitted or match the number of episodes in the batch.")
        if video_durations is not None and len(video_durations) != len(episode_batch):
            raise ValueError("video_durations must either be omitted or match the number of episodes in the batch.")

        next_episode_index = self.meta.total_episodes
        next_frame_index = self.meta.total_frames

        prepared_episodes: list[dict[str, Any]] = []
        episode_infos: list[dict[str, Any]] = []
        for offset, raw_episode in enumerate(episode_batch):
            prepared_buffer, episode_length, episode_tasks, episode_stats = self._prepare_episode_for_save(
                raw_episode,
                total_episodes=next_episode_index + offset,
                total_frames=next_frame_index,
            )
            next_frame_index += episode_length
            prepared_episodes.append(prepared_buffer)
            episode_infos.append(
                {
                    "episode_index": int(prepared_buffer["episode_index"][0]),
                    "episode_length": episode_length,
                    "episode_tasks": episode_tasks,
                    "episode_stats": episode_stats,
                }
            )
        pending_video_snapshot = self._snapshot_pending_video_shards()
        try:
            per_episode_video_metadata = self._save_sharded_videos(
                episode_infos,
                videos if videos is not None else [None] * len(episode_infos),
                video_durations=video_durations,
                probe_video_durations=probe_video_durations,
            )
            per_episode_data_metadata = [self._save_episode_data(prepared_buffer) for prepared_buffer in prepared_episodes]
            self._save_episode_metadata_batch(
                episode_infos,
                per_episode_data_metadata=per_episode_data_metadata,
                per_episode_video_metadata=per_episode_video_metadata,
                on_episode_saved=on_episode_saved,
            )
        except Exception:
            self._restore_pending_video_shards(pending_video_snapshot)
            raise

    def save_episode(
        self,
        episode_data: dict | None = None,
        *,
        videos: dict[str, Path] | None = None,
        video_durations: dict[str, float] | None = None,
        link_mode: MergeLinkMode = "copy",
        passthrough_video_workers: int | None = None,
        probe_video_durations: bool = False,
    ) -> None:
        episode_buffer = episode_data if episode_data is not None else self.episode_buffer
        self.save_episode_batch(
            [episode_buffer],
            videos=[videos] if videos is not None else None,
            video_durations=[video_durations] if video_durations is not None else None,
            link_mode=link_mode,
            passthrough_video_workers=passthrough_video_workers,
            probe_video_durations=probe_video_durations,
        )

        if episode_data is None:
            self.clear_episode_buffer(delete_images=len(self.meta.image_keys) > 0)

    def finalize(self):
        self.flush_pending_video_shards()
        super().finalize()
