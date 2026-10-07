#!/usr/bin/env python3
"""Convert extracted InternData-A1 datasets into embodiment-level LeRobot v3 datasets."""
# ruff: noqa: E402, SLF001

from __future__ import annotations

from collections import defaultdict
import dataclasses
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any
import warnings

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
warnings.filterwarnings(
    "ignore",
    message="The pynvml package is deprecated\\. Please install nvidia-ml-py instead\\..*",
    category=FutureWarning,
)

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
from openpi.datasets.specs import intern_a1 as common

ray, _HAS_RAY = ray_runtime.import_optional_ray()

CONSOLE = progress_display.get_console()


@dataclasses.dataclass(frozen=True)
class Args:
    output_dir: Path
    input_root: Path = common.DEFAULT_RAW_OUTPUT_ROOT
    staging_root: Path | None = None
    asset_output_dir: Path | None = None
    ray_temp_root: Path | None = None
    cleanup_tmp_on_success: bool = False
    embodiments: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    action_horizon: int = 50
    stats_batch_size: int = 32
    limit: int | None = None
    max_episodes_per_dataset: int | None = None
    resume: bool = False
    overwrite: bool = False
    cleanup_resolved_failures: bool = False
    run_compute_stats: bool = False
    write_output_norm_stats: bool = False
    write_asset_norm_stats: bool = False
    validate_source_videos: bool = True
    conversion_num_workers: int | None = None
    episode_commit_batch_size: int | None = None
    max_inflight_episodes: int | None = None
    video_link_mode: common.MergeLinkMode = "copy"
    push_to_hub: bool = False
    hub_owner: str | None = None
    summary_path: Path | None = None


def _resolve_workers(requested: int | None, job_count: int) -> int:
    if job_count <= 0:
        return 1
    cpu_count = os.cpu_count() or 1
    if requested is None:
        requested = min(cpu_count, job_count)
    return max(1, min(requested, cpu_count, job_count))


def _resolve_ray_prepare_cpus() -> float:
    raw = os.environ.get("RAY_PREPARE_CPUS", "0.5")
    try:
        value = float(raw)
    except ValueError:
        value = 0.5
    return max(0.1, value)


RAY_PREPARE_CPUS = _resolve_ray_prepare_cpus()


def _source_episode_key(input_root: Path, dataset_dir: Path, parquet_path: Path) -> str:
    dataset_dir = dataset_dir.expanduser().resolve()
    parquet_path = parquet_path.expanduser().resolve()
    try:
        dataset_prefix = dataset_dir.relative_to(input_root.expanduser().resolve()).as_posix()
    except ValueError:
        dataset_prefix = dataset_dir.as_posix()
    try:
        relative_parquet = parquet_path.relative_to(dataset_dir).as_posix()
    except ValueError:
        relative_parquet = parquet_path.name
    return f"{dataset_prefix}/{relative_parquet}"


def _select_grouped_dataset_dirs(args: Args) -> dict[str, list[Path]]:
    discovered = common.discover_dataset_dirs(
        args.input_root,
        embodiments=args.embodiments,
        categories=args.categories,
    )
    items = [(embodiment, path) for embodiment in sorted(discovered) for path in discovered[embodiment]]
    if args.limit is not None:
        items = items[: args.limit]

    grouped: dict[str, list[Path]] = defaultdict(list)
    for embodiment, path in items:
        grouped[embodiment].append(path)
    return {embodiment: paths for embodiment, paths in grouped.items() if paths}


def _count_selected_episodes(dataset_dirs: list[Path], max_episodes_per_dataset: int | None) -> int:
    return sum(
        min(
            len(sorted(dataset_dir.glob("data/chunk-*/episode_*.parquet"))),
            max_episodes_per_dataset if max_episodes_per_dataset is not None else 1 << 30,
        )
        for dataset_dir in dataset_dirs
    )


def _prepare_episode_payload(job: dict[str, Any]) -> dict[str, Any]:
    _ensure_import_paths()
    warnings.filterwarnings(
        "ignore",
        message="The pynvml package is deprecated\\. Please install nvidia-ml-py instead\\..*",
        category=FutureWarning,
    )

    from openpi.datasets.specs import intern_a1 as convert_inner

    try:
        result = convert_inner._prepare_episode(
            dataset_dir=Path(job["dataset_dir"]),
            parquet_path=Path(job["parquet_path"]),
            info=dict(job["info"]),
            layout=convert_inner.InternA1Layout(**dict(job["layout"])),
            task_lookup={int(key): str(value) for key, value in dict(job["task_lookup"]).items()},
            validate_source_videos=bool(job.get("validate_source_videos", True)),
        )
        result["status"] = "converted"
    except Exception as exc:
        result = {
            "status": "failed",
            "error": str(exc),
            "summary": {
                "raw_episode_index": int(convert_inner._episode_index_from_path(Path(job["parquet_path"]))),
                "frame_count": 0,
            },
        }
    result["job_index"] = int(job["job_index"])
    result["dataset_dir"] = str(job["dataset_dir"])
    result["source_episode_key"] = str(job["source_episode_key"])
    result["parquet_path"] = str(job["parquet_path"])
    return result


@ray.remote(num_cpus=RAY_PREPARE_CPUS)
def _prepare_episode_remote(job: dict[str, Any]) -> dict[str, Any]:
    return _prepare_episode_payload(job)


def _build_embodiment_dataset(job: dict[str, Any], *, workers: int) -> dict[str, Any]:
    _ensure_import_paths()

    from openpi.datasets.specs import intern_a1 as convert_inner
    from openpi.shared import normalize as normalize_inner

    embodiment = str(job["embodiment"])
    input_root = Path(job["input_root"])
    dataset_dirs = [Path(path) for path in job["dataset_dirs"]]
    output_dir = Path(job["output_dir"])
    asset_output_dir_raw = str(job.get("asset_output_dir", ""))
    asset_output_dir = Path(asset_output_dir_raw) if asset_output_dir_raw else None
    resume = bool(job.get("resume", False))
    overwrite = bool(job["overwrite"])
    max_episodes_per_dataset = job["max_episodes_per_dataset"]
    action_horizon = int(job["action_horizon"])
    stats_batch_size = int(job["stats_batch_size"])
    run_compute_stats = bool(job["run_compute_stats"])
    write_output_norm_stats = bool(job["write_output_norm_stats"])
    write_asset_norm_stats = bool(job["write_asset_norm_stats"])
    validate_source_videos = bool(job.get("validate_source_videos", True))
    cleanup_resolved_failures = bool(job.get("cleanup_resolved_failures", False))
    episode_commit_batch_size = job["episode_commit_batch_size"]
    max_inflight_episodes = int(job["max_inflight_episodes"])
    video_link_mode = str(job["video_link_mode"])

    if output_dir.exists():
        if resume:
            info_path = output_dir / "meta" / "info.json"
            if not info_path.exists():
                raise FileNotFoundError(f"Cannot resume without existing dataset metadata at {info_path}")
        elif not overwrite:
            return {
                "embodiment": embodiment,
                "output_dir": str(output_dir),
                "status": "skipped",
                "source_datasets": [str(path) for path in dataset_dirs],
                "episodes_written": 0,
                "frames_written": 0,
            }
        elif overwrite:
            shutil.rmtree(output_dir)

    first_info = common.load_dataset_info(dataset_dirs[0])
    first_layout = common.infer_layout_from_info(first_info, dataset_dir=dataset_dirs[0])
    repo_id = f"intern_a1_{embodiment}"
    committed_episodes = 0
    info_path = output_dir / "meta" / "info.json"
    if info_path.exists():
        if resume:
            committed_episodes = convert_inner.load_committed_episode_count(output_dir) or 0
            requested_episodes = [0] if committed_episodes > 0 else None
            dataset = convert_inner.open_dataset_for_append(
                repo_id,
                output_dir,
                episodes=requested_episodes,
            )
            convert_inner.validate_resume_dataset(
                dataset,
                info=first_info,
                layout=first_layout,
            )
        else:
            shutil.rmtree(output_dir)
            dataset = convert_inner.CanonicalInternA1Dataset.create(
                repo_id=repo_id,
                root=output_dir,
                fps=int(first_info["fps"]),
                robot_type="intern_a1",
                features=convert_inner._build_output_features(first_info, first_layout),
            )
    else:
        dataset = convert_inner.CanonicalInternA1Dataset.create(
            repo_id=repo_id,
            root=output_dir,
            fps=int(first_info["fps"]),
            robot_type="intern_a1",
            features=convert_inner._build_output_features(first_info, first_layout),
        )

    checkpoint = EpisodeCheckpointStore(output_dir, namespace="intern_a1")

    total_selected_episodes = _count_selected_episodes(dataset_dirs, max_episodes_per_dataset)
    commit_batch_size = convert_inner.resolve_episode_commit_batch_size(
        episode_commit_batch_size,
        workers=workers,
        episode_count=total_selected_episodes,
    )

    episode_jobs: list[dict[str, Any]] = []
    source_summaries_by_dir: dict[str, dict[str, Any]] = {}
    job_index = 0
    for dataset_dir in dataset_dirs:
        info = common.load_dataset_info(dataset_dir)
        layout = common.infer_layout_from_info(info, dataset_dir=dataset_dir)
        task_lookup = convert_inner._load_task_lookup(dataset_dir)
        parquet_paths = sorted(dataset_dir.glob("data/chunk-*/episode_*.parquet"))
        if max_episodes_per_dataset is not None:
            parquet_paths = parquet_paths[: max_episodes_per_dataset]

        source_summaries_by_dir[str(dataset_dir)] = {
            "dataset_dir": str(dataset_dir),
            "episodes_selected": len(parquet_paths),
            "episodes_already_present": 0,
            "episodes_written": 0,
            "episodes_failed": 0,
            "frames_written": 0,
        }
        for parquet_path in parquet_paths:
            episode_jobs.append(
                {
                    "job_index": job_index,
                    "dataset_dir": str(dataset_dir),
                    "parquet_path": str(parquet_path),
                    "info": dict(info),
                    "layout": dataclasses.asdict(layout),
                    "task_lookup": {str(key): value for key, value in task_lookup.items()},
                    "validate_source_videos": validate_source_videos,
                    "source_episode_key": _source_episode_key(input_root, dataset_dir, parquet_path),
                }
            )
            job_index += 1

    if committed_episodes > len(episode_jobs):
        raise ValueError(
            f"Cannot resume {output_dir}: existing dataset has {committed_episodes} committed episodes, "
            f"but the current selection only contains {len(episode_jobs)} episodes. "
            "Re-run with the same dataset filters or remove the existing output directory."
        )

    completed_keys = checkpoint.load_completed()
    if committed_episodes > 0:
        bootstrap_records = [
            {
                "key": str(job_item["source_episode_key"]),
                "episode_index": episode_index,
            }
            for episode_index, job_item in enumerate(episode_jobs[:committed_episodes])
            if str(job_item["source_episode_key"]) not in completed_keys
        ]
        if bootstrap_records:
            checkpoint.append_completed_many(bootstrap_records)
            completed_keys.update(str(record["key"]) for record in bootstrap_records)

    remaining_jobs: list[dict[str, Any]] = []
    for job_item in episode_jobs:
        dataset_key = str(job_item["dataset_dir"])
        if str(job_item["source_episode_key"]) in completed_keys:
            source_summaries_by_dir[dataset_key]["episodes_already_present"] += 1
        else:
            remaining_jobs.append(job_item)
    episode_jobs = remaining_jobs
    for next_job_index, job_item in enumerate(episode_jobs):
        job_item["job_index"] = next_job_index

    episodes_written = 0
    frames_written = 0
    failed_count = 0
    failure_records: list[dict[str, Any]] = []
    cleaned_failure_records = 0
    output_norm_stats = (
        convert_inner.init_output_norm_stats()
        if run_compute_stats and write_output_norm_stats and not resume
        else None
    )

    def _record_failed_payload(payload: dict[str, Any], *, error: str, stage: str) -> None:
        nonlocal failed_count
        dataset_key = str(payload["dataset_dir"])
        source_summaries_by_dir[dataset_key]["episodes_failed"] += 1
        failed_count += 1
        raw_episode_index = int(payload["summary"]["raw_episode_index"])
        failure_record = {
            "source_episode_key": str(payload["source_episode_key"]),
            "dataset_dir": dataset_key,
            "parquet_path": str(payload["parquet_path"]),
            "raw_episode_index": raw_episode_index,
            "stage": stage,
            "error": str(error),
        }
        failure_records.append(failure_record)
        checkpoint.append_failure(
            str(payload["source_episode_key"]),
            str(error),
            dataset_dir=dataset_key,
            parquet_path=str(payload["parquet_path"]),
            raw_episode_index=raw_episode_index,
            stage=stage,
        )

    def _persist_saved_payloads(saved_payloads: list[tuple[dict[str, Any], int]]) -> None:
        nonlocal episodes_written, frames_written
        checkpoint_records: list[dict[str, Any]] = []
        for item, episode_index in saved_payloads:
            frame_count = int(item["summary"]["frame_count"])
            episodes_written += 1
            frames_written += frame_count
            source_summaries_by_dir[str(item["dataset_dir"])]["episodes_written"] += 1
            source_summaries_by_dir[str(item["dataset_dir"])]["frames_written"] += frame_count
            if output_norm_stats is not None:
                convert_inner.update_output_norm_stats_from_prepared_episode(
                    output_norm_stats,
                    item,
                    action_horizon=action_horizon,
                )
            source_episode_key = str(item["source_episode_key"])
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
            convert_inner._save_prepared_episode_batch(
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
                convert_inner._save_prepared_episode_batch(
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
        prepare_task_id = progress.add_task(f"[cyan]Preparing {embodiment} episodes", total=len(episode_jobs))
        finalize_task_id = progress.add_task(f"[green]Finalizing {embodiment} episodes", total=len(episode_jobs))
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
                        buffered_payloads[ref_to_index.pop(ref)] = ray.get(ref)
                        progress.advance(prepare_task_id)
                        if job_cursor < len(episode_jobs):
                            episode_job = episode_jobs[job_cursor]
                            ref_to_index[_prepare_episode_remote.remote(episode_job)] = int(episode_job["job_index"])
                            job_cursor += 1
                    while True:
                        batch, next_commit_index = pop_contiguous_batch(
                            buffered_payloads,
                            next_index=next_commit_index,
                            batch_size=commit_batch_size,
                        )
                        if not batch:
                            break
                        failed_batch = [item for item in batch if item["status"] != "converted"]
                        if failed_batch:
                            for item in failed_batch:
                                _record_failed_payload(item, error=str(item["error"]), stage="prepare")
                            progress.advance(finalize_task_id, advance=len(failed_batch))
                            batch = [item for item in batch if item["status"] == "converted"]
                        if not batch:
                            continue
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
                    while len(ready_payloads) >= commit_batch_size:
                        batch = ready_payloads[:commit_batch_size]
                        del ready_payloads[:commit_batch_size]
                        progress.advance(finalize_task_id, advance=committer.submit(batch))
                    progress.advance(finalize_task_id, advance=committer.drain_completed())
                if ready_payloads:
                    progress.advance(finalize_task_id, advance=committer.submit(ready_payloads))
            progress.advance(finalize_task_id, advance=committer.close())
            committer = None
        finally:
            if committer is not None:
                committer.close()

    dataset.consolidate(run_compute_stats=run_compute_stats)
    finalized_norm_stats = (
        convert_inner.finalize_output_norm_stats(output_norm_stats)
        if output_norm_stats is not None
        else None
    )
    if cleanup_resolved_failures:
        cleaned_failure_records = checkpoint.prune_failures_for_completed(completed_keys)
    convert_inner._write_norm_stats(
        output_dir,
        action_horizon=action_horizon,
        stats_batch_size=stats_batch_size,
        run_compute_stats=run_compute_stats and write_output_norm_stats,
        norm_stats=finalized_norm_stats,
    )
    if run_compute_stats and write_asset_norm_stats:
        if asset_output_dir is None:
            raise ValueError("asset_output_dir is required when write_asset_norm_stats is enabled.")
        normalize_inner.save(
            asset_output_dir / embodiment,
            finalized_norm_stats
            or convert_inner.compute_direct_norm_stats(
                [output_dir],
                action_horizon,
                batch_size=stats_batch_size,
            ),
        )

    return {
        "embodiment": embodiment,
        "output_dir": str(output_dir),
        "status": "converted",
        "source_datasets": [source_summaries_by_dir[key] for key in sorted(source_summaries_by_dir)],
        "episodes_already_present": sum(
            int(summary["episodes_already_present"])
            for summary in source_summaries_by_dir.values()
        ),
        "episodes_written": episodes_written,
        "episodes_failed": failed_count,
        "frames_written": frames_written,
        "failures": failure_records,
        "checkpoint_path": str(checkpoint.completed_path),
        "failure_path": str(checkpoint.failed_path),
        "cleaned_failure_records": cleaned_failure_records,
    }


def main(args: Args) -> None:
    if args.push_to_hub and not args.hub_owner:
        raise ValueError("hub_owner is required when push_to_hub is enabled.")
    if args.write_output_norm_stats and not args.run_compute_stats:
        raise ValueError("--write-output-norm-stats requires --run-compute-stats.")
    if args.write_asset_norm_stats and not args.run_compute_stats:
        raise ValueError("--write-asset-norm-stats requires --run-compute-stats.")
    if args.write_asset_norm_stats and args.asset_output_dir is None:
        raise ValueError("--write-asset-norm-stats requires --asset-output-dir.")

    grouped_dataset_dirs = _select_grouped_dataset_dirs(args)
    if not grouped_dataset_dirs:
        raise FileNotFoundError(f"No extracted InternData-A1 datasets found under {args.input_root}.")

    output_root = args.output_dir.expanduser().resolve()
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive.")
    if args.staging_root is not None:
        CONSOLE.print("[yellow]Ignoring --staging-root: shard mode writes directly to --output-dir.")
    if args.video_link_mode != "copy":
        CONSOLE.print("[yellow]Ignoring --video-link-mode: conversion now always writes real shard files.")
    build_root = output_root
    using_staging = False
    asset_output_dir = args.asset_output_dir.expanduser().resolve() if args.asset_output_dir is not None else None
    if output_root.exists() and not (args.overwrite or args.resume):
        raise FileExistsError(f"Output already exists: {output_root}")
    if build_root.exists() and args.overwrite:
        staging.remove_tree(build_root)
    build_root.mkdir(parents=True, exist_ok=True)
    if asset_output_dir is not None:
        asset_output_dir.mkdir(parents=True, exist_ok=True)
    ray_temp_dir = ray_runtime.resolve_ray_temp_dir(
        label="intern-a1-v3",
        unique_key=str(build_root),
        requested_root=args.ray_temp_root,
    )
    ray_runtime.configure_process_temp_dir(ray_temp_dir)

    jobs = [
        {
            "embodiment": embodiment,
            "input_root": str(args.input_root),
            "dataset_dirs": [str(path) for path in paths],
            "output_dir": str(build_root / embodiment),
            "asset_output_dir": str(asset_output_dir) if asset_output_dir is not None else "",
            "resume": args.resume,
            "overwrite": args.overwrite,
            "cleanup_resolved_failures": args.cleanup_resolved_failures,
            "max_episodes_per_dataset": args.max_episodes_per_dataset,
            "action_horizon": args.action_horizon,
            "stats_batch_size": args.stats_batch_size,
            "run_compute_stats": args.run_compute_stats,
            "write_output_norm_stats": args.write_output_norm_stats,
            "write_asset_norm_stats": args.write_asset_norm_stats,
            "validate_source_videos": args.validate_source_videos,
            "episode_commit_batch_size": args.episode_commit_batch_size,
            "video_link_mode": args.video_link_mode,
            "selected_episode_count": _count_selected_episodes(paths, args.max_episodes_per_dataset),
        }
        for embodiment, paths in sorted(grouped_dataset_dirs.items())
    ]

    total_selected_episodes = sum(
        int(job["selected_episode_count"])
        for job in jobs
    )
    workers = _resolve_workers(args.conversion_num_workers, total_selected_episodes)
    if not _HAS_RAY and workers > 1 and total_selected_episodes > 1:
        CONSOLE.print("[yellow]ray is not installed; falling back to serial conversion (workers=1).[/yellow]")
        workers = 1
    for job in jobs:
        job["max_inflight_episodes"] = resolve_max_inflight_episodes(
            args.max_inflight_episodes,
            workers=workers,
            episode_count=int(job["selected_episode_count"]),
            prepare_task_cpus=RAY_PREPARE_CPUS,
        )
    CONSOLE.print(
        progress_display.render_key_value_panel(
            "InternData-A1 v3 Conversion",
            [
                ("Input Root", args.input_root),
                ("Output Dir", output_root),
                ("Build Root", build_root if using_staging else "in-place"),
                ("Asset Dir", asset_output_dir if args.write_asset_norm_stats else "disabled"),
                ("Embodiments", ", ".join(embodiment for embodiment, _ in sorted(grouped_dataset_dirs.items()))),
                ("Selected Episodes", total_selected_episodes),
                ("Ray Workers", workers),
                ("Ray Prepare Task CPUs", RAY_PREPARE_CPUS),
                ("Max In-Flight Episodes", max(int(job["max_inflight_episodes"]) for job in jobs)),
                ("Episode Commit Batch", args.episode_commit_batch_size or "auto"),
                ("Ray Temp", ray_temp_dir),
            ],
        )
    )

    if workers > 1 and total_selected_episodes > 1:
        ray_runtime.init_ray(num_workers=workers, temp_dir=ray_temp_dir, python_paths=PYTHON_PATHS)

    try:
        results = []
        for job in jobs:
            results.append(_build_embodiment_dataset(job, workers=workers))
    finally:
        if ray.is_initialized():
            ray.shutdown()

    results.sort(key=lambda item: str(item["embodiment"]))
    hub_results: list[dict[str, Any]] = []
    if args.push_to_hub:
        with progress_display.create_progress(console=CONSOLE) as progress:
            push_task_id = progress.add_task("[yellow]Pushing datasets to Hub", total=len(results))
            for result in results:
                if result["status"] != "converted":
                    hub_results.append(
                        {
                            "embodiment": result["embodiment"],
                            "status": "skipped",
                            "reason": f"dataset status is {result['status']}",
                        }
                    )
                    progress.advance(push_task_id)
                    continue
                repo_id = hub_push.build_repo_id(args.hub_owner or "", "intern-a1-v3", str(result["embodiment"]))
                try:
                    hub_push.push_dataset_root_to_hub(
                        Path(result["output_dir"]),
                        repo_id,
                        tags=tuple(dict.fromkeys(("intern-a1", str(result["embodiment"])))),
                        push_videos=True,
                    )
                    hub_results.append({"embodiment": result["embodiment"], "repo_id": repo_id, "status": "pushed"})
                except Exception as exc:
                    hub_results.append(
                        {
                            "embodiment": result["embodiment"],
                            "repo_id": repo_id,
                            "status": "failed",
                            "error": str(exc),
                        }
                    )
                progress.advance(push_task_id)
    summary = {
        "input_root": str(args.input_root),
        "output_root": str(output_root),
        "build_root": str(build_root),
        "asset_output_dir": str(asset_output_dir) if asset_output_dir is not None else None,
        "results": results,
        "hub_push": hub_results,
        "max_inflight_episodes": max(int(job["max_inflight_episodes"]) for job in jobs),
        "episode_commit_batch_size": args.episode_commit_batch_size,
        "totals": {
            "datasets_written": sum(1 for item in results if item["status"] == "converted"),
            "episodes_already_present": sum(int(item.get("episodes_already_present", 0)) for item in results),
            "episodes_written": sum(int(item["episodes_written"]) for item in results),
            "episodes_failed": sum(int(item.get("episodes_failed", 0)) for item in results),
            "frames_written": sum(int(item["frames_written"]) for item in results),
            "cleaned_failure_records": sum(int(item.get("cleaned_failure_records", 0)) for item in results),
        },
    }
    summary_path = args.summary_path.expanduser().resolve() if args.summary_path else (output_root / "build_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.cleanup_tmp_on_success:
        cleaned = ray_runtime.cleanup_temp_paths([ray_temp_dir])
        if cleaned:
            CONSOLE.print(f"[green]Cleaned temporary paths:[/green] {', '.join(str(path) for path in cleaned)}")
    CONSOLE.print(
        progress_display.render_summary_table(
            "InternData-A1 v3 Summary",
            [
                ("Datasets written", summary["totals"]["datasets_written"]),
                ("Episodes already present", summary["totals"]["episodes_already_present"]),
                ("Episodes written", summary["totals"]["episodes_written"]),
                ("Episodes failed", summary["totals"]["episodes_failed"]),
                ("Frames written", summary["totals"]["frames_written"]),
                ("Resolved failures cleaned", summary["totals"]["cleaned_failure_records"]),
                ("Summary path", summary_path),
            ],
        )
    )

if __name__ == "__main__":
    main(tyro.cli(Args))
