#!/usr/bin/env python3
"""Convert paired EgoDex raw episodes into a single LeRobot v3 dataset."""
# ruff: noqa: E402

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")

import ray
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
from openpi.datasets.specs import egodex as convert_egodex

CONSOLE = progress_display.get_console()


@dataclasses.dataclass(frozen=True)
class Args:
    output_dir: Path
    data_dir: Path = Path("data/raw/egodex")
    repo_name: str = convert_egodex.REPO_NAME
    ray_temp_root: Path | None = None
    cleanup_tmp_on_success: bool = False
    subdirs: tuple[str, ...] = ()
    task_names: tuple[str, ...] = ()
    start_task_name: str | None = None
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
    push_to_hub: bool = False
    hub_owner: str | None = None
    summary_json: Path | None = None


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


def _source_episode_key(job: dict[str, Any]) -> str:
    return "/".join(
        (
            str(job["split"]),
            str(job["part_name"]),
            str(job["task_name"]),
            str(job["file_index"]),
        )
    )


def _prepare_episode_payload(job: dict[str, Any]) -> dict[str, Any]:
    _ensure_import_paths()

    from openpi.datasets.specs import egodex as convert_inner

    episode = convert_inner.EpisodeSpec(
        split=str(job["split"]),
        part_name=str(job["part_name"]),
        part_id=int(job["part_id"]),
        task_name=str(job["task_name"]),
        task_id=int(job["task_id"]),
        file_index=int(job["file_index"]),
        mp4_path=Path(job["mp4_path"]),
        hdf5_path=Path(job["hdf5_path"]),
    )
    schema = convert_inner.JointSchema(joint_names=tuple(job["joint_names"]))
    result = convert_inner.prepare_episode_payload(
        episode,
        schema=schema,
        video_shape=tuple(job["video_shape"]),
        keep_extra_fields=bool(job["keep_extra_fields"]),
    )
    result["job_index"] = int(job["job_index"])
    result["source_episode_key"] = str(job["source_episode_key"])
    result["part_name"] = str(job["part_name"])
    result["task_name"] = str(job["task_name"])
    result["file_index"] = int(job["file_index"])
    result["mp4_path"] = str(job["mp4_path"])
    result["hdf5_path"] = str(job["hdf5_path"])
    return result


@ray.remote(num_cpus=RAY_PREPARE_CPUS)
def _prepare_episode_remote(job: dict[str, Any]) -> dict[str, Any]:
    return _prepare_episode_payload(job)


def main(args: Args) -> None:
    if args.push_to_hub and not args.hub_owner:
        raise ValueError("hub_owner is required when push_to_hub is enabled.")

    data_root = convert_egodex.resolve_data_root(args.data_dir.expanduser().resolve())
    output_root = args.output_dir.expanduser().resolve()
    if output_root == data_root or output_root in data_root.parents:
        raise ValueError("Output directory must not be the input directory or an ancestor of it.")
    build_root = output_root
    ray_temp_dir = ray_runtime.resolve_ray_temp_dir(
        label="egodex-v3",
        unique_key=str(build_root),
        requested_root=args.ray_temp_root,
    )
    ray_runtime.configure_process_temp_dir(ray_temp_dir)

    discovered_task_names = convert_egodex.discover_task_names(data_root, subdirs=list(args.subdirs) or None)
    if not discovered_task_names:
        raise ValueError(f"No EgoDex task directories found under {data_root}.")

    selected_task_names = convert_egodex.select_task_names(
        {task_name: [] for task_name in discovered_task_names},
        task_names=list(args.task_names) or None,
        start_task_name=args.start_task_name,
        max_tasks=args.max_tasks,
    )
    if not selected_task_names:
        raise ValueError("No EgoDex tasks matched the requested selection.")

    episodes = convert_egodex.discover_episode_pairs(
        data_root,
        subdirs=list(args.subdirs) or None,
        task_names=selected_task_names,
    )
    if not episodes:
        raise ValueError(f"No valid paired mp4+hdf5 episodes found under {data_root}.")

    grouped_episodes = convert_egodex.group_episodes_by_task(episodes)

    schema = convert_egodex.infer_joint_schema(episodes[0].hdf5_path)
    video_shape = convert_egodex.infer_video_shape(episodes[0].mp4_path)
    task_totals: dict[str, dict[str, int]] = {
        task_name: {
            "episodes_selected": 0,
            "episodes_already_present": 0,
            "episodes_written": 0,
            "episodes_skipped": 0,
            "episodes_failed": 0,
            "frames_written": 0,
        }
        for task_name in selected_task_names
    }
    episode_jobs: list[dict[str, Any]] = []
    job_index = 0
    for task_name in selected_task_names:
        task_episodes = grouped_episodes[task_name]
        if args.episode_offset:
            task_episodes = task_episodes[args.episode_offset :]
        if args.episodes_per_task is not None:
            task_episodes = task_episodes[: args.episodes_per_task]
        for episode in task_episodes:
            job = {
                "job_index": job_index,
                "split": episode.split,
                "part_name": episode.part_name,
                "part_id": episode.part_id,
                "task_name": episode.task_name,
                "task_id": episode.task_id,
                "file_index": episode.file_index,
                "mp4_path": str(episode.mp4_path),
                "hdf5_path": str(episode.hdf5_path),
                "joint_names": list(schema.joint_names),
                "video_shape": list(video_shape),
                "keep_extra_fields": args.keep_extra_fields,
            }
            job["source_episode_key"] = _source_episode_key(job)
            episode_jobs.append(job)
            task_totals[task_name]["episodes_selected"] += 1
            job_index += 1

    if not episode_jobs:
        raise ValueError("No EgoDex episodes remain after applying task/episode filters.")

    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive.")
    if output_root.exists() and not (args.overwrite or args.resume):
        raise FileExistsError(f"Output already exists: {output_root}")
    if build_root.exists() and args.overwrite:
        staging.remove_tree(build_root)

    info_path = build_root / "meta" / "info.json"
    committed_episodes = 0
    if info_path.exists():
        if args.resume:
            committed_episodes = convert_egodex.load_committed_episode_count(build_root) or 0
            if committed_episodes > len(episode_jobs):
                raise ValueError(
                    f"Cannot resume {output_root}: existing dataset has {committed_episodes} committed episodes, "
                    f"but the current selection only contains {len(episode_jobs)} episodes. "
                    "Re-run with the same task filters or remove the existing output directory."
                )
            requested_episodes = [0] if committed_episodes > 0 else None
            dataset = convert_egodex.EgoDexDataset(
                repo_id=args.repo_name,
                root=build_root,
                episodes=requested_episodes,
            )
            convert_egodex.validate_resume_dataset(
                dataset,
                video_shape=video_shape,
                joint_count=len(schema.joint_names),
                keep_extra_fields=args.keep_extra_fields,
            )
        else:
            staging.remove_tree(build_root)
            dataset = convert_egodex.create_dataset(
                args.repo_name,
                video_shape,
                len(schema.joint_names),
                output_path=build_root,
                keep_extra_fields=args.keep_extra_fields,
            )
    else:
        if output_root.exists() and args.resume:
            raise FileNotFoundError(f"Cannot resume without existing dataset metadata at {info_path}")
        dataset = convert_egodex.create_dataset(
            args.repo_name,
            video_shape,
            len(schema.joint_names),
            output_path=build_root,
            keep_extra_fields=args.keep_extra_fields,
        )

    checkpoint = EpisodeCheckpointStore(build_root, namespace="egodex")
    completed_keys = checkpoint.load_completed()
    if committed_episodes > 0:
        bootstrap_records = [
            {
                "key": str(job["source_episode_key"]),
                "episode_index": episode_index,
            }
            for episode_index, job in enumerate(episode_jobs[:committed_episodes])
            if str(job["source_episode_key"]) not in completed_keys
        ]
        if bootstrap_records:
            checkpoint.append_completed_many(bootstrap_records)
            completed_keys.update(str(record["key"]) for record in bootstrap_records)

    remaining_jobs: list[dict[str, Any]] = []
    for job in episode_jobs:
        if str(job["source_episode_key"]) in completed_keys:
            task_totals[str(job["task_name"])]["episodes_already_present"] += 1
        else:
            remaining_jobs.append(job)
    episode_jobs = remaining_jobs
    for next_job_index, job in enumerate(episode_jobs):
        job["job_index"] = next_job_index

    workers = _resolve_workers(args.conversion_num_workers, len(episode_jobs))
    max_inflight_episodes = resolve_max_inflight_episodes(
        args.max_inflight_episodes,
        workers=workers,
        episode_count=len(episode_jobs),
        prepare_task_cpus=RAY_PREPARE_CPUS,
    )
    episode_commit_batch_size = convert_egodex.resolve_episode_commit_batch_size(
        args.episode_commit_batch_size,
        workers=workers,
        episode_count=len(episode_jobs),
    )
    CONSOLE.print(
        progress_display.render_key_value_panel(
            "EgoDex v3 Conversion",
            [
                ("Data Root", data_root),
                ("Output Dir", output_root),
                ("Selected Tasks", len(selected_task_names)),
                ("Selected Episodes", len(episode_jobs)),
                ("Keep Extra Fields", args.keep_extra_fields),
                ("Ray Workers", workers),
                ("Ray Prepare Task CPUs", RAY_PREPARE_CPUS),
                ("Max In-Flight Episodes", max_inflight_episodes),
                ("Episode Commit Batch", episode_commit_batch_size),
                ("Ray Temp", ray_temp_dir),
            ],
        )
    )

    failure_records: list[dict[str, Any]] = []
    cleaned_failure_records = 0

    def _record_problem_payload(
        payload: dict[str, Any],
        *,
        error: str,
        stage: str,
        counter_key: str,
    ) -> None:
        task_name = str(payload["task_name"])
        task_totals[task_name][counter_key] += 1
        failure_record = {
            "source_episode_key": str(payload["source_episode_key"]),
            "task_name": task_name,
            "part_name": str(payload["part_name"]),
            "file_index": int(payload["file_index"]),
            "stage": stage,
            "error": str(error),
            "mp4_path": str(payload["mp4_path"]),
            "hdf5_path": str(payload["hdf5_path"]),
        }
        failure_records.append(failure_record)
        checkpoint.append_failure(
            str(payload["source_episode_key"]),
            str(error),
            task_name=task_name,
            part_name=str(payload["part_name"]),
            file_index=int(payload["file_index"]),
            stage=stage,
            mp4_path=str(payload["mp4_path"]),
            hdf5_path=str(payload["hdf5_path"]),
        )

    def _persist_saved_payloads(saved_payloads: list[tuple[dict[str, Any], int]]) -> None:
        checkpoint_records: list[dict[str, Any]] = []
        for payload, episode_index in saved_payloads:
            task_name = str(payload["task_name"])
            task_totals[task_name]["episodes_written"] += 1
            task_totals[task_name]["frames_written"] += int(payload["frame_count"])
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
            convert_egodex.save_prepared_episode_batch(
                dataset,
                payload_batch,
                link_mode="copy",
                on_episode_saved=_on_episode_saved,
            )
            _persist_saved_payloads(saved_payloads)
            return
        except Exception as exc:
            if saved_payloads:
                _persist_saved_payloads(saved_payloads)
            if len(payload_batch) == 1:
                if not saved_offsets:
                    _record_problem_payload(
                        payload_batch[0],
                        error=str(exc),
                        stage="commit",
                        counter_key="episodes_failed",
                    )
                return

        for batch_offset, payload in enumerate(payload_batch):
            if batch_offset in saved_offsets:
                continue
            try:
                single_saved_payloads: list[tuple[dict[str, Any], int]] = []
                convert_egodex.save_prepared_episode_batch(
                    dataset,
                    [payload],
                    link_mode="copy",
                    on_episode_saved=lambda _offset, episode_index, payload=payload: single_saved_payloads.append(
                        (payload, episode_index)
                    ),
                )
                _persist_saved_payloads(single_saved_payloads)
            except Exception as item_exc:
                _record_problem_payload(
                    payload,
                    error=str(item_exc),
                    stage="commit",
                    counter_key="episodes_failed",
                )

    if workers <= 1 or len(episode_jobs) == 1:
        with progress_display.create_progress(console=CONSOLE) as progress:
            prepare_task_id = progress.add_task("[cyan]Preparing EgoDex episodes", total=len(episode_jobs))
            finalize_task_id = progress.add_task("[green]Finalizing EgoDex episodes", total=len(episode_jobs))
            ready_payloads: list[dict[str, Any]] = []
            committer = OrderedAsyncBatchCommitter(
                _commit_payload_batch,
                max_pending_batches=2,
            )
            try:
                for job in episode_jobs:
                    result = _prepare_episode_payload(job)
                    progress.advance(prepare_task_id)
                    if result["status"] == "converted":
                        ready_payloads.append(result)
                        while len(ready_payloads) >= episode_commit_batch_size:
                            batch = ready_payloads[:episode_commit_batch_size]
                            del ready_payloads[:episode_commit_batch_size]
                            progress.advance(finalize_task_id, advance=committer.submit(batch))
                    else:
                        _record_problem_payload(
                            result,
                            error=str(result["reason"]),
                            stage="prepare",
                            counter_key="episodes_skipped",
                        )
                        progress.advance(finalize_task_id)
                    progress.advance(finalize_task_id, advance=committer.drain_completed())
                if ready_payloads:
                    progress.advance(finalize_task_id, advance=committer.submit(ready_payloads))
                progress.advance(finalize_task_id, advance=committer.close())
                committer = None
            finally:
                if committer is not None:
                    committer.close()
    else:
        ray_runtime.init_ray(num_workers=workers, temp_dir=ray_temp_dir, python_paths=PYTHON_PATHS)
        try:
            with progress_display.create_progress(console=CONSOLE) as progress:
                prepare_task_id = progress.add_task("[cyan]Preparing EgoDex episodes", total=len(episode_jobs))
                finalize_task_id = progress.add_task("[green]Finalizing EgoDex episodes", total=len(episode_jobs))
                ref_to_index: dict[ray.ObjectRef, int] = {}
                job_cursor = 0
                while job_cursor < len(episode_jobs) and len(ref_to_index) < max_inflight_episodes:
                    job = episode_jobs[job_cursor]
                    ref_to_index[_prepare_episode_remote.remote(job)] = int(job["job_index"])
                    job_cursor += 1
                buffered_results: dict[int, dict[str, Any]] = {}
                next_commit_index = 0
                wait_num_returns = max(1, min(8, max_inflight_episodes))
                committer = OrderedAsyncBatchCommitter(
                    _commit_payload_batch,
                    max_pending_batches=2,
                )
                try:
                    while ref_to_index:
                        done, _ = ray.wait(
                            list(ref_to_index),
                            num_returns=min(len(ref_to_index), wait_num_returns),
                        )
                        for ref in done:
                            buffered_results[ref_to_index.pop(ref)] = ray.get(ref)
                            progress.advance(prepare_task_id)
                            if job_cursor < len(episode_jobs):
                                job = episode_jobs[job_cursor]
                                ref_to_index[_prepare_episode_remote.remote(job)] = int(job["job_index"])
                                job_cursor += 1
                        while True:
                            batch, next_commit_index = pop_contiguous_batch(
                                buffered_results,
                                next_index=next_commit_index,
                                batch_size=episode_commit_batch_size,
                            )
                            if not batch:
                                break
                            non_converted = [item for item in batch if item["status"] != "converted"]
                            if non_converted:
                                for item in non_converted:
                                    _record_problem_payload(
                                        item,
                                        error=str(item["reason"]),
                                        stage="prepare",
                                        counter_key="episodes_skipped",
                                    )
                                progress.advance(finalize_task_id, advance=len(non_converted))
                                batch = [item for item in batch if item["status"] == "converted"]
                            if batch:
                                progress.advance(finalize_task_id, advance=committer.submit(batch))
                        progress.advance(finalize_task_id, advance=committer.drain_completed())
                    progress.advance(finalize_task_id, advance=committer.close())
                    committer = None
                finally:
                    if committer is not None:
                        committer.close()
        finally:
            ray.shutdown()

    dataset.finalize()
    if args.cleanup_resolved_failures:
        cleaned_failure_records = checkpoint.prune_failures_for_completed(completed_keys)
    convert_egodex.write_norm_stats(build_root, run_compute_stats=not args.skip_stats)
    metadata = {
        "repo_name": args.repo_name,
        "tasks": [
            {"task_name": task_name, **task_totals[task_name]}
            for task_name in sorted(task_totals)
        ],
        "video_shape": list(video_shape),
        "joint_names": list(schema.joint_names),
        "keep_extra_fields": args.keep_extra_fields,
    }
    (build_root / "egodex_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    hub_result: dict[str, Any] | None = None
    if args.push_to_hub:
        repo_id = hub_push.build_repo_id(args.hub_owner or "", "egodex-v3", args.repo_name)
        hub_push.push_dataset_root_to_hub(
            output_root,
            repo_id,
            tags=("egodex", "egocentric", "dexterous-manipulation"),
            license="cc-by-nc-nd-4.0",
            push_videos=True,
        )
        hub_result = {"repo_id": repo_id, "status": "pushed"}

    summary = {
        "data_root": str(data_root),
        "output_root": str(output_root),
        "build_root": str(build_root),
        "repo_name": args.repo_name,
        "tasks": metadata["tasks"],
        "totals": {
            "episodes_already_present": sum(item["episodes_already_present"] for item in metadata["tasks"]),
            "episodes_written": sum(item["episodes_written"] for item in metadata["tasks"]),
            "episodes_skipped": sum(item["episodes_skipped"] for item in metadata["tasks"]),
            "episodes_failed": sum(item["episodes_failed"] for item in metadata["tasks"]),
            "frames_written": sum(item["frames_written"] for item in metadata["tasks"]),
        },
        "max_inflight_episodes": max_inflight_episodes,
        "episode_commit_batch_size": episode_commit_batch_size,
        "checkpoint_path": str(checkpoint.completed_path),
        "failure_path": str(checkpoint.failed_path),
        "failures": failure_records,
        "cleaned_failure_records": cleaned_failure_records,
        "hub_push": hub_result,
    }
    summary_path = args.summary_json.expanduser().resolve() if args.summary_json else (output_root / "build_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    failed_episodes = summary["totals"]["episodes_failed"]
    empty_conversion = summary["totals"]["episodes_written"] + summary["totals"]["episodes_already_present"] == 0
    if args.cleanup_tmp_on_success and not (failed_episodes or empty_conversion):
        cleaned = ray_runtime.cleanup_temp_paths([ray_temp_dir])
        if cleaned:
            CONSOLE.print(f"[green]Cleaned temporary paths:[/green] {', '.join(str(path) for path in cleaned)}")
    CONSOLE.print(
        progress_display.render_summary_table(
            "EgoDex v3 Summary",
            [
                ("Episodes already present", summary["totals"]["episodes_already_present"]),
                ("Episodes written", summary["totals"]["episodes_written"]),
                ("Episodes skipped", summary["totals"]["episodes_skipped"]),
                ("Episodes failed", summary["totals"]["episodes_failed"]),
                ("Frames written", summary["totals"]["frames_written"]),
                ("Resolved failures cleaned", summary["cleaned_failure_records"]),
                ("Summary path", summary_path),
            ],
        )
    )
    if failed_episodes or empty_conversion:
        raise RuntimeError(
            f"EgoDex conversion is incomplete ({failed_episodes} episode(s) failed, "
            f"{summary['totals']['episodes_skipped']} skipped); see {summary_path} and the source selection."
        )

if __name__ == "__main__":
    main(tyro.cli(Args))
