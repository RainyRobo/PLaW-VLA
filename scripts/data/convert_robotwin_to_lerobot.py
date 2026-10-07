#!/usr/bin/env python3
"""Convert task-level RoboTwin LeRobot v2.1 datasets into embodiment-level v3 datasets."""
# ruff: noqa: E402

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

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

from openpi.datasets.common import progress as progress_display
from openpi.datasets.common import ray_runtime
from openpi.datasets.common import staging
from openpi.datasets.specs import lerobot_v21
from openpi.datasets.specs import robotwin as robotwin_datasets

CONSOLE = progress_display.get_console()


@dataclasses.dataclass(frozen=True)
class Args:
    output_dir: Path
    input_roots: tuple[Path, ...] = robotwin_datasets.DEFAULT_INPUT_ROOTS
    staging_root: Path | None = None
    embodiments: tuple[str, ...] = ()
    link_mode: lerobot_v21.MergeLinkMode = "auto"
    overwrite: bool = False
    conversion_num_workers: int | None = None
    ray_temp_root: Path | None = None
    cleanup_tmp_on_success: bool = False
    compute_norm_stats: bool = False
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


@ray.remote(num_cpus=RAY_PREPARE_CPUS)
def _load_bundle_remote(dataset_dir: str) -> lerobot_v21.V21DatasetBundle:
    _ensure_import_paths()
    from openpi.datasets.specs import lerobot_v21 as lerobot_v21_inner

    return lerobot_v21_inner.load_v21_dataset_bundle(Path(dataset_dir))


def _load_bundles(dataset_dirs: list[Path], *, workers: int) -> list[lerobot_v21.V21DatasetBundle]:
    if workers <= 1 or len(dataset_dirs) <= 1:
        return [lerobot_v21.load_v21_dataset_bundle(path) for path in dataset_dirs]

    pending = [_load_bundle_remote.remote(str(path)) for path in dataset_dirs]
    bundles: list[lerobot_v21.V21DatasetBundle] = []
    with progress_display.create_progress(console=CONSOLE) as progress:
        task_id = progress.add_task("[cyan]Loading RobotWin v2.1 metadata", total=len(dataset_dirs))
        while pending:
            done, pending = ray.wait(pending, num_returns=1)
            bundles.extend(ray.get(done))
            progress.advance(task_id, len(done))
    return bundles


def _validate_output_path(path: Path, source_paths: tuple[Path, ...]) -> None:
    for source_path in source_paths:
        if path == source_path or path in source_path.parents:
            raise ValueError("Output directory must not be an input directory or an ancestor of it.")


def main(args: Args) -> None:
    grouped = robotwin_datasets.discover_child_datasets(args.input_roots, embodiments=args.embodiments)
    if not grouped:
        raise FileNotFoundError("No RobotWin LeRobot v2.1 child datasets matched the requested selection.")

    source_paths = tuple(path.expanduser().resolve() for path in args.input_roots) + tuple(
        dataset.root for datasets in grouped.values() for dataset in datasets
    )
    output_root = args.output_dir.expanduser().resolve()
    _validate_output_path(output_root, source_paths)
    if output_root.exists() and not args.overwrite:
        summaries: list[dict[str, object]] = []
        for embodiment in grouped:
            dataset_root = output_root / embodiment
            if not lerobot_v21.is_v3_dataset_root(dataset_root):
                raise FileExistsError(
                    f"Output already exists: {output_root}, but {dataset_root} is not a completed RobotWin v3 dataset."
                )
            summaries.append(
                {
                    "embodiment": embodiment,
                    "output_root": str(dataset_root),
                    "mode": "norm_stats_only" if args.compute_norm_stats else "existing_output",
                }
            )
            if args.compute_norm_stats:
                summaries[-1]["norm_stats"] = lerobot_v21.write_output_norm_stats(
                    dataset_root,
                    run_compute_stats=True,
                )
        if args.summary_json is not None:
            args.summary_json.expanduser().resolve().write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        print(json.dumps(summaries, indent=2))
        return
    should_stage = args.staging_root is not None or args.link_mode != "copy"
    build_root = (
        staging.resolve_staging_root(output_root, label="robotwin-v3", requested_root=args.staging_root)
        if should_stage
        else None
    ) or output_root
    _validate_output_path(build_root, source_paths)
    using_staging = build_root != output_root
    if output_root.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_root}")
    if build_root.exists():
        staging.remove_tree(build_root)
    build_root.mkdir(parents=True, exist_ok=True)
    total_child_datasets = sum(len(paths) for paths in grouped.values())
    workers = _resolve_workers(args.conversion_num_workers, total_child_datasets)
    ray_temp_dir = ray_runtime.resolve_ray_temp_dir(
        label="robotwin-v3",
        unique_key=str(build_root),
        requested_root=args.ray_temp_root,
    )
    ray_runtime.configure_process_temp_dir(ray_temp_dir)

    CONSOLE.print(
        progress_display.render_key_value_panel(
            "RobotWin v3 Conversion",
            [
                ("Input Roots", ", ".join(str(path) for path in args.input_roots)),
                ("Output Dir", output_root),
                ("Build Root", build_root if using_staging else "in-place"),
                ("Embodiments", ", ".join(grouped) if grouped else "-"),
                ("Child Datasets", total_child_datasets),
                ("Ray Workers", workers),
                ("Ray Prepare Task CPUs", RAY_PREPARE_CPUS),
                ("Ray Temp", ray_temp_dir),
            ],
        )
    )

    if workers > 1 and total_child_datasets > 1:
        ray_runtime.init_ray(num_workers=workers, temp_dir=ray_temp_dir, python_paths=PYTHON_PATHS)

    summaries: list[dict[str, object]] = []
    try:
        for embodiment, datasets in grouped.items():
            child_roots = [item.root for item in datasets]
            bundles = _load_bundles(child_roots, workers=workers)
            finalize_total = lerobot_v21.estimate_merge_finalize_steps(bundles)
            with progress_display.create_progress(console=CONSOLE) as progress:
                finalize_task_id = progress.add_task(
                    f"[green]Finalizing RobotWin {embodiment}",
                    total=max(1, finalize_total),
                )
                summary = lerobot_v21.merge_datasets(
                    None,
                    build_root / embodiment,
                    link_mode=args.link_mode,
                    overwrite=args.overwrite,
                    conversion_num_workers=workers,
                    bundles=bundles,
                    finalize_progress=lambda completed, total: progress.update(
                        finalize_task_id,
                        completed=completed,
                        total=max(1, total),
                    ),
                )
            if args.compute_norm_stats:
                summary["norm_stats"] = lerobot_v21.write_output_norm_stats(
                    build_root / embodiment,
                    run_compute_stats=True,
                )
            summary["embodiment"] = embodiment
            summaries.append(summary)
    finally:
        if ray.is_initialized():
            ray.shutdown()

    if using_staging:
        staging.publish_tree(build_root, output_root, overwrite=True)
        for summary in summaries:
            summary["output_root"] = str(output_root / str(summary["embodiment"]))
        staging.remove_tree(build_root)

    if args.summary_json is not None:
        args.summary_json.expanduser().resolve().write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    if args.cleanup_tmp_on_success:
        cleaned = ray_runtime.cleanup_temp_paths([ray_temp_dir])
        if cleaned:
            CONSOLE.print(f"[green]Cleaned temporary paths:[/green] {', '.join(str(path) for path in cleaned)}")

    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Args))
