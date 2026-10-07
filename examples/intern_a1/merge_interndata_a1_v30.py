#!/usr/bin/env python3
"""Merge extracted task-level InternData-A1 LeRobot v3 datasets into embodiment-level canonical v3 datasets."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import sys
import warnings

import tyro

warnings.filterwarnings(
    "ignore",
    message="The pynvml package is deprecated\\. Please install nvidia-ml-py instead\\..*",
    category=FutureWarning,
)

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
from openpi.datasets.specs import intern_a1_v30

CONSOLE = progress_display.get_console()


DEFAULT_TASK_LEVEL_V30_ROOT = Path("data/raw/intern_a1/lerobot_v30_raw")
DEFAULT_EMBODIMENT_V30_ROOT = Path("data/pretrain/intern_a1")


@dataclasses.dataclass(frozen=True)
class Args:
    output_dir: Path = DEFAULT_EMBODIMENT_V30_ROOT
    input_root: Path = DEFAULT_TASK_LEVEL_V30_ROOT
    embodiments: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    overwrite: bool = False
    action_horizon: int = 50
    write_output_norm_stats: bool = False
    video_worker_threads: int | None = None
    link_mode: intern_a1_v30.MergeLinkMode = "copy"
    summary_json: Path | None = None


def main(args: Args) -> None:
    grouped = intern_a1_v30.discover_task_level_dataset_dirs(
        args.input_root.expanduser().resolve(),
        embodiments=args.embodiments,
        categories=args.categories,
    )
    if not grouped:
        raise FileNotFoundError(f"No extracted InternData-A1 task-level v3 datasets found under {args.input_root}.")

    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    with progress_display.create_progress(console=CONSOLE) as progress:
        embodiment_task_id = progress.add_task("[cyan]Merging embodiment datasets", total=len(grouped))
        merge_task_id = progress.add_task("[green]Preparing merge", total=None)

        for embodiment, dataset_roots in grouped.items():
            progress.update(
                merge_task_id,
                description=f"[yellow]Loading {embodiment} task bundles",
                total=len(dataset_roots),
                completed=0,
            )
            bundles: list[intern_a1_v30.V30DatasetBundle] = []
            for path in dataset_roots:
                bundles.append(intern_a1_v30.load_v30_dataset_bundle(path))
                progress.advance(merge_task_id)

            def _on_finalize_progress(completed: int, total: int, *, embodiment: str = embodiment) -> None:
                progress.update(
                    merge_task_id,
                    description=f"[green]Merging {embodiment}",
                    total=max(1, total),
                    completed=completed,
                )

            summary = intern_a1_v30.merge_datasets(
                bundles,
                output_root / embodiment,
                overwrite=args.overwrite,
                link_mode=args.link_mode,
                action_horizon=args.action_horizon,
                write_output_norm_stats=args.write_output_norm_stats,
                video_worker_threads=args.video_worker_threads,
                finalize_progress=_on_finalize_progress,
            )
            if summary["status"] != "converted":
                progress.update(
                    merge_task_id,
                    description=f"[yellow]{embodiment} {summary['status']}",
                    total=1,
                    completed=1,
                )
            summary["embodiment"] = embodiment
            summaries.append(summary)
            progress.advance(embodiment_task_id)

        progress.update(merge_task_id, description="[green]Merge complete", total=1, completed=1)

    if args.summary_json is not None:
        args.summary_json.expanduser().resolve().write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Args))
