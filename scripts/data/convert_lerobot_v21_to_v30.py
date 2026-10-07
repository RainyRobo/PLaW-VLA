#!/usr/bin/env python3
"""Shard-based LeRobot v2.1 -> v3.0 converter."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys

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

from lerobot.datasets.compute_stats import compute_episode_stats
import numpy as np
import pyarrow.parquet as pq

from openpi.datasets.common import progress as progress_display
from openpi.datasets.common import ray_runtime
from openpi.datasets.specs import lerobot_v21

CONSOLE = progress_display.get_console()


def _parse_vector_element_remaps(values: list[str]) -> tuple[lerobot_v21.VectorElementRemap, ...]:
    remaps: list[lerobot_v21.VectorElementRemap] = []
    for value in values:
        column_name, first_sep, remainder = value.partition(":")
        index_text, second_sep, mapping_text = remainder.partition(":")
        if not first_sep or not second_sep or not column_name or not index_text or not mapping_text:
            raise ValueError(
                f"Expected --vector-element-remap in the form column:index:src=dst[,src=dst...], got {value!r}."
            )

        mappings: list[tuple[float, float]] = []
        for pair_text in mapping_text.split(","):
            source_text, pair_sep, target_text = pair_text.partition("=")
            if not pair_sep or not source_text or not target_text:
                raise ValueError(
                    f"Expected each --vector-element-remap mapping to use src=dst syntax, got {pair_text!r}."
                )
            mappings.append((float(source_text), float(target_text)))

        remaps.append(
            lerobot_v21.VectorElementRemap(
                column_name=column_name,
                element_index=int(index_text),
                mapping=tuple(mappings),
            )
        )
    return tuple(remaps)


def _with_vector_remaps(
    bundle: lerobot_v21.V21DatasetBundle,
    remaps: tuple[lerobot_v21.VectorElementRemap, ...],
) -> lerobot_v21.V21DatasetBundle:
    columns = sorted({remap.column_name for remap in remaps})
    feature_specs = {name: bundle.info["features"][name] for name in columns}
    episodes = []
    for episode in bundle.episodes:
        table = pq.read_table(episode.source_data_path, columns=columns)
        table = lerobot_v21.apply_vector_element_remaps_to_table(table, remaps)
        episode_data = {name: np.asarray(table.column(name).to_pylist()) for name in columns}
        stats = dict(episode.stats)
        stats.update(compute_episode_stats(episode_data, feature_specs))
        episodes.append(dataclasses.replace(episode, stats=stats))
    return dataclasses.replace(bundle, episodes=tuple(episodes), vector_element_remaps=remaps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--conversion-num-workers", type=int, default=None)
    parser.add_argument("--ray-temp-root", type=Path, default=None)
    parser.add_argument("--cleanup-tmp-on-success", action="store_true")
    parser.add_argument("--compute-norm-stats", action="store_true")
    parser.add_argument(
        "--vector-element-remap",
        action="append",
        default=[],
        metavar="COLUMN:INDEX:SRC=DST[,SRC=DST...]",
        help=(
            "Optional remap for one element inside a vector-valued v2.1 column before writing the v3 shards. "
            "Example: action:6:0=1,1=-1"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    if output_root == input_root or output_root in input_root.parents:
        raise ValueError("Output directory must not be the input directory or an ancestor of it.")
    vector_element_remaps = _parse_vector_element_remaps(args.vector_element_remap)
    build_root = output_root
    if output_root.exists() and not args.overwrite:
        if not lerobot_v21.is_v3_dataset_root(output_root):
            raise FileExistsError(f"Output already exists: {output_root}")
        summary = {
            "input_root": str(input_root),
            "output_root": str(output_root),
            "mode": "norm_stats_only" if args.compute_norm_stats else "existing_output",
        }
        if args.compute_norm_stats:
            summary["norm_stats"] = lerobot_v21.write_output_norm_stats(
                output_root,
                run_compute_stats=True,
            )
        print(json.dumps(summary, indent=2))
        return
    temp_dir = ray_runtime.resolve_ray_temp_dir(
        label="v21-v30",
        unique_key=str(build_root),
        requested_root=args.ray_temp_root,
    )
    ray_runtime.configure_process_temp_dir(temp_dir)
    bundle = lerobot_v21.load_v21_dataset_bundle(input_root)
    if vector_element_remaps:
        bundle = _with_vector_remaps(bundle, vector_element_remaps)
    finalize_total = lerobot_v21.estimate_convert_finalize_steps(bundle)
    with progress_display.create_progress(console=CONSOLE) as progress:
        finalize_task_id = progress.add_task(
            "[green]Finalizing LeRobot v2.1 -> v3.0",
            total=max(1, finalize_total),
        )
        summary = lerobot_v21.convert_dataset(
            input_root,
            build_root,
            link_mode="copy",
            overwrite=args.overwrite,
            conversion_num_workers=args.conversion_num_workers,
            bundle=bundle,
            finalize_progress=lambda completed, total: progress.update(
                finalize_task_id,
                completed=completed,
                total=max(1, total),
            ),
        )
    if args.compute_norm_stats:
        summary["norm_stats"] = lerobot_v21.write_output_norm_stats(
            build_root,
            run_compute_stats=True,
        )
    if args.cleanup_tmp_on_success:
        ray_runtime.cleanup_temp_paths([temp_dir])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
