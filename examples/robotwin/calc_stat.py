# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PLaW-VLA authors.
"""Summarize RoboTwin client metrics by task.

Pass one or more ``--save-root`` directories from ``launch_client.sh`` or the
run directory printed by ``eval.sh``. Only the client's current
``stseed-*/metrics/<task>/res.json`` files are read; video files are not used.

Example:
    python examples/robotwin/calc_stat.py results/robotwin/eval_run
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# Task groups: 1 = single arm, 2 = bimanual, 3 = bimanual with sequencing.
TASK_CLASS: dict[str, int] = {
    "adjust_bottle": 1,
    "beat_block_hammer": 1,
    "blocks_ranking_rgb": 3,
    "blocks_ranking_size": 3,
    "click_alarmclock": 1,
    "click_bell": 1,
    "dump_bin_bigbin": 1,
    "grab_roller": 1,
    "handover_block": 2,
    "handover_mic": 2,
    "hanging_mug": 2,
    "lift_pot": 1,
    "move_can_pot": 1,
    "move_pillbottle_pad": 1,
    "move_playingcard_away": 1,
    "move_stapler_pad": 1,
    "open_laptop": 1,
    "open_microwave": 1,
    "pick_diverse_bottles": 2,
    "pick_dual_bottles": 2,
    "place_a2b_left": 1,
    "place_a2b_right": 1,
    "place_bread_basket": 1,
    "place_bread_skillet": 2,
    "place_burger_fries": 2,
    "place_can_basket": 2,
    "place_cans_plasticbox": 2,
    "place_container_plate": 1,
    "place_dual_shoes": 2,
    "place_empty_cup": 1,
    "place_fan": 1,
    "place_mouse_pad": 1,
    "place_object_basket": 2,
    "place_object_scale": 1,
    "place_object_stand": 1,
    "place_phone_stand": 1,
    "place_shoe": 1,
    "press_stapler": 1,
    "put_bottles_dustbin": 3,
    "put_object_cabinet": 2,
    "rotate_qrcode": 1,
    "scan_object": 2,
    "shake_bottle_horizontally": 1,
    "shake_bottle": 1,
    "stack_blocks_three": 3,
    "stack_blocks_two": 2,
    "stack_bowls_three": 3,
    "stack_bowls_two": 2,
    "stamp_seal": 1,
    "turn_switch": 1,
}


def _read_counts(path: Path) -> tuple[int, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a metrics object")
    counts = []
    for name in ("succ_num", "total_num"):
        value = data.get(name)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{path}: {name} must be a nonnegative integer")
        if not math.isfinite(value) or value < 0 or int(value) != value:
            raise ValueError(f"{path}: {name} must be a nonnegative integer")
        counts.append(int(value))
    success, total = counts
    if success > total:
        raise ValueError(f"{path}: succ_num cannot exceed total_num")
    return success, total


def compute_success_rates(roots: list[Path]) -> list[tuple[str, int, int, int, float | None]]:
    """Pool metrics across the supplied roots, counting each metrics file once."""
    task_counts: dict[str, tuple[int, int]] = {}
    seen: set[Path] = set()
    for root_arg in roots:
        root = root_arg.expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"Result root is not a directory: {root}")
        metric_files = sorted(root.glob("**/stseed-*/metrics/*/res.json"))
        if not metric_files:
            raise ValueError(f"No stseed-*/metrics/<task>/res.json files under {root}")
        for path in metric_files:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            success, total = _read_counts(path)
            task = path.parent.name
            previous_success, previous_total = task_counts.get(task, (0, 0))
            task_counts[task] = (previous_success + success, previous_total + total)
    return [
        (task, success, total - success, total, success / total if total else None)
        for task, (success, total) in sorted(task_counts.items())
    ]


def _mean_rate(results: list[tuple[str, int, int, int, float | None]]) -> float | None:
    rates = [result[4] for result in results if result[4] is not None]
    return sum(rates) / len(rates) if rates else None


def print_table(results: list[tuple[str, int, int, int, float | None]]) -> None:
    print(f"{'task':30s} {'succ':>6s} {'fail':>6s} {'total':>6s} {'SuccessRate':>12s} {'Class':>6s}")
    print("-" * 90)
    for task, success, failure, total, rate in results:
        rate_str = f"{rate * 100:.2f}%" if rate is not None else "N/A"
        group = str(TASK_CLASS.get(task, "N/A"))
        print(f"{task:30s} {success:6d} {failure:6d} {total:6d} {rate_str:>12s} {group:>6s}")
    print("-" * 90)
    groups = [("MEAN (ALL TASKS)", results)]
    groups.extend(
        (f"MEAN (CLASS {group})", [result for result in results if TASK_CLASS.get(result[0]) == group])
        for group in (1, 2, 3)
    )
    unknown = [result for result in results if result[0] not in TASK_CLASS]
    if unknown:
        groups.append(("MEAN (UNKNOWN)", unknown))
    for label, subset in groups:
        rate = _mean_rate(subset)
        rate_str = f"{rate * 100:.2f}%" if rate is not None else "N/A"
        print(f"{label:30s} {'':6s} {'':6s} {'':6s} {rate_str:>12s}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize current RoboTwin client metrics by task.")
    parser.add_argument("save_roots", nargs="+", type=Path, help="Client --save-root or eval.sh run directories.")
    args = parser.parse_args(argv)
    try:
        results = compute_success_rates(args.save_roots)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
