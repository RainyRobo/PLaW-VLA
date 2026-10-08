#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PLaW-VLA authors.
"""Summarize the episode checkpoint written by the LIBERO-Plus client.

Pass the client result directory or its ``checkpoint.json`` file. This script
uses the Python standard library and can summarize an evaluation while it runs.

Example:
    python examples/libero_plus/aggregate.py results/libero_plus --counts --save
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

PERTURBATION_CATEGORIES = {
    "Camera Viewpoints": "Camera",
    "Robot Initial States": "Robot",
    "Language Instructions": "Language",
    "Light Conditions": "Light",
    "Background Textures": "Background",
    "Sensor Noise": "Noise",
    "Objects Layout": "Layout",
}


def _load_records(checkpoint: pathlib.Path) -> list[dict[str, Any]]:
    data = json.loads(checkpoint.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("records"), list):
        raise ValueError(f"{checkpoint}: expected an object with a records list")
    for index, record in enumerate(data["records"]):
        if not isinstance(record, dict):
            raise ValueError(f"{checkpoint}: record {index} must be an object")
        if not isinstance(record.get("suite"), str) or not record["suite"]:
            raise ValueError(f"{checkpoint}: record {index} must have a suite name")
        if not isinstance(record.get("success"), bool):
            raise ValueError(f"{checkpoint}: record {index} must have a boolean success value")
        if record.get("category") is not None and not isinstance(record["category"], str):
            raise ValueError(f"{checkpoint}: record {index} category must be a string or null")
    return data["records"]


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    columns = [*PERTURBATION_CATEGORIES.values(), "Total"]
    per_suite: dict[str, dict[str, dict[str, int]]] = {}
    pooled = {column: {"success": 0, "total": 0} for column in columns}
    unclassified = 0
    for record in records:
        suite = record["suite"]
        counts = per_suite.setdefault(suite, {column: {"success": 0, "total": 0} for column in columns})
        category = PERTURBATION_CATEGORIES.get(record.get("category"))
        record_columns = ["Total"]
        if category is None:
            unclassified += 1
        else:
            record_columns.append(category)
        for column in record_columns:
            counts[column]["success"] += int(record["success"])
            counts[column]["total"] += 1
            pooled[column]["success"] += int(record["success"])
            pooled[column]["total"] += 1
    return {"per_suite": per_suite, "pooled": pooled, "unclassified_episodes": unclassified}


def _format_table(checkpoint: pathlib.Path, summary: dict[str, Any], *, show_counts: bool) -> str:
    columns = [*PERTURBATION_CATEGORIES.values(), "Total"]
    label_width = max([18, *(len(suite) for suite in summary["per_suite"])])
    cell_width = 12
    header = f"{'Suite':<{label_width}}" + "".join(f"{column:>{cell_width}}" for column in columns)
    lines = [f"LIBERO-Plus: {checkpoint}", header, "-" * len(header)]
    rows = [*sorted(summary["per_suite"].items()), ("Pooled", summary["pooled"])]
    for label, counts in rows:
        cells = []
        for column in columns:
            success, total = counts[column]["success"], counts[column]["total"]
            cells.append(f"{success / total * 100:.1f}%" if total else "N/A")
        lines.append(f"{label:<{label_width}}" + "".join(f"{cell:>{cell_width}}" for cell in cells))
        if show_counts:
            cells = [f"{counts[column]['success']}/{counts[column]['total']}" for column in columns]
            lines.append(f"{'  success/total':<{label_width}}" + "".join(f"{cell:>{cell_width}}" for cell in cells))
    if summary["unclassified_episodes"]:
        lines.append(f"Unclassified episodes included in Total: {summary['unclassified_episodes']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize the LIBERO-Plus client's episode checkpoint.")
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="results/libero_plus/checkpoint.json",
        type=pathlib.Path,
        help="Checkpoint JSON file or result directory (default: results/libero_plus/checkpoint.json).",
    )
    parser.add_argument("--counts", action="store_true", help="Include success/total counts in the table.")
    parser.add_argument(
        "--save", action="store_true", help="Save aggregate.txt and aggregate.json beside the checkpoint."
    )
    args = parser.parse_args(argv)
    checkpoint = args.checkpoint.expanduser().resolve()
    if checkpoint.is_dir():
        checkpoint /= "checkpoint.json"
    try:
        summary = _aggregate(_load_records(checkpoint))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    table = _format_table(checkpoint, summary, show_counts=args.counts)
    print(table)
    if args.save:
        (checkpoint.parent / "aggregate.txt").write_text(table + "\n", encoding="utf-8")
        (checkpoint.parent / "aggregate.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
