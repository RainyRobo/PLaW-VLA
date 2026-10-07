# Copyright 2026 PLaW-VLA authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

DEFAULT_INPUT_ROOTS = (
    Path("data/raw/robotwin/lerobot_robotwin_eef_aug_500"),
    Path("data/raw/robotwin/lerobot_robotwin_eef_clean_50"),
)
DEFAULT_OUTPUT_ROOT = Path("data/pretrain/robotwin")


@dataclass(frozen=True)
class RobotWinChildDataset:
    root: Path
    parent_root: Path
    embodiment: str


def _load_info_json(dataset_dir: Path) -> dict:
    info_path = dataset_dir / "meta" / "info.json"
    with info_path.open(encoding="utf-8") as f:
        return json.load(f)


def infer_embodiment(dataset_dir: Path) -> str:
    info = _load_info_json(dataset_dir)
    robot_type = str(info.get("robot_type") or "").strip().lower()
    robot_type = robot_type.replace("-", "_") or "unknown"

    state_shape = tuple(info.get("features", {}).get("observation.state", {}).get("shape") or ())
    action_shape = tuple(info.get("features", {}).get("action", {}).get("shape") or ())
    if not state_shape or not action_shape:
        raise ValueError(f"Missing RobotWin state/action shape metadata under {dataset_dir / 'meta' / 'info.json'}")

    state_dim = int(state_shape[-1])
    action_dim = int(action_shape[-1])
    if state_dim != action_dim:
        raise ValueError(f"State/action dimension mismatch for {dataset_dir}: {state_dim} != {action_dim}")

    return f"{robot_type}_eef{action_dim}"


def _is_lerobot_v21_dataset_dir(path: Path) -> bool:
    info_path = path / "meta" / "info.json"
    return info_path.is_file()


def discover_child_datasets(
    input_roots: tuple[Path, ...] = DEFAULT_INPUT_ROOTS,
    *,
    embodiments: tuple[str, ...] = (),
) -> dict[str, list[RobotWinChildDataset]]:
    requested = {item.strip().lower() for item in embodiments if item.strip()}
    grouped: dict[str, list[RobotWinChildDataset]] = {}

    for input_root in input_roots:
        resolved_root = input_root.expanduser().resolve()
        if not resolved_root.is_dir():
            raise FileNotFoundError(f"RobotWin input root does not exist: {resolved_root}")

        for child in sorted(resolved_root.iterdir(), key=lambda path: path.name):
            if not child.is_dir() or child.name.startswith("."):
                continue
            if not _is_lerobot_v21_dataset_dir(child):
                continue

            embodiment = infer_embodiment(child)
            if requested and embodiment not in requested:
                continue

            grouped.setdefault(embodiment, []).append(
                RobotWinChildDataset(
                    root=child.resolve(),
                    parent_root=resolved_root,
                    embodiment=embodiment,
                )
            )

    return {embodiment: datasets for embodiment, datasets in sorted(grouped.items()) if datasets}
