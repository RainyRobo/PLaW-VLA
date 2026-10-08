#!/usr/bin/env python3
"""Extract EgoDex episode metadata to JSONL for inspecting local source data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py


def decode_attr(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    return str(value).strip()


def parse_part(part_name: str) -> tuple[str, int]:
    if part_name.startswith("part") and part_name[4:].isdigit():
        return "train", int(part_name[4:])
    if part_name == "test":
        return "test", 0
    if part_name == "extra":
        return "extra", 0
    raise ValueError(part_name)


def resolve_data_root(data_dir: Path) -> Path:
    splits = ("part1", "part2", "part3", "part4", "part5", "test", "extra")
    if any((data_dir / d).is_dir() for d in splits):
        return data_dir
    if (data_dir / "v1").is_dir() and any((data_dir / "v1" / d).is_dir() for d in splits):
        return data_dir / "v1"
    return data_dir


def iter_pairs(data_root: Path):
    valid = {"part1", "part2", "part3", "part4", "part5", "test", "extra"}
    for split_dir in sorted(p for p in data_root.iterdir() if p.is_dir() and p.name in valid):
        split, part_id = parse_part(split_dir.name)
        for task_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            mp4_by_stem = {p.stem: p for p in task_dir.glob("*.mp4")}
            hdf5_by_stem = {p.stem: p for p in task_dir.glob("*.hdf5")}
            for stem in sorted(
                set(mp4_by_stem).intersection(hdf5_by_stem),
                key=lambda s: (0, int(s)) if s.isdigit() else (1, s),
            ):
                yield split, part_id, task_dir.name, mp4_by_stem[stem], hdf5_by_stem[stem]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract EgoDex metadata rows from raw data")
    parser.add_argument("--data_dir", "-d", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    data_root = resolve_data_root(args.data_dir.expanduser().resolve())

    rows = []
    for i, (split, part_id, task_name, mp4_path, hdf5_path) in enumerate(iter_pairs(data_root)):
        if args.limit is not None and i >= args.limit:
            break

        with h5py.File(hdf5_path, "r") as f:
            joints = sorted(f["transforms"].keys())
            prompt1 = decode_attr(f.attrs.get("llm_description"))
            prompt2 = decode_attr(f.attrs.get("llm_description2"))
            which = decode_attr(f.attrs.get("which_llm_description"))
            n_frames = int(f["transforms"][joints[0]].shape[0]) if joints else 0
            has_conf = "confidences" in f

        rows.append(
            {
                "split": split,
                "part_id": part_id,
                "task_name": task_name,
                "file_index": int(mp4_path.stem) if mp4_path.stem.isdigit() else -1,
                "mp4_path": str(mp4_path),
                "hdf5_path": str(hdf5_path),
                "n_frames": n_frames,
                "joint_count": len(joints),
                "has_confidences": has_conf,
                "llm_description": prompt1,
                "llm_description2": prompt2,
                "which_llm_description": which,
            }
        )

    payload = "\n".join(json.dumps(r) for r in rows)
    if payload:
        payload += "\n"

    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
        print(f"Wrote {len(rows)} metadata rows to {args.output}")


if __name__ == "__main__":
    main()
