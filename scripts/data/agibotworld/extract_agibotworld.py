# Copyright 2026 PLaW-VLA authors. SPDX-License-Identifier: Apache-2.0
"""Extract downloaded AgiBot tar shards into the converter's raw directory layout."""

from __future__ import annotations

import dataclasses
from pathlib import Path
import shutil
import tarfile
import tempfile

import tyro


@dataclasses.dataclass(frozen=True)
class Args:
    input_root: Path
    output_dir: Path | None = None
    task_ids: tuple[str, ...] = ()
    overwrite: bool = False


def _destination_parts(archive: Path, input_root: Path, member_name: str) -> tuple[str, ...]:
    parts = tuple(part for part in Path(member_name).parts if part != ".")
    if not parts or Path(member_name).is_absolute() or ".." in parts:
        raise ValueError(f"Invalid archive member path: {member_name!r}")
    if parts[0] == input_root.name:
        parts = parts[1:]
    if not parts:
        raise ValueError(f"Archive member has no data path: {member_name!r}")
    relative_archive = archive.relative_to(input_root)
    group = relative_archive.parts[0]
    if group not in ("observations", "proprio_stats", "parameters"):
        raise ValueError(f"Place task tar shards under observations/, proprio_stats/, or parameters/: {archive}")
    # Shards may retain a dataset-directory prefix before the documented group.
    if group in parts:
        parts = parts[parts.index(group):]
    if parts[0] == group:
        return parts
    if group == "observations":
        if len(relative_archive.parts) < 3:
            raise ValueError(f"Observation shards must be stored under observations/<task-id>/: {archive}")
        task_id = relative_archive.parts[1]
        has_task_prefix = parts[0] == task_id and len(parts) > 1 and parts[1].isdigit()
        return (group, *parts) if has_task_prefix else (group, task_id, *parts)
    return (group, *parts)


def main(args: Args) -> None:
    source = args.input_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve() if args.output_dir is not None else source
    if not source.is_dir():
        raise FileNotFoundError(f"Downloaded raw dataset root does not exist: {source}")
    selected_tasks = set(args.task_ids)
    if any(not task.isdigit() for task in selected_tasks):
        raise ValueError("Task IDs must be non-negative integers.")
    archives = sorted(
        path for group in ("observations", "proprio_stats", "parameters")
        for path in (source / group).rglob("*")
        if path.is_file() and path.name.endswith((".tar", ".tar.gz", ".tgz"))
    )
    if not archives:
        raise FileNotFoundError(f"No AgiBot tar shards found under {source}")
    output.mkdir(parents=True, exist_ok=True)
    archive_paths = set(archives)
    written = skipped = 0
    for archive in archives:
        relative = archive.relative_to(source).parts
        if selected_tasks and relative[0] == "observations" and relative[1] not in selected_tasks:
            continue
        with tarfile.open(archive, "r|*") as stream:
            for member in stream:
                if member.isdir() or member.name in (".", "./"):
                    continue
                if not member.isfile():
                    raise ValueError(f"Only regular raw-data files can be extracted: {archive}: {member.name}")
                parts = _destination_parts(archive, source, member.name)
                if selected_tasks and (len(parts) < 2 or parts[1] not in selected_tasks):
                    continue
                target = output.joinpath(*parts)
                if not target.resolve().is_relative_to(output) or target.resolve() in archive_paths:
                    raise ValueError(f"Archive member escapes its output or overwrites a source archive: {member.name}")
                if target.exists() and not args.overwrite:
                    if target.is_file() and target.stat().st_size == member.size:
                        skipped += 1
                        continue
                    raise FileExistsError(f"Existing output differs from the archive size: {target}; use --overwrite.")
                target.parent.mkdir(parents=True, exist_ok=True)
                extracted = stream.extractfile(member)
                if extracted is None:
                    raise ValueError(f"Unreadable archive member: {archive}: {member.name}")
                with extracted, tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
                    temporary_path = Path(temporary.name)
                    try:
                        shutil.copyfileobj(extracted, temporary)
                    except BaseException:
                        temporary_path.unlink(missing_ok=True)
                        raise
                try:
                    temporary_path.replace(target)
                finally:
                    temporary_path.unlink(missing_ok=True)
                written += 1
        print(f"Extracted {archive.relative_to(source)}")
    if output != source:
        for metadata in (source / "task_info").glob("task_*.json"):
            if selected_tasks and metadata.stem.removeprefix("task_") not in selected_tasks:
                continue
            target = output / "task_info" / metadata.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and not args.overwrite and target.read_bytes() != metadata.read_bytes():
                raise FileExistsError(f"Existing task metadata differs: {target}; use --overwrite.")
            shutil.copyfile(metadata, target)
    print(f"Prepared raw layout in {output}: {written} files extracted, {skipped} existing files reused")


if __name__ == "__main__":
    main(tyro.cli(Args))
