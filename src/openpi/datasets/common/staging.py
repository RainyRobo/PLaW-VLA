# Copyright 2026 PLaW-VLA authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import tempfile

NETWORK_FILESYSTEM_TYPES = {
    "9p",
    "afs",
    "ceph",
    "cifs",
    "fuse.gcsfuse",
    "fuse.sshfs",
    "gcsfuse",
    "lustre",
    "nfs",
    "nfs4",
    "smb3",
    "sshfs",
}


def _nearest_existing_path(path: Path) -> Path:
    candidate = path.expanduser().resolve(strict=False)
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def detect_filesystem_type(path: Path) -> str | None:
    candidate = _nearest_existing_path(path)
    candidate_str = candidate.as_posix()

    mounts: list[tuple[str, str]] = []
    try:
        with Path("/proc/mounts").open("r", encoding="utf-8") as f:
            for raw_line in f:
                parts = raw_line.split()
                if len(parts) < 3:
                    continue
                mount_point = parts[1].replace("\\040", " ")
                mounts.append((mount_point.rstrip("/") or "/", parts[2]))
    except OSError:
        return None

    mounts.sort(key=lambda item: len(item[0]), reverse=True)
    for mount_point, fs_type in mounts:
        if mount_point == "/" or candidate_str == mount_point or candidate_str.startswith(f"{mount_point}/"):
            return fs_type
    return None


def is_network_filesystem(path: Path) -> bool:
    fs_type = detect_filesystem_type(path)
    if fs_type is None:
        return False
    return fs_type in NETWORK_FILESYSTEM_TYPES


def default_staging_base() -> Path:
    env_root = os.environ.get("LOCAL_STAGING_ROOT")
    candidates = [Path(env_root).expanduser() if env_root else None, Path(tempfile.gettempdir())]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            candidate.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        return candidate.resolve()
    raise RuntimeError("No usable local staging directory found.")


def resolve_staging_root(
    output_root: Path,
    *,
    label: str,
    requested_root: Path | None = None,
) -> Path | None:
    if requested_root is None and not is_network_filesystem(output_root):
        return None

    base = requested_root.expanduser().resolve() if requested_root is not None else default_staging_base()
    base.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(str(output_root).encode("utf-8"), usedforsecurity=False).hexdigest()[:10]
    return (base / f"dataset-stage-{label}-{digest}").resolve()


def remove_tree(path: Path) -> None:
    if not path.exists():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def publish_tree(source_root: Path, target_root: Path, *, overwrite: bool) -> None:
    source_root = source_root.expanduser().resolve()
    target_root = target_root.expanduser().resolve()
    if source_root == target_root:
        return
    if source_root.is_relative_to(target_root) or target_root.is_relative_to(source_root):
        raise ValueError("Staging source and output directories must not contain one another.")

    if target_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {target_root}")
        remove_tree(target_root)

    target_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_root, target_root, symlinks=True)
