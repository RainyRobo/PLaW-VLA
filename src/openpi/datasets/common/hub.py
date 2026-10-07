# Copyright 2026 PLaW-VLA authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any

from lerobot.datasets.lerobot_dataset import LeRobotDataset

_MAX_HF_REPO_ID_LENGTH = 96


def sanitize_repo_fragment(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9._-]+", "-", value.strip().lower())
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-.")
    return normalized or "dataset"


def build_repo_id(owner: str, prefix: str | None, name: str) -> str:
    owner = owner.strip()
    if not owner:
        raise ValueError("hub_owner must be non-empty.")

    fragments = []
    if prefix:
        fragments.append(sanitize_repo_fragment(prefix))
    fragments.append(sanitize_repo_fragment(name))
    repo_name = "-".join(fragment for fragment in fragments if fragment)

    max_repo_name_length = _MAX_HF_REPO_ID_LENGTH - len(owner) - 1
    if len(repo_name) > max_repo_name_length:
        digest = hashlib.sha1(repo_name.encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
        keep = max(1, max_repo_name_length - len(digest) - 1)
        repo_name = f"{repo_name[:keep].rstrip('-.')}-{digest}"

    return f"{owner}/{repo_name}"


def push_dataset_root_to_hub(
    dataset_root: Path,
    repo_id: str,
    *,
    branch: str | None = None,
    tags: tuple[str, ...] = (),
    license: str | None = None,
    tag_version: bool = True,
    push_videos: bool = True,
    private: bool = False,
    upload_large_folder: bool = False,
    card_kwargs: dict[str, Any] | None = None,
) -> str:
    dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root, download_videos=False)
    push_kwargs: dict[str, Any] = {
        "branch": branch,
        "tags": list(tags) if tags else None,
        "tag_version": tag_version,
        "push_videos": push_videos,
        "private": private,
        "upload_large_folder": upload_large_folder,
    }
    if license is not None:
        push_kwargs["license"] = license
    if card_kwargs:
        push_kwargs.update(card_kwargs)
    dataset.push_to_hub(**push_kwargs)
    return repo_id
