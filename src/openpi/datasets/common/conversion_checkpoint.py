# Copyright 2026 PLaW-VLA authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class EpisodeCheckpointStore:
    def __init__(self, dataset_root: Path, *, namespace: str) -> None:
        meta_root = dataset_root / "meta"
        meta_root.mkdir(parents=True, exist_ok=True)
        self._completed_path = meta_root / f"{namespace}_completed.jsonl"
        self._failed_path = meta_root / f"{namespace}_failed.jsonl"

    @property
    def completed_path(self) -> Path:
        return self._completed_path

    @property
    def failed_path(self) -> Path:
        return self._failed_path

    def load_completed(self) -> set[str]:
        if not self._completed_path.exists():
            return set()
        return {
            str(record["key"])
            for record in self._iter_jsonl(self._completed_path)
            if "key" in record
        }

    def load_completed_records(self) -> list[dict[str, Any]]:
        if not self._completed_path.exists():
            return []
        return self._iter_jsonl(self._completed_path)

    def load_failed(self) -> list[dict[str, Any]]:
        if not self._failed_path.exists():
            return []
        return self._iter_jsonl(self._failed_path)

    def append_completed(self, key: str, *, episode_index: int | None = None) -> None:
        record: dict[str, Any] = {"key": str(key)}
        if episode_index is not None:
            record["episode_index"] = int(episode_index)
        self._append_jsonl(self._completed_path, record)

    def append_completed_many(self, records: list[dict[str, Any]]) -> None:
        normalized: list[dict[str, Any]] = []
        for record in records:
            normalized_record: dict[str, Any] = {"key": str(record["key"])}
            if "episode_index" in record and record["episode_index"] is not None:
                normalized_record["episode_index"] = int(record["episode_index"])
            normalized.append(normalized_record)
        self._append_jsonl_many(self._completed_path, normalized)

    def prune_completed_after_episode_index(self, episode_count: int) -> int:
        if not self._completed_path.exists():
            return 0
        records = self.load_completed_records()
        if not records:
            return 0

        kept_records: list[dict[str, Any]] = []
        fallback_counter = 0
        for record in records:
            record_episode_index = record.get("episode_index")
            if record_episode_index is None:
                keep = fallback_counter < episode_count
                fallback_counter += 1
            else:
                keep = int(record_episode_index) < episode_count
            if keep:
                kept_records.append(record)

        removed = len(records) - len(kept_records)
        if removed <= 0:
            return 0
        self._rewrite_jsonl(self._completed_path, kept_records)
        return removed

    def rewrite_completed_records(self, records: list[dict[str, Any]]) -> None:
        normalized: list[dict[str, Any]] = []
        for record in records:
            normalized_record: dict[str, Any] = {"key": str(record["key"])}
            if "episode_index" in record and record["episode_index"] is not None:
                normalized_record["episode_index"] = int(record["episode_index"])
            normalized.append(normalized_record)
        self._rewrite_jsonl(self._completed_path, normalized)

    def append_failure(self, key: str, error: str, **extra: Any) -> None:
        record: dict[str, Any] = {"key": str(key), "error": str(error)}
        if extra:
            record.update(extra)
        self._append_jsonl(self._failed_path, record)

    def append_failure_many(self, records: list[dict[str, Any]]) -> None:
        normalized: list[dict[str, Any]] = []
        for record in records:
            normalized_record: dict[str, Any] = {
                "key": str(record["key"]),
                "error": str(record["error"]),
            }
            for key, value in record.items():
                if key in {"key", "error"}:
                    continue
                normalized_record[key] = value
            normalized.append(normalized_record)
        self._append_jsonl_many(self._failed_path, normalized)

    def prune_failures_for_completed(self, completed_keys: set[str] | None = None) -> int:
        if not self._failed_path.exists():
            return 0
        if completed_keys is None:
            completed_keys = self.load_completed()
        if not completed_keys:
            return 0

        failed_records = self.load_failed()
        kept_records = [record for record in failed_records if str(record.get("key", "")) not in completed_keys]
        removed = len(failed_records) - len(kept_records)
        if removed <= 0:
            return 0
        self._rewrite_jsonl(self._failed_path, kept_records)
        return removed

    def _iter_jsonl(self, path: Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    def _append_jsonl(self, path: Path, record: dict[str, Any]) -> None:
        self._append_jsonl_many(path, [record])

    def _append_jsonl_many(self, path: Path, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        with path.open("a", encoding="utf-8") as f:
            for record in records:
                encoded = json.dumps(record, separators=(",", ":"))
                f.write(encoded)
                f.write("\n")
            f.flush()
            os.fsync(f.fileno())

    def _rewrite_jsonl(self, path: Path, records: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not records:
            if path.exists():
                path.unlink()
            return

        tmp_path = path.with_suffix(f"{path.suffix}.tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            for record in records:
                encoded = json.dumps(record, separators=(",", ":"))
                f.write(encoded)
                f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        tmp_path.replace(path)
