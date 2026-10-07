#!/usr/bin/env python3
"""Extract and normalize InternData-A1 simulation tar shards into a raw local staging tree."""

from __future__ import annotations

from concurrent import futures
from collections.abc import Callable
import dataclasses
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import shutil
import sys
import tarfile
import time
import threading
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


def _ensure_import_paths() -> None:
    for path in (THIS_DIR, REPO_ROOT, SRC_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _load_common():
    _ensure_import_paths()
    from openpi.datasets.common import progress as progress_display
    from openpi.datasets.specs import intern_a1 as common

    return common, progress_display


common, progress_display = _load_common()
CONSOLE = progress_display.get_console()


@dataclasses.dataclass(frozen=True)
class Args:
    output_dir: Path
    source_root: Path = common.DEFAULT_SOURCE_ROOT
    embodiments: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    limit: int | None = None
    num_workers: int | None = None
    overwrite: bool = False
    summary_path: Path | None = None


_PROGRESS_UPDATE_EVERY = 32


def _available_cpu_count() -> int:
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


def _resolve_num_workers(requested: int | None, job_count: int) -> int:
    if job_count <= 0:
        return 1

    available_cpus = _available_cpu_count()
    if requested is None:
        requested = min(job_count, max(8, min(32, available_cpus)))
    if requested < 1:
        raise ValueError("num_workers must be at least 1 when provided.")
    return max(1, min(requested, available_cpus, job_count))


@dataclasses.dataclass(frozen=True)
class _ArchiveMetadata:
    archive_path: str
    archive_size: int
    archive_mtime_ns: int
    dataset_root: str
    extractable_files: int | None = None
    extractable_bytes: int | None = None


def _format_bytes(num_bytes: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(max(0, num_bytes))
    unit_index = 0
    while value >= 1024.0 and unit_index < len(units) - 1:
        value /= 1024.0
        unit_index += 1
    if unit_index == 0:
        return f"{int(value)}{units[unit_index]}"
    return f"{value:.1f}{units[unit_index]}"


def _format_rate(num_bytes: int, elapsed_s: float) -> str:
    if elapsed_s <= 0:
        return "0B/s"
    return f"{_format_bytes(int(num_bytes / elapsed_s))}/s"


def _archive_cache_dir(output_root: Path) -> Path:
    return output_root / ".archive_cache"


def _archive_cache_path(archive_path: Path, output_root: Path) -> Path:
    digest = hashlib.sha256(str(archive_path.resolve()).encode("utf-8")).hexdigest()
    return _archive_cache_dir(output_root) / f"{digest}.json"


def _load_archive_metadata_cache(archive_path: Path, output_root: Path) -> _ArchiveMetadata | None:
    cache_path = _archive_cache_path(archive_path, output_root)
    if not cache_path.exists():
        return None

    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    stat = archive_path.stat()
    if payload.get("archive_path") != str(archive_path.resolve()):
        return None
    if int(payload.get("archive_size", -1)) != stat.st_size:
        return None
    if int(payload.get("archive_mtime_ns", -1)) != stat.st_mtime_ns:
        return None

    try:
        dataset_root = str(payload["dataset_root"])
    except KeyError:
        return None

    return _ArchiveMetadata(
        archive_path=str(archive_path.resolve()),
        archive_size=stat.st_size,
        archive_mtime_ns=stat.st_mtime_ns,
        dataset_root=dataset_root,
        extractable_files=int(payload["extractable_files"]) if payload.get("extractable_files") is not None else None,
        extractable_bytes=int(payload["extractable_bytes"]) if payload.get("extractable_bytes") is not None else None,
    )


def _write_archive_metadata_cache(output_root: Path, metadata: _ArchiveMetadata) -> None:
    cache_path = _archive_cache_path(Path(metadata.archive_path), output_root)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dataclasses.asdict(metadata)
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_archive_metadata(archive_path: Path, output_root: Path) -> _ArchiveMetadata:
    cached = _load_archive_metadata_cache(archive_path, output_root)
    if cached is not None:
        return cached

    dataset_root = common.find_dataset_root_in_archive(archive_path)
    stat = archive_path.stat()
    return _ArchiveMetadata(
        archive_path=str(archive_path.resolve()),
        archive_size=stat.st_size,
        archive_mtime_ns=stat.st_mtime_ns,
        dataset_root=str(dataset_root),
    )


def _member_relative_path(member: tarfile.TarInfo, dataset_root: PurePosixPath) -> PurePosixPath | None:
    member_path = PurePosixPath(member.name)
    try:
        relative_path = member_path.relative_to(dataset_root)
    except ValueError:
        return None
    if not relative_path.parts:
        return None
    return relative_path


def _is_extractable_file(member: tarfile.TarInfo, dataset_root: PurePosixPath) -> bool:
    return _member_relative_path(member, dataset_root) is not None and member.isfile() and not (
        member.islnk() or member.issym()
    )


class _ExtractionProgressReporter:
    def __init__(self, progress, *, total_archives: int) -> None:
        self._progress = progress
        self._lock = threading.Lock()
        self._archives_task_id = progress.add_task("[cyan]Extracting InternData-A1 archives", total=total_archives)
        self._files_task_id = progress.add_task("[green]Extracted files", total=None)
        self._bytes_task_id = progress.add_task("[yellow]Extracted bytes", total=None)
        self._archive_task_ids: dict[str, int] = {}
        self._archive_totals: dict[str, tuple[int, int, float, str]] = {}

    def archive_started(
        self,
        archive_path: Path,
        target_root: Path,
        total_files: int | None,
        total_bytes: int | None,
    ) -> None:
        tail = Path(*target_root.parts[-3:]).as_posix()
        byte_summary = (
            f"{_format_bytes(0)}/{_format_bytes(total_bytes)}"
            if total_bytes is not None
            else _format_bytes(0)
        )
        description = f"[white]{archive_path.name}[/white] -> {tail} [dim]({byte_summary})[/dim]"
        with self._lock:
            task_id = self._progress.add_task(description, total=max(1, total_files) if total_files is not None else None)
            self._archive_task_ids[str(archive_path)] = task_id
            self._archive_totals[str(archive_path)] = (0, total_bytes or 0, time.perf_counter(), tail)
            if total_bytes is not None:
                bytes_task = self._progress.tasks[self._bytes_task_id]
                self._progress.update(self._bytes_task_id, total=(bytes_task.total or 0) + total_bytes)

    def files_advanced(self, archive_path: Path, file_advance: int, byte_advance: int) -> None:
        if file_advance <= 0 and byte_advance <= 0:
            return
        with self._lock:
            if file_advance > 0:
                self._progress.advance(self._files_task_id, file_advance)
            if byte_advance > 0:
                self._progress.advance(self._bytes_task_id, byte_advance)

            extracted_bytes, total_bytes, started_at, tail = self._archive_totals.get(
                str(archive_path),
                (0, 0, time.perf_counter(), archive_path.name),
            )
            extracted_bytes += byte_advance
            elapsed_s = max(0.0, time.perf_counter() - started_at)
            self._archive_totals[str(archive_path)] = (extracted_bytes, total_bytes, started_at, tail)
            task_id = self._archive_task_ids.get(str(archive_path))
            if task_id is not None:
                if file_advance > 0:
                    self._progress.advance(task_id, file_advance)
                self._progress.update(
                    task_id,
                    description=(
                        f"[white]{archive_path.name}[/white] -> {tail} "
                        f"[dim]({_format_bytes(extracted_bytes)}"
                        f"{'/' + _format_bytes(total_bytes) if total_bytes > 0 else ''}, "
                        f"{_format_rate(extracted_bytes, elapsed_s)})[/dim]"
                    ),
                )

    def archive_finished(self, archive_path: Path, status: str, extracted_files: int) -> None:
        with self._lock:
            self._progress.advance(self._archives_task_id, 1)
            task_id = self._archive_task_ids.pop(str(archive_path), None)
            extracted_bytes, total_bytes, started_at, _tail = self._archive_totals.pop(
                str(archive_path),
                (0, 0, time.perf_counter(), archive_path.name),
            )
            if task_id is not None:
                total = self._progress.tasks[task_id].total or max(1, extracted_files)
                elapsed_s = max(0.0, time.perf_counter() - started_at)
                self._progress.update(
                    task_id,
                    completed=total,
                    description=(
                        f"[green]{archive_path.name}[/green] ({status}, "
                        f"{_format_bytes(extracted_bytes)}/{_format_bytes(total_bytes)}, "
                        f"{_format_rate(extracted_bytes, elapsed_s)})"
                    ),
                    visible=False,
                )

    def archive_skipped(self, archive_path: Path) -> None:
        with self._lock:
            self._progress.advance(self._archives_task_id, 1)


def _archive_result(
    archive_path: Path,
    *,
    category: str,
    embodiment: str,
    dataset_name: str,
    output_dir: Path | None,
    status: str,
    num_files: int,
    error: str | None = None,
) -> dict[str, str | int]:
    result: dict[str, str | int] = {
        "archive": str(archive_path),
        "category": category,
        "embodiment": embodiment,
        "dataset_name": dataset_name,
        "output_dir": str(output_dir) if output_dir is not None else "",
        "status": status,
        "num_files": num_files,
    }
    if error is not None:
        result["error"] = error
    return result


def _safe_output_path(root: Path, relative_path: PurePosixPath) -> Path:
    destination = root.joinpath(*relative_path.parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    resolved_root = root.resolve()
    resolved_destination = destination.resolve()
    if not resolved_destination.is_relative_to(resolved_root):
        raise ValueError(f"Refusing to write outside {root}: {relative_path}")
    return destination


def _target_root_suffix(dataset_root: PurePosixPath, *, category: str, embodiment: str, archive_path: Path) -> PurePosixPath:
    parts = list(dataset_root.parts)
    try:
        category_index = parts.index(category)
        embodiment_index = parts.index(embodiment, category_index + 1)
    except ValueError:
        archive_stem = archive_path.name.removesuffix(".tar.gz")
        return PurePosixPath(archive_stem, dataset_root.name)

    suffix_parts = parts[embodiment_index + 1 :]
    if not suffix_parts:
        archive_stem = archive_path.name.removesuffix(".tar.gz")
        return PurePosixPath(archive_stem)
    return PurePosixPath(*suffix_parts)


def _extract_archive(
    archive_path: Path,
    output_root: Path,
    *,
    overwrite: bool,
    on_archive_started: Callable[[Path, Path, int | None, int | None], None] | None = None,
    on_archive_skipped: Callable[[Path], None] | None = None,
    on_files_advanced: Callable[[Path, int, int], None] | None = None,
    on_archive_finished: Callable[[Path, str, int], None] | None = None,
) -> dict[str, str | int]:
    category = archive_path.parent.parent.name
    embodiment = archive_path.parent.name
    dataset_name = archive_path.name.removesuffix(".tar.gz")
    target_root: Path | None = None
    extracted_files = 0

    def _failed_result(error: str) -> dict[str, str | int]:
        if target_root is not None:
            shutil.rmtree(target_root, ignore_errors=True)
        if on_archive_finished is not None:
            on_archive_finished(archive_path, "failed", extracted_files)
        return _archive_result(
            archive_path,
            category=category,
            embodiment=embodiment,
            dataset_name=dataset_name,
            output_dir=target_root,
            status="failed",
            num_files=extracted_files,
            error=error,
        )

    try:
        metadata = _build_archive_metadata(archive_path, output_root)
        dataset_root = PurePosixPath(metadata.dataset_root)
        dataset_name = dataset_root.name
        target_suffix = _target_root_suffix(
            dataset_root,
            category=category,
            embodiment=embodiment,
            archive_path=archive_path,
        )
        target_root = output_root / category / embodiment / Path(*target_suffix.parts)

        info_path = target_root / "meta" / "info.json"
        if info_path.exists():
            if common.is_complete_dataset_dir(target_root) and not overwrite:
                if on_archive_skipped is not None:
                    on_archive_skipped(archive_path)
                return _archive_result(
                    archive_path,
                    category=category,
                    embodiment=embodiment,
                    dataset_name=dataset_name,
                    output_dir=target_root,
                    status="skipped",
                    num_files=0,
                )
            shutil.rmtree(target_root)

        target_root.mkdir(parents=True, exist_ok=True)
        scanned_extractable_files = metadata.extractable_files
        scanned_extractable_bytes = metadata.extractable_bytes

        with tarfile.open(archive_path, "r:gz") as archive:
            if on_archive_started is not None:
                on_archive_started(archive_path, target_root, scanned_extractable_files, scanned_extractable_bytes)

            pending_progress = 0
            pending_bytes = 0
            computed_files = 0
            computed_bytes = 0
            for member in archive:
                relative_path = _member_relative_path(member, dataset_root)
                if relative_path is None:
                    continue
                if member.islnk() or member.issym():
                    continue

                destination = _safe_output_path(target_root, relative_path)
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    continue

                with archive.extractfile(member) as src:
                    if src is None:
                        continue
                    with destination.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                extracted_files += 1
                computed_files += 1
                computed_bytes += member.size
                pending_progress += 1
                pending_bytes += member.size
                if on_files_advanced is not None and pending_progress >= _PROGRESS_UPDATE_EVERY:
                    on_files_advanced(archive_path, pending_progress, pending_bytes)
                    pending_progress = 0
                    pending_bytes = 0

            if on_files_advanced is not None and pending_progress > 0:
                on_files_advanced(archive_path, pending_progress, pending_bytes)

            if scanned_extractable_files is None or scanned_extractable_bytes is None:
                _write_archive_metadata_cache(
                    output_root,
                    _ArchiveMetadata(
                        archive_path=metadata.archive_path,
                        archive_size=metadata.archive_size,
                        archive_mtime_ns=metadata.archive_mtime_ns,
                        dataset_root=metadata.dataset_root,
                        extractable_files=computed_files,
                        extractable_bytes=computed_bytes,
                    ),
                )
    except Exception as exc:
        return _failed_result(f"{type(exc).__name__}: {exc}")

    if target_root is None:
        return _failed_result("RuntimeError: target_root was not initialized.")

    if not common.is_complete_dataset_dir(target_root):
        return _failed_result(
            "FileNotFoundError: "
            f"Extracted dataset is incomplete for {archive_path}; expected tasks.jsonl and episode parquet files "
            f"under {target_root}."
        )

    if on_archive_finished is not None:
        on_archive_finished(archive_path, "extracted", extracted_files)

    return _archive_result(
        archive_path,
        category=category,
        embodiment=embodiment,
        dataset_name=dataset_name,
        output_dir=target_root,
        status="extracted",
        num_files=extracted_files,
    )


def main(args: Args) -> None:
    archives = common.discover_archives(
        args.source_root,
        embodiments=args.embodiments,
        categories=args.categories,
    )
    if args.limit is not None:
        archives = archives[: args.limit]

    if not archives:
        raise FileNotFoundError(f"No InternData-A1 archives found under {args.source_root}.")

    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    workers = _resolve_num_workers(args.num_workers, len(archives))
    summary: list[dict[str, str | int]] = []
    with progress_display.create_progress(console=CONSOLE) as progress:
        reporter = _ExtractionProgressReporter(progress, total_archives=len(archives))

        if workers <= 1 or len(archives) == 1:
            for archive in archives:
                summary.append(
                    _extract_archive(
                        archive,
                        output_root,
                        overwrite=args.overwrite,
                        on_archive_started=reporter.archive_started,
                        on_archive_skipped=reporter.archive_skipped,
                        on_files_advanced=reporter.files_advanced,
                        on_archive_finished=reporter.archive_finished,
                    )
                )
        else:
            with futures.ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_archive = {
                    executor.submit(
                        _extract_archive,
                        archive,
                        output_root,
                        overwrite=args.overwrite,
                        on_archive_started=reporter.archive_started,
                        on_archive_skipped=reporter.archive_skipped,
                        on_files_advanced=reporter.files_advanced,
                        on_archive_finished=reporter.archive_finished,
                    ): archive
                    for archive in archives
                }
                for future in futures.as_completed(future_to_archive):
                    summary.append(future.result())

            summary.sort(key=lambda item: str(item["archive"]))

    extracted = sum(1 for item in summary if item["status"] == "extracted")
    skipped = sum(1 for item in summary if item["status"] == "skipped")
    failed = sum(1 for item in summary if item["status"] == "failed")
    print(f"Processed {len(summary)} archives: extracted={extracted}, skipped={skipped}, failed={failed}")
    if failed:
        CONSOLE.print("[yellow]Some archives failed and were skipped. See the extraction summary for details.[/yellow]")

    summary_path = args.summary_path or output_root / "extraction_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote extraction summary to {summary_path}")

if __name__ == "__main__":
    main(tyro.cli(Args))
