#!/usr/bin/env python3
"""Extract InternData-A1 task-level LeRobot v3 tar shards into local staging roots."""

from __future__ import annotations

from concurrent import futures
from collections.abc import Callable
import dataclasses
import json
from pathlib import Path
from pathlib import PurePosixPath
import shutil
import sys
import tarfile
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


_ensure_import_paths()

import extract_interndata_a1 as extract_v21  # noqa: E402

from openpi.datasets.common import progress as progress_display  # noqa: E402
from openpi.datasets.specs import intern_a1 as common  # noqa: E402
from openpi.datasets.specs import intern_a1_v30 as common_v30  # noqa: E402

CONSOLE = progress_display.get_console()

DEFAULT_SOURCE_ROOT = Path("data/raw/intern_a1/sim_updated_lerobotv30")
DEFAULT_OUTPUT_ROOT = Path("data/raw/intern_a1/lerobot_v30_raw")

_PROGRESS_UPDATE_EVERY = extract_v21._PROGRESS_UPDATE_EVERY
_ArchiveMetadata = extract_v21._ArchiveMetadata
_ExtractionProgressReporter = extract_v21._ExtractionProgressReporter
_archive_result = extract_v21._archive_result
_build_archive_metadata = extract_v21._build_archive_metadata
_member_relative_path = extract_v21._member_relative_path
_resolve_num_workers = extract_v21._resolve_num_workers
_safe_output_path = extract_v21._safe_output_path
_target_root_suffix = extract_v21._target_root_suffix
_write_archive_metadata_cache = extract_v21._write_archive_metadata_cache


@dataclasses.dataclass(frozen=True)
class Args:
    output_dir: Path = DEFAULT_OUTPUT_ROOT
    source_root: Path = DEFAULT_SOURCE_ROOT
    embodiments: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    limit: int | None = None
    num_workers: int | None = None
    overwrite: bool = False
    summary_path: Path | None = None


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
        candidate_root = (output_root / category / embodiment / Path(*target_suffix.parts)).resolve()
        if not candidate_root.is_relative_to(output_root.resolve()):
            raise ValueError(f"Archive dataset path must remain within {output_root}.")
        target_root = candidate_root

        info_path = target_root / "meta" / "info.json"
        if info_path.exists():
            if common_v30.is_task_level_v30_dataset_dir(target_root) and not overwrite:
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

    if not common_v30.is_task_level_v30_dataset_dir(target_root):
        return _failed_result(
            "FileNotFoundError: "
            f"Extracted dataset is incomplete for {archive_path}; expected tasks.parquet, "
            f"meta/episodes chunk parquet files, and data chunk parquet files under {target_root}."
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
        raise FileNotFoundError(f"No InternData-A1 v3 archives found under {args.source_root}.")

    output_root = args.output_dir.expanduser().resolve()
    source_root = args.source_root.expanduser().resolve()
    if output_root == source_root or output_root in source_root.parents:
        raise ValueError("Output directory must not be the input directory or an ancestor of it.")
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
