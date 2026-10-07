# Copyright 2026 PLaW-VLA authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from contextlib import suppress
import logging
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from lerobot.datasets.video_utils import concatenate_video_files as _lerobot_concatenate_video_files

LOGGER = logging.getLogger(__name__)


def _validation_enabled() -> bool:
    raw = os.environ.get("VALIDATE_CONCAT_VIDEOS", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _validation_tolerance_s() -> float:
    raw = os.environ.get("CONCAT_VALIDATION_TOLERANCE_S", "30").strip()
    try:
        value = float(raw)
    except ValueError:
        LOGGER.warning("Invalid CONCAT_VALIDATION_TOLERANCE_S=%r; defaulting to 30s.", raw)
        value = 30.0
    return max(1.0, value)


def _ffprobe_path() -> str:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise FileNotFoundError("ffprobe executable is not available.")
    return ffprobe


def _probe_video_duration_s(video_path: Path) -> float:
    command = [
        _ffprobe_path(),
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip() or "unknown ffprobe error"
        raise RuntimeError(f"ffprobe duration probe failed for {video_path}: {stderr}")
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"ffprobe returned invalid duration for {video_path}: {result.stdout!r}") from exc


def _probe_seek_first_packet_pts(video_path: Path, timestamp_s: float) -> float:
    command = [
        _ffprobe_path(),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "packet=pts_time",
        "-read_intervals",
        f"{timestamp_s}%+0.5",
        "-of",
        "csv=p=0",
        str(video_path),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip() or "unknown ffprobe error"
        raise RuntimeError(f"ffprobe seek probe failed for {video_path} at {timestamp_s:.3f}s: {stderr}")

    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        first_value = line.split(",", 1)[0].strip()
        try:
            return float(first_value)
        except ValueError:
            continue

    raise RuntimeError(f"ffprobe produced no packet timestamps for {video_path} at {timestamp_s:.3f}s.")


def _validate_concatenated_video_file(video_path: Path) -> None:
    if not _validation_enabled():
        return

    try:
        duration_s = _probe_video_duration_s(video_path)
        tolerance_s = _validation_tolerance_s()
        probe_points = {0.0}
        if duration_s > 1.0:
            probe_points.add(duration_s * 0.5)
            probe_points.add(max(0.0, duration_s * 0.95))
            probe_points.add(max(0.0, duration_s - 1.0))

        for requested_ts in sorted(probe_points):
            observed_ts = _probe_seek_first_packet_pts(video_path, requested_ts)
            if abs(observed_ts - requested_ts) > tolerance_s:
                raise RuntimeError(
                    f"Concatenated video validation failed for {video_path}: requested seek at {requested_ts:.3f}s "
                    f"returned packet at {observed_ts:.3f}s (tolerance {tolerance_s:.3f}s)."
                )
    except FileNotFoundError:
        LOGGER.warning("ffprobe is unavailable; skipping concat video validation for %s.", video_path)


def _run_ffmpeg_concat(
    input_video_paths: list[Path | str],
    output_video_path: Path,
    *,
    overwrite: bool = True,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg executable is not available.")
    if output_video_path.exists() and not overwrite:
        return

    concat_manifest_path: Path | None = None
    tmp_output_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ffconcat", delete=False, encoding="utf-8") as tmp_file:
            tmp_file.write("ffconcat version 1.0\n")
            for input_path in input_video_paths:
                resolved_input = Path(input_path).expanduser().resolve()
                escaped_input = resolved_input.as_posix().replace("'", "'\\''")
                tmp_file.write(f"file '{escaped_input}'\n")
            tmp_file.flush()
            concat_manifest_path = Path(tmp_file.name)

        with tempfile.NamedTemporaryFile(
            suffix=output_video_path.suffix or ".mp4", dir=output_video_path.parent, delete=False
        ) as tmp_named_file:
            tmp_output_path = Path(tmp_named_file.name)

        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-xerror",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_manifest_path),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(tmp_output_path),
        ]
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            stderr = result.stderr.strip() or result.stdout.strip() or "unknown ffmpeg error"
            raise RuntimeError(f"ffmpeg concat failed for {output_video_path}: {stderr}")

        _validate_concatenated_video_file(tmp_output_path)
        tmp_output_path.replace(output_video_path)
        tmp_output_path = None
    finally:
        if concat_manifest_path is not None:
            with suppress(FileNotFoundError):
                concat_manifest_path.unlink()
        if tmp_output_path is not None and tmp_output_path.exists():
            with suppress(FileNotFoundError):
                tmp_output_path.unlink()


def concatenate_video_files(
    input_video_paths: list[Path | str],
    output_video_path: Path,
    *,
    overwrite: bool = True,
) -> None:
    """Concatenate video shards with `ffmpeg`.

    We intentionally prefer the `ffmpeg` CLI over the PyAV packet-remux path
    used in LeRobot because large AV1/MP4 shards have proven susceptible to
    broken sample tables and seek offsets after PyAV concat/remux. If `ffmpeg`
    is unavailable, we fall back to LeRobot's upstream helper.
    """
    output_video_path = Path(output_video_path)
    if output_video_path.exists() and not overwrite:
        return

    if not input_video_paths:
        raise FileNotFoundError("No input video paths provided.")

    resolved_inputs = [Path(path).expanduser().resolve() for path in input_video_paths]
    for input_path in resolved_inputs:
        if not input_path.is_file():
            raise FileNotFoundError(f"Missing input video: {input_path}")
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    if len(resolved_inputs) == 1:
        with tempfile.TemporaryDirectory(prefix="video-concat-", dir=output_video_path.parent) as temp_dir:
            temp_output = Path(temp_dir) / output_video_path.name
            shutil.copy2(resolved_inputs[0], temp_output)
            _validate_concatenated_video_file(temp_output)
            temp_output.replace(output_video_path)
        return

    try:
        _run_ffmpeg_concat(resolved_inputs, output_video_path, overwrite=overwrite)
    except FileNotFoundError:
        LOGGER.warning("ffmpeg is unavailable; falling back to LeRobot's PyAV video concat helper.")
        with tempfile.TemporaryDirectory(prefix="video-concat-", dir=output_video_path.parent) as temp_dir:
            temp_output = Path(temp_dir) / output_video_path.name
            _lerobot_concatenate_video_files(resolved_inputs, temp_output, overwrite=True)
            _validate_concatenated_video_file(temp_output)
            temp_output.replace(output_video_path)
