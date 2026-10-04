import ast
from collections.abc import Iterator, Sequence
import dataclasses
import logging
import multiprocessing
import os
import pathlib
import time
import typing
from typing import Any, Literal, Protocol, SupportsIndex, TypeVar
import warnings

try:
    from av import error as av_error
except Exception:  # pragma: no cover - pyav is optional in some environments.
    av_error = None

from datasets.packaged_modules.parquet.parquet import Parquet as HFParquetBuilder
import jax
import jax.numpy as jnp
import lerobot.datasets.dataset_metadata as lerobot_dataset_metadata
import lerobot.datasets.io_utils as lerobot_io_utils
import lerobot.datasets.lerobot_dataset as lerobot_dataset
import lerobot.datasets.multi_dataset as lerobot_multi_dataset
import lerobot.datasets.video_utils as lerobot_video_utils
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)
_DEFAULT_LEROBOT_PARQUET_NUM_PROC = 32
_ORIG_LEROBOT_LOAD_NESTED_DATASET = lerobot_io_utils.load_nested_dataset
_ORIG_LEROBOT_DECODE_VIDEO_FRAMES = lerobot_video_utils.decode_video_frames
_LEROBOT_PARQUET_PATCH_ACTIVE = False
_LEROBOT_VIDEO_CHECK_PATCH_ACTIVE = False
_LEROBOT_VIDEO_WARNING_FILTER_ACTIVE = False
_LEROBOT_VIDEO_FALLBACK_PATCH_ACTIVE = False
_LEROBOT_TORCHCODEC_AVAILABLE: bool | None = None
_BAD_SAMPLE_LOGGED_SIGNATURES: set[str] = set()
_BAD_SAMPLE_INDEX_BLACKLIST: dict[str, set[int]] = {}
_BAD_VIDEO_PATH_BLACKLIST: set[str] = set()
_BAD_SAMPLE_LOG_SYNC_STATE: dict[str, tuple[int, int]] = {}
_RETRYABLE_LEROBOT_VIDEO_ERROR_SNIPPETS = (
    "Sample size ",
    "Failed to parse temporal unit",
    "Invalid OBU",
    "Invalid leb128",
    "Invalid data found when processing input",
    "Unknown OBU type",
    "obu_forbidden_bit",
    "obu_reserved_1bit",
    "extension_header_reserved_3bits",
    "zero_bit out of range",
)


def _torchcodec_decoder_available() -> bool:
    global _LEROBOT_TORCHCODEC_AVAILABLE
    if _LEROBOT_TORCHCODEC_AVAILABLE is not None:
        return _LEROBOT_TORCHCODEC_AVAILABLE

    try:
        import torchcodec  # noqa: F401
        from torchcodec.decoders import VideoDecoder  # noqa: F401
    except Exception as exc:
        logging.info("TorchCodec decoder unavailable; using LeRobot video backend 'pyav': %s", exc)
        _LEROBOT_TORCHCODEC_AVAILABLE = False
    else:
        _LEROBOT_TORCHCODEC_AVAILABLE = True

    return _LEROBOT_TORCHCODEC_AVAILABLE


class _HFColumnSequenceCompat:
    """Wrap an HF dataset while preserving lazy indexed column access."""

    def __init__(self, dataset: Any):
        self._dataset = dataset

    def __getitem__(self, key):
        value = self._dataset[key]
        return value

    def __len__(self) -> int:
        return len(self._dataset)

    def select(self, *args, **kwargs):
        return type(self)(self._dataset.select(*args, **kwargs))

    def __getattr__(self, name: str) -> Any:
        # Guard against partially initialized/unpickled wrapper objects.
        if name == "_dataset":
            raise AttributeError(name)
        try:
            dataset = object.__getattribute__(self, "_dataset")
        except AttributeError as exc:
            raise AttributeError(name) from exc
        return getattr(dataset, name)

    def __getstate__(self):
        return {"_dataset": self._dataset}

    def __setstate__(self, state):
        self._dataset = state["_dataset"]


def _patch_lerobot_hf_dataset_column_access() -> None:
    """Patch LeRobot to wrap local HF datasets with `_HFColumnSequenceCompat` once."""
    if getattr(lerobot_dataset.LeRobotDataset.load_hf_dataset, "_plaw_vla_column_compat", False):
        return

    original_load_hf_dataset = lerobot_dataset.LeRobotDataset.load_hf_dataset

    def _patched_load_hf_dataset(self) -> _HFColumnSequenceCompat:
        hf_dataset = original_load_hf_dataset(self)
        if isinstance(hf_dataset, _HFColumnSequenceCompat):
            return hf_dataset
        return _HFColumnSequenceCompat(hf_dataset)

    _patched_load_hf_dataset._plaw_vla_column_compat = True  # type: ignore[attr-defined]
    lerobot_dataset.LeRobotDataset.load_hf_dataset = _patched_load_hf_dataset


def _default_lerobot_video_backend() -> str:
    """Use the most stable default backend while preserving explicit env overrides."""
    if backend := os.environ.get("PLAW_VLA_LEROBOT_VIDEO_BACKEND"):
        return backend
    return "pyav"


def _install_torchvision_video_warning_filter() -> None:
    global _LEROBOT_VIDEO_WARNING_FILTER_ACTIVE
    if _LEROBOT_VIDEO_WARNING_FILTER_ACTIVE:
        return

    warnings.filterwarnings(
        "ignore",
        message=r"The video decoding and encoding capabilities of torchvision are deprecated.*",
        category=UserWarning,
        module=r"torchvision\.io\._video_deprecation_warning",
    )
    _LEROBOT_VIDEO_WARNING_FILTER_ACTIVE = True


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _retryable_bad_sample_log_path() -> pathlib.Path:
    raw_path = os.environ.get("PLAW_VLA_BAD_SAMPLE_LOG_PATH")
    if raw_path:
        return pathlib.Path(raw_path).expanduser()
    return pathlib.Path.cwd() / "plaw_vla_bad_samples.log"


def _max_bad_sample_retries() -> int:
    raw = os.environ.get("PLAW_VLA_BAD_SAMPLE_RETRIES", "32").strip()
    try:
        value = int(raw)
    except ValueError:
        logging.warning("Invalid PLAW_VLA_BAD_SAMPLE_RETRIES=%r; defaulting to 32.", raw)
        value = 32
    return max(0, value)


def _bad_sample_log_sync_interval_s() -> float:
    raw = os.environ.get("PLAW_VLA_BAD_SAMPLE_LOG_SYNC_INTERVAL_S", "1.0").strip()
    try:
        value = float(raw)
    except ValueError:
        logging.warning("Invalid PLAW_VLA_BAD_SAMPLE_LOG_SYNC_INTERVAL_S=%r; defaulting to 1.0.", raw)
        value = 1.0
    return max(0.0, value)


def _is_retryable_lerobot_sample_error(exc: Exception) -> bool:
    if isinstance(exc, lerobot_video_utils.FrameTimestampError):
        return True
    if av_error is not None and isinstance(exc, av_error.InvalidDataError):
        return True

    if not isinstance(exc, (RuntimeError, ValueError, OSError)):
        return False

    message = str(exc)
    return any(snippet in message for snippet in _RETRYABLE_LEROBOT_VIDEO_ERROR_SNIPPETS)


def _append_bad_sample_log_line(line: str) -> None:
    log_path = _retryable_bad_sample_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(log_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(fd, line.encode("utf-8", errors="replace"))
    finally:
        os.close(fd)


def _register_bad_sample(repo_id: str, failed_index: int, video_paths: Sequence[str] = ()) -> None:
    _BAD_SAMPLE_INDEX_BLACKLIST.setdefault(repo_id, set()).add(int(failed_index))
    for path in video_paths:
        if path:
            _BAD_VIDEO_PATH_BLACKLIST.add(path)


def _is_blacklisted_sample(repo_id: str, index: int, video_paths: Sequence[str] = ()) -> bool:
    if index in _BAD_SAMPLE_INDEX_BLACKLIST.get(repo_id, ()):
        return True
    return any(path in _BAD_VIDEO_PATH_BLACKLIST for path in video_paths)


def _parse_bad_sample_log_line(line: str) -> tuple[str, int, tuple[str, ...]] | None:
    line = line.strip()
    if not line:
        return None

    parsed_fields: dict[str, str] = {}
    for field in line.split("\t")[1:]:
        if "=" not in field:
            continue
        key, value = field.split("=", 1)
        parsed_fields[key] = value

    repo_id = parsed_fields.get("repo_id")
    failed_index_raw = parsed_fields.get("failed_index")
    if not repo_id or failed_index_raw is None:
        return None

    try:
        failed_index = int(failed_index_raw)
    except ValueError:
        return None

    video_paths: tuple[str, ...] = ()
    if video_paths_raw := parsed_fields.get("video_paths"):
        try:
            parsed_video_paths = ast.literal_eval(video_paths_raw)
        except (ValueError, SyntaxError):
            parsed_video_paths = ()
        if isinstance(parsed_video_paths, str):
            video_paths = (parsed_video_paths,)
        elif isinstance(parsed_video_paths, Sequence):
            video_paths = tuple(str(path) for path in parsed_video_paths if path)

    return repo_id, failed_index, video_paths


def _sync_bad_sample_blacklist_from_log() -> int:
    log_path = _retryable_bad_sample_log_path()
    try:
        stat_result = log_path.stat()
    except FileNotFoundError:
        return 0

    cache_key = str(log_path.resolve())
    inode = int(getattr(stat_result, "st_ino", 0))
    offset, cached_inode = _BAD_SAMPLE_LOG_SYNC_STATE.get(cache_key, (0, inode))
    if cached_inode != inode or stat_result.st_size < offset:
        offset = 0

    loaded = 0
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        f.seek(offset)
        for line in f:
            parsed = _parse_bad_sample_log_line(line)
            if parsed is None:
                continue
            repo_id, failed_index, video_paths = parsed
            _register_bad_sample(repo_id, failed_index, video_paths)
            loaded += 1
        new_offset = f.tell()

    _BAD_SAMPLE_LOG_SYNC_STATE[cache_key] = (new_offset, inode)
    return loaded


@dataclasses.dataclass(frozen=True)
class _StartupMetadata:
    fps: float
    tasks: Any | None


def _is_local_lerobot_dataset_dir(path: pathlib.Path) -> bool:
    return (path / "meta" / "info.json").is_file() and (path / "data").is_dir()


def _local_lerobot_root(repo_id: str) -> pathlib.Path | None:
    """Return the on-disk dataset root when ``repo_id`` points at a local LeRobot folder."""
    repo_path = pathlib.Path(repo_id).expanduser()
    if not repo_path.is_dir():
        return None
    resolved = repo_path.resolve()
    if _is_local_lerobot_dataset_dir(resolved):
        return resolved
    return None


def _load_startup_metadata(repo_id: str) -> _StartupMetadata:
    repo_path = pathlib.Path(repo_id).expanduser()
    if repo_path.is_dir():
        resolved = repo_path.resolve()
        if _is_local_lerobot_dataset_dir(resolved):
            info = lerobot_io_utils.load_info(resolved)
            tasks = lerobot_io_utils.load_tasks(resolved)
            return _StartupMetadata(fps=float(info["fps"]), tasks=tasks)

    dataset_meta = lerobot_dataset_metadata.LeRobotDatasetMetadata(repo_id)
    return _StartupMetadata(fps=float(dataset_meta.fps), tasks=dataset_meta.tasks)


def _load_nested_dataset_with_default_num_proc(
    pq_dir: pathlib.Path,
    features: Any | None = None,
    episodes: list[int] | None = None,
):
    # Keep metadata on the original path. Only the main data parquet load gets the
    # aggressive parallel read because it is the dominant cost on large datasets.
    if pq_dir.name != "data" or episodes is not None or not pq_dir.is_dir():
        return _ORIG_LEROBOT_LOAD_NESTED_DATASET(pq_dir, features=features, episodes=episodes)

    parquet_files = sorted(str(path) for path in pq_dir.glob("*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"Provided directory does not contain any parquet file: {pq_dir}")

    builder = HFParquetBuilder(
        dataset_name="parquet",
        data_files={"train": parquet_files},
        features=features,
    )
    builder.download_and_prepare(num_proc=_DEFAULT_LEROBOT_PARQUET_NUM_PROC)
    return builder.as_dataset(split="train")


def _check_cached_episodes_without_video_file_scan(self) -> bool:
    """Mirror LeRobot's cache sufficiency check but skip eager MP4 existence scans."""
    if self.hf_dataset is None or len(self.hf_dataset) == 0:
        return False

    available_episodes = {
        ep_idx.item() if isinstance(ep_idx, torch.Tensor) else ep_idx
        for ep_idx in self.hf_dataset.unique("episode_index")
    }

    if self.episodes is None:
        requested_episodes = set(range(self.meta.total_episodes))
    else:
        requested_episodes = set(self.episodes)

    return requested_episodes.issubset(available_episodes)


def _decode_video_frames_with_torchcodec_fallback(
    video_path: pathlib.Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str | None = None,
):
    effective_backend = backend or lerobot_video_utils.get_safe_default_codec()
    try:
        return _ORIG_LEROBOT_DECODE_VIDEO_FRAMES(video_path, timestamps, tolerance_s, effective_backend)
    except RuntimeError as exc:
        if effective_backend != "torchcodec" or "Invalid frame index" not in str(exc):
            raise
        logging.warning(
            "TorchCodec failed for %s with %s; retrying with pyav.",
            video_path,
            exc,
        )
        return _ORIG_LEROBOT_DECODE_VIDEO_FRAMES(video_path, timestamps, tolerance_s, "pyav")


def _apply_lerobot_startup_patches() -> None:
    global _LEROBOT_PARQUET_PATCH_ACTIVE, _LEROBOT_VIDEO_CHECK_PATCH_ACTIVE, _LEROBOT_VIDEO_FALLBACK_PATCH_ACTIVE
    _install_torchvision_video_warning_filter()
    _patch_lerobot_hf_dataset_column_access()

    if not _LEROBOT_PARQUET_PATCH_ACTIVE:
        lerobot_io_utils.load_nested_dataset = _load_nested_dataset_with_default_num_proc
        lerobot_dataset.load_nested_dataset = _load_nested_dataset_with_default_num_proc
        _LEROBOT_PARQUET_PATCH_ACTIVE = True
        logging.info(
            "Enabled parallel LeRobot data parquet loading with num_proc=%d",
            _DEFAULT_LEROBOT_PARQUET_NUM_PROC,
        )

    if _env_flag("PLAW_VLA_SKIP_LEROBOT_VIDEO_FILE_CHECK") and not _LEROBOT_VIDEO_CHECK_PATCH_ACTIVE:
        lerobot_dataset.LeRobotDataset._check_cached_episodes_sufficient = (
            _check_cached_episodes_without_video_file_scan
        )
        _LEROBOT_VIDEO_CHECK_PATCH_ACTIVE = True
        logging.info("Skipping LeRobot video file existence scan during dataset init.")

    if not _LEROBOT_VIDEO_FALLBACK_PATCH_ACTIVE:
        lerobot_video_utils.decode_video_frames = _decode_video_frames_with_torchcodec_fallback
        lerobot_dataset.decode_video_frames = _decode_video_frames_with_torchcodec_fallback
        _LEROBOT_VIDEO_FALLBACK_PATCH_ACTIVE = True
        logging.info("Enabled TorchCodec-to-pyav fallback for LeRobot video decoding.")


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class FaultTolerantDataset(Dataset[T_co]):
    """Skip known-bad LeRobot samples and log them locally for later repair."""

    def __init__(
        self,
        dataset: Dataset[T_co],
        *,
        repo_id: str,
        max_retries: int | None = None,
        catch_all_errors: bool = False,
    ):
        self._dataset = dataset
        self._repo_id = repo_id
        self._max_retries = _max_bad_sample_retries() if max_retries is None else max(0, int(max_retries))
        self._catch_all_errors = catch_all_errors
        self._index_video_paths_cache: dict[int, tuple[str, ...]] = {}
        self._bad_sample_log_sync_interval_s = _bad_sample_log_sync_interval_s()
        self._last_bad_sample_log_sync = 0.0

    def _sync_shared_blacklist_if_due(self, *, force: bool = False) -> None:
        if not force and self._bad_sample_log_sync_interval_s > 0:
            now = time.monotonic()
            if now - self._last_bad_sample_log_sync < self._bad_sample_log_sync_interval_s:
                return
            self._last_bad_sample_log_sync = now
        elif not force:
            self._last_bad_sample_log_sync = time.monotonic()

        _sync_bad_sample_blacklist_from_log()

    def _dataset_video_paths(self, index: int) -> tuple[str, ...]:
        cached = self._index_video_paths_cache.get(index)
        if cached is not None:
            return cached

        ensure_loaded = getattr(self._dataset, "_ensure_hf_dataset_loaded", None)
        if callable(ensure_loaded):
            ensure_loaded()

        hf_dataset = getattr(self._dataset, "hf_dataset", None)
        meta = getattr(self._dataset, "meta", None)
        root = getattr(self._dataset, "root", None)
        if hf_dataset is None or meta is None or root is None or not getattr(meta, "video_keys", None):
            self._index_video_paths_cache[index] = ()
            return ()

        def _scalar_int(value: Any) -> int:
            return int(value.item()) if hasattr(value, "item") else int(value)

        try:
            item = hf_dataset[index]
            ep_idx = _scalar_int(item["episode_index"])
            paths = tuple(
                str((pathlib.Path(root) / meta.get_video_file_path(ep_idx, vid_key)).resolve())
                for vid_key in meta.video_keys
            )
        except Exception:
            paths = ()

        self._index_video_paths_cache[index] = paths
        return paths

    def _is_known_bad_candidate(self, index: int) -> bool:
        return _is_blacklisted_sample(self._repo_id, index, self._dataset_video_paths(index))

    def _next_fallback_index(self, current_index: int, visited: set[int], *, requested_index: int, attempt: int) -> int:
        dataset_len = len(self._dataset)
        if dataset_len <= 1:
            return current_index

        step = 1 + ((requested_index + attempt * 104_729 + os.getpid()) % (dataset_len - 1))
        next_index = (current_index + step) % dataset_len
        while next_index in visited and len(visited) < dataset_len:
            next_index = (next_index + 1) % dataset_len
        visited.add(next_index)
        return next_index

    def _log_retryable_error(
        self,
        *,
        requested_index: int,
        failed_index: int,
        attempt: int,
        exc: Exception,
        video_paths: Sequence[str],
    ) -> None:
        first_line = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
        signature = f"{self._repo_id}|{failed_index}|{exc.__class__.__name__}|{first_line}"
        if signature in _BAD_SAMPLE_LOGGED_SIGNATURES:
            return

        _BAD_SAMPLE_LOGGED_SIGNATURES.add(signature)
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        line = (
            f"{timestamp}\tpid={os.getpid()}\trepo_id={self._repo_id}\trequested_index={requested_index}"
            f"\tfailed_index={failed_index}\tattempt={attempt}\terror_type={exc.__class__.__name__}"
            f"\tvideo_paths={tuple(video_paths)!r}\tmessage={str(exc)!r}\n"
        )
        _append_bad_sample_log_line(line)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        requested_index = index.__index__()
        current_index = requested_index
        visited = {current_index}
        last_exc: Exception | None = None
        self._sync_shared_blacklist_if_due()

        for attempt in range(self._max_retries + 1):
            if self._is_known_bad_candidate(current_index):
                if attempt >= self._max_retries:
                    break
                current_index = self._next_fallback_index(
                    current_index,
                    visited,
                    requested_index=requested_index,
                    attempt=attempt,
                )
                continue

            try:
                return self._dataset[current_index]
            except Exception as exc:
                if not self._catch_all_errors and not _is_retryable_lerobot_sample_error(exc):
                    raise
                last_exc = exc
                video_paths = self._dataset_video_paths(current_index)
                _register_bad_sample(self._repo_id, current_index, video_paths)
                self._log_retryable_error(
                    requested_index=requested_index,
                    failed_index=current_index,
                    attempt=attempt,
                    exc=exc,
                    video_paths=video_paths,
                )
                self._sync_shared_blacklist_if_due(force=True)
                if attempt >= self._max_retries:
                    break
                current_index = self._next_fallback_index(
                    current_index,
                    visited,
                    requested_index=requested_index,
                    attempt=attempt,
                )

        raise RuntimeError(
            f"Exceeded {self._max_retries} retries while skipping bad LeRobot samples for repo_id={self._repo_id!r}."
        ) from last_exc

    def __len__(self) -> int:
        return len(self._dataset)

    @property
    def num_frames(self) -> int:
        return getattr(self._dataset, "num_frames", len(self._dataset))

    @property
    def num_episodes(self) -> int:
        return getattr(self._dataset, "num_episodes", 1)


class NoVideoQueryDataset(Dataset[dict[str, Any]]):
    """Wrap a LeRobot dataset and avoid decoding video files.

    The wrapper reproduces LeRobot's non-video indexing path, but replaces each
    video query with a zero-valued frame tensor of the expected shape. This is
    sufficient for jobs such as normalization-stat computation that only depend
    on non-visual fields while still running the same downstream transforms.
    """

    def __init__(self, dataset: Dataset[dict[str, Any]]):
        self._dataset = dataset
        self._video_placeholder_cache: dict[tuple[str, int], torch.Tensor] = {}

    def __getattr__(self, name: str) -> Any:
        if name == "_dataset":
            raise AttributeError(name)
        return getattr(self._dataset, name)

    def _video_placeholder(self, key: str, num_frames: int) -> torch.Tensor:
        cache_key = (key, num_frames)
        cached = self._video_placeholder_cache.get(cache_key)
        if cached is not None:
            return cached.clone()

        feature = getattr(self._dataset, "meta").features[key]
        if len(feature["shape"]) != 3:
            raise ValueError(f"Expected 3D video feature shape for {key!r}, got {feature['shape']!r}.")
        _height, _width, channels = feature["shape"]
        # Norm-stat scans do not depend on pixels, so use a minimal uint8 placeholder
        # to keep policy input transforms satisfied without paying full image costs.
        frame = torch.zeros((channels, 1, 1), dtype=torch.uint8)
        if num_frames > 1:
            frame = frame.unsqueeze(0).expand(num_frames, -1, -1, -1).clone()
        self._video_placeholder_cache[cache_key] = frame
        return frame.clone()

    def __getitem__(self, idx: SupportsIndex) -> dict[str, Any]:
        dataset = self._dataset
        ensure_loaded = getattr(dataset, "_ensure_hf_dataset_loaded", None)
        if callable(ensure_loaded):
            ensure_loaded()

        hf_dataset = getattr(dataset, "hf_dataset", None)
        meta = getattr(dataset, "meta", None)
        if hf_dataset is None or meta is None:
            raise TypeError("NoVideoQueryDataset requires an underlying LeRobot dataset with hf_dataset and meta.")

        item = hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        abs_idx = item["index"].item()

        query_indices = None
        if getattr(dataset, "delta_indices", None) is not None:
            query_indices, padding = dataset._get_query_indices(abs_idx, ep_idx)
            query_result = dataset._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, value in query_result.items():
                item[key] = value

        if len(meta.video_keys) > 0:
            item = {
                **{
                    vid_key: self._video_placeholder(
                        vid_key,
                        len(query_indices[vid_key]) if query_indices is not None and vid_key in query_indices else 1,
                    )
                    for vid_key in meta.video_keys
                },
                **item,
            }

        if getattr(dataset, "image_transforms", None) is not None:
            for cam in meta.camera_keys:
                item[cam] = dataset.image_transforms(item[cam])

        task_idx = item["task_index"].item()
        item["task"] = meta.tasks.iloc[task_idx].name

        if "subtask_index" in dataset.features and meta.subtasks is not None:
            subtask_idx = item["subtask_index"].item()
            item["subtask"] = meta.subtasks.iloc[subtask_idx].name

        return item

    def __len__(self) -> int:
        return len(self._dataset)

    @property
    def num_frames(self) -> int:
        return getattr(self._dataset, "num_frames", len(self._dataset))

    @property
    def num_episodes(self) -> int:
        return getattr(self._dataset, "num_episodes", 1)


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def data_configs(self) -> Sequence[_config.DataConfig]:
        """Get all data configs represented by this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_configs.")

    def checkpoint_asset_metadata(self) -> Sequence[dict[str, Any]]:
        """Get optional checkpoint asset metadata for manifest generation."""
        raise NotImplementedError("Subclasses of DataLoader should implement checkpoint_asset_metadata.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)

    @property
    def num_frames(self) -> int:
        """Delegate to underlying dataset for MultiLeRobotDataset compatibility."""
        return getattr(self._dataset, "num_frames", len(self._dataset))

    @property
    def num_episodes(self) -> int:
        """Delegate to underlying dataset for MultiLeRobotDataset compatibility."""
        return getattr(self._dataset, "num_episodes", 1)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class WeightedConcatDataset(Dataset):
    """Concatenates multiple datasets and optionally stores explicit sample weights for testing."""

    def __init__(self, datasets: list[Dataset], sample_weights: Sequence[float] | None = None):
        self._datasets = datasets
        self._dataset_lengths = [len(d) for d in datasets]
        self._cumulative_sizes: list[int] = []
        total = 0
        for dataset_length in self._dataset_lengths:
            total += dataset_length
            self._cumulative_sizes.append(total)

        self._sample_weights: list[float] | None = None
        if sample_weights is not None:
            sample_weights = list(sample_weights)
            if total != len(sample_weights):
                raise ValueError(
                    f"Expected {total} sample weights for concatenated dataset, got {len(sample_weights)}."
                )
            self._sample_weights = sample_weights

    def _find_dataset(self, index: int) -> tuple[int, int]:
        for i, cum_size in enumerate(self._cumulative_sizes):
            if index < cum_size:
                offset = self._cumulative_sizes[i - 1] if i > 0 else 0
                return i, index - offset
        raise IndexError(f"Index {index} out of range for {len(self)} samples")

    def __getitem__(self, index):
        ds_idx, local_idx = self._find_dataset(index)
        return self._datasets[ds_idx][local_idx]

    def __len__(self) -> int:
        return self._cumulative_sizes[-1] if self._cumulative_sizes else 0

    @property
    def sample_weights(self) -> list[float]:
        if self._sample_weights is None:
            raise ValueError("Explicit sample weights were not provided for this dataset.")
        return self._sample_weights


class WeightedGroupSampler(torch.utils.data.Sampler[int]):
    """Sample from concatenated datasets by first sampling a parent group, then a frame within that group."""

    def __init__(
        self,
        group_dataset_lengths: Sequence[Sequence[int]],
        group_weights: Sequence[float],
        *,
        num_samples: int,
        generator: torch.Generator | None = None,
        chunk_size: int = 65_536,
    ):
        if len(group_dataset_lengths) != len(group_weights):
            raise ValueError(
                f"Expected group_dataset_lengths and group_weights to have the same length, "
                f"got {len(group_dataset_lengths)} and {len(group_weights)}."
            )
        if not group_dataset_lengths:
            raise ValueError("WeightedGroupSampler requires at least one dataset group.")
        if num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, got {num_samples}.")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}.")

        self._group_weights = torch.as_tensor(group_weights, dtype=torch.double)
        if torch.any(self._group_weights < 0):
            raise ValueError(f"group_weights must be non-negative, got {group_weights}.")
        if torch.sum(self._group_weights) <= 0:
            raise ValueError(f"group_weights must sum to a positive value, got {group_weights}.")

        self._group_total_frames: list[int] = []
        self._group_child_cumulative: list[torch.Tensor] = []
        self._group_child_global_offsets: list[torch.Tensor] = []
        global_offset = 0
        for group_lengths in group_dataset_lengths:
            lengths = [int(length) for length in group_lengths]
            if not lengths:
                raise ValueError("Each dataset group must contain at least one child dataset.")
            if any(length <= 0 for length in lengths):
                raise ValueError(f"Child dataset lengths must be > 0, got {lengths}.")

            child_offsets: list[int] = []
            child_cumulative: list[int] = []
            group_running = 0
            for length in lengths:
                child_offsets.append(global_offset + group_running)
                group_running += length
                child_cumulative.append(group_running)

            self._group_total_frames.append(group_running)
            self._group_child_global_offsets.append(torch.tensor(child_offsets, dtype=torch.int64))
            self._group_child_cumulative.append(torch.tensor(child_cumulative, dtype=torch.int64))
            global_offset += group_running

        self._num_samples = num_samples
        self._generator = generator if generator is not None else torch.Generator()
        self._chunk_size = chunk_size

    def __len__(self) -> int:
        return self._num_samples

    def __iter__(self):
        remaining = self._num_samples
        while remaining > 0:
            chunk = min(self._chunk_size, remaining)
            group_indices = torch.multinomial(
                self._group_weights,
                chunk,
                replacement=True,
                generator=self._generator,
            )
            sampled_indices = torch.empty(chunk, dtype=torch.int64)

            for group_idx in torch.unique(group_indices, sorted=True).tolist():
                positions = torch.nonzero(group_indices == group_idx, as_tuple=False).flatten()
                local_offsets = torch.randint(
                    self._group_total_frames[group_idx],
                    (positions.numel(),),
                    generator=self._generator,
                    dtype=torch.int64,
                )
                child_cumulative = self._group_child_cumulative[group_idx]
                child_indices = torch.bucketize(local_offsets, child_cumulative, right=False)
                prev_cumulative = torch.zeros_like(local_offsets)
                has_prev = child_indices > 0
                if torch.any(has_prev):
                    prev_cumulative[has_prev] = child_cumulative[child_indices[has_prev] - 1]
                sampled_indices[positions] = (
                    self._group_child_global_offsets[group_idx][child_indices] + (local_offsets - prev_cumulative)
                )

            yield from sampled_indices.tolist()
            remaining -= chunk


@dataclasses.dataclass(frozen=True)
class _LoadedChildDataset:
    repo_id: str
    dataset: Dataset
    num_frames: int
    elapsed_s: float


@dataclasses.dataclass(frozen=True)
class _LoadedDatasetGroup:
    datasets: list[Dataset]
    child_lengths: list[int]
    group_frames: int
    weight: float
def _load_child_dataset(
    child_config: _config.DataConfig,
    *,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    skip_norm_stats: bool,
) -> _LoadedChildDataset:
    child_start = time.perf_counter()
    ds = create_torch_dataset(child_config, action_horizon, model_config)
    ds = FaultTolerantDataset(
        ds,
        repo_id=str(child_config.repo_id),
        catch_all_errors=True,
    )
    ds = transform_dataset(ds, child_config, skip_norm_stats=skip_norm_stats)
    num_frames = len(ds)
    return _LoadedChildDataset(
        repo_id=str(child_config.repo_id),
        dataset=ds,
        num_frames=num_frames,
        elapsed_s=time.perf_counter() - child_start,
    )


def _load_child_datasets(
    child_configs: Sequence[_config.DataConfig],
    *,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    skip_norm_stats: bool,
) -> list[_LoadedChildDataset]:
    return [
        _load_child_dataset(
            child_config,
            action_horizon=action_horizon,
            model_config=model_config,
            skip_norm_stats=skip_norm_stats,
        )
        for child_config in child_configs
    ]


def _load_dataset_group(
    data_config: _config.DataConfig,
    *,
    weight: float,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    skip_norm_stats: bool,
) -> _LoadedDatasetGroup:
    child_configs = _expand_data_config_children(data_config)
    if len(child_configs) == 1:
        logging.info("Creating dataset for repo_id=%s with weight=%s", child_configs[0].repo_id, weight)
    else:
        logging.info(
            "Creating dataset group with %d child datasets (first repo_id=%s) and weight=%s",
            len(child_configs),
            child_configs[0].repo_id,
            weight,
        )

    group_frames = 0
    loaded_children = _load_child_datasets(
        child_configs,
        action_horizon=action_horizon,
        model_config=model_config,
        skip_norm_stats=skip_norm_stats,
    )
    group_datasets = []
    child_lengths: list[int] = []
    for loaded_child in loaded_children:
        ds = loaded_child.dataset
        group_datasets.append(ds)
        group_frames += loaded_child.num_frames
        child_lengths.append(loaded_child.num_frames)
        logging.info(
            "  child repo_id=%s -> %d frames loaded in %.2fs",
            loaded_child.repo_id,
            loaded_child.num_frames,
            loaded_child.elapsed_s,
        )
    if group_frames <= 0:
        raise ValueError(f"Dataset group for repo_id={child_configs[0].repo_id} has no frames.")
    logging.info("  -> %d frames across %d dataset(s)", group_frames, len(child_configs))

    return _LoadedDatasetGroup(
        datasets=group_datasets,
        child_lengths=child_lengths,
        group_frames=group_frames,
        weight=weight,
    )


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        # Add minimal world-model history and future clips for debug configs.
        # Counts match WorldModelDataConfig defaults (3 history frames, including
        # the current frame, and 2 future frames).
        if "base_0_rgb" in observation.images and "base_0_rgb_history" not in observation.images:
            base_image = observation.images["base_0_rgb"]
            history_len = 3
            future_len = 2
            observation.images["base_0_rgb_history"] = jnp.stack([base_image] * history_len, axis=0)
            observation.image_masks["base_0_rgb_history"] = jnp.ones((history_len,), dtype=bool)
            observation.images["base_0_rgb_future"] = jnp.stack([base_image] * future_len, axis=0)
            observation.image_masks["base_0_rgb_future"] = jnp.ones((future_len,), dtype=bool)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def _expand_data_config_children(data_config: _config.DataConfig) -> list[_config.DataConfig]:
    repo_ids = data_config.repo_id
    if not isinstance(repo_ids, Sequence) or isinstance(repo_ids, str):
        return [data_config]

    asset_ids = data_config.asset_id
    per_repo_norm_stats = data_config.per_repo_norm_stats or {}
    expanded: list[_config.DataConfig] = []
    for idx, repo_id in enumerate(repo_ids):
        if isinstance(asset_ids, Sequence) and not isinstance(asset_ids, str):
            child_asset_id = asset_ids[idx]
        else:
            child_asset_id = asset_ids
        expanded.append(
            dataclasses.replace(
                data_config,
                repo_id=repo_id,
                asset_id=child_asset_id,
                norm_stats=per_repo_norm_stats.get(repo_id, data_config.norm_stats),
                per_repo_norm_stats=None,
            )
        )
    return expanded


def create_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    include_videos: bool = True,
) -> Dataset:
    """Create a dataset for training.

    Args:
        data_config: Data configuration containing repo_id, world_model settings, etc.
            repo_id can be a single dataset path, a parent directory containing multiple
            task-level datasets, or an explicit list of dataset paths for multi-dataset training.
        action_horizon: Number of future action steps to predict.
        model_config: Model configuration for FakeDataset.
        include_videos: Whether to decode LeRobot video-backed camera streams. Disable this for
            non-visual jobs such as normalization-stat scans.

    Returns:
        Dataset ready for training.
    """
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    _apply_lerobot_startup_patches()
    video_backend = _default_lerobot_video_backend()

    if isinstance(repo_id, list):
        metadata_start = time.perf_counter()
        startup_metas = [_load_startup_metadata(r) for r in repo_id]
        delta_timestamps = _build_delta_timestamps(data_config, action_horizon, startup_metas[0].fps)
        logging.info(
            "Resolved metadata for %d repo_ids in %.2fs (first repo_id=%s)",
            len(repo_id),
            time.perf_counter() - metadata_start,
            repo_id[0],
        )

        dataset_start = time.perf_counter()
        dataset = lerobot_multi_dataset.MultiLeRobotDataset(
            repo_id,
            delta_timestamps=delta_timestamps,
            video_backend=video_backend,
        )
        for index, single_repo_id in enumerate(repo_id):
            local_root = _local_lerobot_root(single_repo_id)
            if local_root is None:
                continue
            opened_root = pathlib.Path(dataset._datasets[index].root).resolve()
            if opened_root == local_root:
                continue
            dataset._datasets[index] = lerobot_dataset.LeRobotDataset(
                single_repo_id,
                root=local_root,
                delta_timestamps=delta_timestamps,
                video_backend=video_backend,
            )
        logging.info(
            "Initialized MultiLeRobotDataset with %d repo_ids in %.2fs (first repo_id=%s)",
            len(repo_id),
            time.perf_counter() - dataset_start,
            repo_id[0],
        )

        for n, d in enumerate(dataset._datasets):
            ds = d
            if not include_videos:
                ds = NoVideoQueryDataset(ds)
            if data_config.prompt_from_task:
                ds = TransformedDataset(ds, [_transforms.PromptFromLeRobotTask(startup_metas[n].tasks)])
            dataset._datasets[n] = ds

        for i, d in enumerate(dataset._datasets):
            logging.info(f"Dataset {i} ({repo_id[i]}) has {len(d)} frames.")
    else:
        metadata_start = time.perf_counter()
        startup_meta = _load_startup_metadata(repo_id)
        delta_timestamps = _build_delta_timestamps(data_config, action_horizon, startup_meta.fps)
        logging.info("Resolved metadata for repo_id=%s in %.2fs", repo_id, time.perf_counter() - metadata_start)

        dataset_start = time.perf_counter()
        dataset = lerobot_dataset.LeRobotDataset(
            repo_id,
            root=_local_lerobot_root(repo_id),
            delta_timestamps=delta_timestamps,
            video_backend=video_backend,
        )
        logging.info(
            "Initialized LeRobotDataset for repo_id=%s in %.2fs",
            repo_id,
            time.perf_counter() - dataset_start,
        )

        if not include_videos:
            dataset = NoVideoQueryDataset(dataset)

        if data_config.prompt_from_task:
            dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(startup_meta.tasks)])

    return dataset


def _build_delta_timestamps(
    data_config: _config.DataConfig,
    action_horizon: int,
    fps: float,
) -> dict[str, list[float]]:
    """Build delta_timestamps for LeRobotDataset.

    Args:
        data_config: Data configuration with world_model and action_sequence_keys.
        action_horizon: Number of action steps.
        fps: Dataset frames per second.

    Returns:
        Dictionary mapping data keys to their temporal sampling deltas.
    """

    wm = data_config.world_model
    wm_enabled = len(data_config.world_model_transforms.inputs) > 0

    action_times = data_config.resolve_action_time_offsets(action_horizon, fps)

    if not wm_enabled:
        return {key: list(action_times) for key in data_config.action_sequence_keys}

    layout_indices = list(wm.resolve_layout_indices())
    _validate_world_model_frame_indices(layout_indices)
    history_num_frames = wm.resolve_history_num_frames()
    future_num_frames = wm.resolve_future_num_frames()
    image_time_offsets = wm.resolve_time_offsets(fps)

    if wm.time_offsets_s is None:
        logging.info(
            "World model enabled: history=%s, future=%s, stride=%s, fps=%s",
            history_num_frames,
            future_num_frames,
            wm.frame_stride,
            fps,
        )
    else:
        logging.info(
            "World model enabled: history=%s, future=%s, explicit_time_offsets=%s, fps=%s",
            history_num_frames,
            future_num_frames,
            wm.time_offsets_s,
            fps,
        )
    logging.info(f"Image layout indices: {layout_indices} ({len(layout_indices)} total)")
    logging.info(f"Image time offsets (s): {image_time_offsets}")

    return {
        **{key: list(image_time_offsets) for key in (wm.image_keys or ())},
        **{key: list(action_times) for key in data_config.action_sequence_keys},
    }


def _validate_world_model_frame_indices(frame_indices: Sequence[int]) -> None:
    """Validate the resolved world-model temporal window."""
    if not frame_indices:
        raise ValueError("World model frame_indices must not be empty.")
    if 0 not in frame_indices:
        raise ValueError(f"World model frame_indices must include 0, got {frame_indices}.")
    if any(next_idx <= idx for idx, next_idx in zip(frame_indices, frame_indices[1:])):
        raise ValueError(f"World model frame_indices must be strictly increasing, got {frame_indices}.")


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.world_model_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.world_model_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    effective_skip_norm_stats = skip_norm_stats

    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")
    if not effective_skip_norm_stats and len(data_config.action_sequence_keys) == 0 and data_config.norm_stats is None:
        effective_skip_norm_stats = True
        logging.info("Skipping norm stats because action_sequence_keys is empty and no norm stats are available.")

    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=effective_skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    child_configs = _expand_data_config_children(data_config)
    loaded_children = _load_child_datasets(
        child_configs,
        action_horizon=action_horizon,
        model_config=model_config,
        skip_norm_stats=skip_norm_stats,
    )
    if len(loaded_children) == 1:
        dataset = loaded_children[0].dataset
        logging.info(
            "  child repo_id=%s -> %d frames loaded in %.2fs",
            loaded_children[0].repo_id,
            loaded_children[0].num_frames,
            loaded_children[0].elapsed_s,
        )
    else:
        logging.info("Creating %d child datasets for repo group", len(child_configs))
        child_datasets: list[Dataset] = []
        for loaded_child in loaded_children:
            child_datasets.append(loaded_child.dataset)
            logging.info(
                "  child repo_id=%s -> %d frames loaded in %.2fs",
                loaded_child.repo_id,
                loaded_child.num_frames,
                loaded_child.elapsed_s,
            )
        dataset = torch.utils.data.ConcatDataset(child_datasets)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader, seed=seed, framework=framework)


def create_multi_torch_data_loader(
    data_configs: list[tuple[_config.DataConfig, float]],
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = True,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
    checkpoint_asset_metadata: Sequence[dict[str, Any]] | None = None,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader that combines multiple datasets with weighted sampling.

    Each (DataConfig, weight) pair creates an independent dataset with its own
    transforms. Sampling first chooses a parent dataset group by weight and then
    uniformly samples a frame from within that group.
    """
    datasets = []
    group_dataset_lengths: list[list[int]] = []
    group_weights: list[float] = []

    group_specs = list(data_configs)
    loaded_groups = [
        _load_dataset_group(
            data_config,
            weight=weight,
            action_horizon=action_horizon,
            model_config=model_config,
            skip_norm_stats=skip_norm_stats,
        )
        for data_config, weight in group_specs
    ]

    for loaded_group in loaded_groups:
        datasets.extend(loaded_group.datasets)
        group_dataset_lengths.append(loaded_group.child_lengths)
        group_weights.append(loaded_group.weight)

    combined = WeightedConcatDataset(datasets)
    logging.info(f"Combined dataset: {len(combined)} total frames from {len(datasets)} dataset groups")

    sampler_generator = torch.Generator()
    sampler_generator.manual_seed(seed)
    sampler = WeightedGroupSampler(
        group_dataset_lengths,
        group_weights,
        num_samples=len(combined),
        generator=sampler_generator,
    )

    if framework == "pytorch":
        local_batch_size = batch_size
        if torch.distributed.is_initialized():
            local_batch_size = batch_size // torch.distributed.get_world_size()
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        combined,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=False,
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    if checkpoint_asset_metadata is None:
        checkpoint_asset_metadata = [
            {
                "repo_id": data_config.repo_id,
                "asset_id": data_config.asset_id,
                "weight": weight,
            }
            for data_config, weight in data_configs
        ]

    return DataLoaderImpl(
        data_configs[0][0],
        data_loader,
        data_configs=[data_config for data_config, _ in data_configs],
        checkpoint_asset_metadata=checkpoint_asset_metadata,
        seed=seed,
        framework=framework,
    )


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches
        self._epoch = 0

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            pin_memory=torch.cuda.is_available(),
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used for the next dataset pass.

        ``DistributedSampler`` derives its shuffle from this value. The training
        loop sets the starting epoch on resume; each later pass increments it.
        """
        self._epoch = int(epoch)
        sampler = getattr(self._data_loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self._epoch)

    def __len__(self) -> int:
        """Number of batches in one dataset pass."""
        return len(self._data_loader)

    def __iter__(self):
        num_items = 0
        while True:
            sampler = getattr(self._data_loader, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self._epoch)
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    # The next pass must use a new shuffle. Leaving the epoch at 0
                    # makes every DDP pass repeat the same sample order.
                    self._epoch += 1
                    break
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(_to_torch_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _to_torch_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, jax.Array):
        value = np.asarray(value)
    return torch.as_tensor(value)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    _install_torchvision_video_warning_filter()

    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
def _resolve_sampled_frame_count(
    min_count: int | None,
    max_count: int,
    rng: np.random.Generator,
    *,
    name: str,
    sample_power: float = 0.0,
) -> int:
    if max_count < 0:
        raise ValueError(f"{name} max_count must be >= 0, got {max_count}")
    if max_count == 0:
        return 0

    effective_min = max_count if min_count is None else min_count
    if effective_min <= 0:
        raise ValueError(f"{name} min_count must be > 0, got {effective_min}")
    if effective_min > max_count:
        raise ValueError(f"{name} min_count ({effective_min}) must be <= max_count ({max_count})")
    if effective_min == max_count:
        return max_count

    candidates = np.arange(effective_min, max_count + 1, dtype=np.int64)
    if sample_power == 0.0:
        return int(rng.choice(candidates))

    weights = candidates.astype(np.float64) ** sample_power
    weights /= weights.sum()
    return int(rng.choice(candidates, p=weights))


def _sample_temporal_selection_mask(
    total_count: int,
    selected_count: int,
    *,
    sampler: Literal["prefix", "suffix"],
) -> np.ndarray:
    if total_count <= 0:
        raise ValueError(f"total_count must be > 0, got {total_count}")
    if selected_count <= 0:
        raise ValueError(f"selected_count must be > 0, got {selected_count}")
    if selected_count > total_count:
        raise ValueError(f"selected_count ({selected_count}) must be <= total_count ({total_count})")

    if selected_count == total_count:
        return np.ones((total_count,), dtype=bool)

    if sampler == "prefix":
        selected = np.arange(selected_count, dtype=np.int64)
    elif sampler == "suffix":
        selected = np.arange(total_count - selected_count, total_count, dtype=np.int64)
    else:
        raise ValueError(f"Unsupported temporal sampler: {sampler!r}")

    mask = np.zeros((total_count,), dtype=bool)
    mask[selected] = True
    return mask


def _broadcast_temporal_selection_mask(
    selection_mask: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    return np.broadcast_to(selection_mask[None, :], (batch_size, selection_mask.shape[0])).copy()


def _combine_temporal_selection_mask(
    existing_mask: np.ndarray,
    sampled_mask: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    existing = np.asarray(existing_mask, dtype=bool)
    if existing.ndim == 1:
        existing = _broadcast_temporal_selection_mask(existing, batch_size=batch_size)
    elif existing.ndim != 2 or existing.shape[0] != batch_size:
        raise ValueError(
            f"Expected temporal selection mask with shape ({batch_size}, T) or (T,), got {existing.shape}."
        )

    if existing.shape[1] != sampled_mask.shape[0]:
        raise ValueError(
            f"Expected temporal selection mask with {sampled_mask.shape[0]} entries, got {existing.shape[1]}."
        )

    return existing & _broadcast_temporal_selection_mask(sampled_mask, batch_size=batch_size)


def _maybe_apply_temporal_sampler(
    *,
    images: dict[str, Any],
    image_masks: dict[str, Any] | None,
    image_key: str,
    selection_key: str,
    min_count: int | None,
    rng: np.random.Generator,
    name: str,
    sample_power: float,
    sampler: Literal["prefix", "suffix"],
) -> dict[str, Any] | None:
    image = images.get(image_key)
    if image is None or getattr(image, "ndim", 0) < 2:
        return image_masks

    total_count = int(image.shape[1])
    selected_count = _resolve_sampled_frame_count(
        min_count,
        total_count,
        rng,
        name=name,
        sample_power=sample_power,
    )
    if selected_count == total_count and (image_masks is None or selection_key not in image_masks):
        return image_masks

    sampled_mask = _sample_temporal_selection_mask(total_count, selected_count, sampler=sampler)
    batch_size = int(image.shape[0])
    if image_masks is None:
        image_masks = {}

    if selection_key in image_masks:
        image_masks[selection_key] = _combine_temporal_selection_mask(
            image_masks[selection_key],
            sampled_mask,
            batch_size=batch_size,
        )
    else:
        image_masks[selection_key] = _broadcast_temporal_selection_mask(sampled_mask, batch_size=batch_size)

    return image_masks


def _maybe_apply_world_model_sampling(
    batch: dict[str, Any],
    data_config: _config.DataConfig,
    rng: np.random.Generator,
) -> dict[str, Any]:
    if len(data_config.world_model_transforms.inputs) == 0:
        return batch

    images = batch.get("image")
    image_masks = batch.get("image_mask")
    if not isinstance(images, dict):
        return batch

    image_masks = _maybe_apply_temporal_sampler(
        images=images,
        image_masks=image_masks if isinstance(image_masks, dict) else None,
        image_key="base_0_rgb_history",
        selection_key="base_0_rgb_history_selection",
        min_count=data_config.world_model.train_min_history_num_frames,
        rng=rng,
        name="history_num_frames",
        sample_power=data_config.world_model.train_history_sample_power,
        sampler="suffix",
    )
    image_masks = _maybe_apply_temporal_sampler(
        images=images,
        image_masks=image_masks,
        image_key="base_0_rgb_future",
        selection_key="base_0_rgb_future_selection",
        min_count=data_config.world_model.train_min_future_num_frames,
        rng=rng,
        name="future_num_frames",
        sample_power=data_config.world_model.train_future_sample_power,
        sampler="prefix",
    )

    if image_masks is not None:
        batch["image_mask"] = image_masks

    return batch


class DataLoaderImpl(DataLoader):
    def __init__(
        self,
        data_config: _config.DataConfig,
        data_loader: TorchDataLoader,
        *,
        data_configs: Sequence[_config.DataConfig] | None = None,
        checkpoint_asset_metadata: Sequence[dict[str, Any]] | None = None,
        seed: int = 0,
        framework: Literal["jax", "pytorch"] = "pytorch",
    ):
        self._data_config = data_config
        self._data_configs = tuple(data_configs) if data_configs is not None else (data_config,)
        self._checkpoint_asset_metadata = (
            tuple(checkpoint_asset_metadata) if checkpoint_asset_metadata is not None else ()
        )
        self._data_loader = data_loader
        self._temporal_rng = np.random.default_rng(seed)
        self._framework = framework

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def data_configs(self) -> Sequence[_config.DataConfig]:
        return self._data_configs

    def checkpoint_asset_metadata(self) -> Sequence[dict[str, Any]]:
        return self._checkpoint_asset_metadata

    def set_epoch(self, epoch: int) -> None:
        self._data_loader.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self._data_loader)

    def __iter__(self):
        for batch in self._data_loader:
            batch = _maybe_apply_world_model_sampling(batch, self._data_config, self._temporal_rng)
            if self._framework == "pytorch":
                batch = jax.tree.map(_to_torch_tensor, batch)
            yield _model.Observation.from_dict(batch), batch.get("actions")
