# Copyright 2026 PLaW-VLA authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Iterable
import hashlib
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from openpi.datasets.common import staging

_SAFE_RAY_BASE_PATH_LEN = 40
_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_PYTHON_PATHS = (_REPO_ROOT, _REPO_ROOT / "src")


class _RayStub:
    @staticmethod
    def remote(*args, **kwargs):
        del args, kwargs

        def decorator(fn):
            def _missing_remote(*_args, **_kwargs):
                raise RuntimeError("ray is not installed; use conversion_num_workers=1 for serial conversion.")

            fn.remote = _missing_remote
            return fn

        return decorator

    @staticmethod
    def is_initialized() -> bool:
        return False

    @staticmethod
    def shutdown() -> None:
        return None

    @staticmethod
    def wait(*args, **kwargs):
        raise RuntimeError("ray is not installed; parallel conversion is unavailable.")

    @staticmethod
    def get(*args, **kwargs):
        raise RuntimeError("ray is not installed; parallel conversion is unavailable.")


def import_optional_ray() -> tuple[Any, bool]:
    try:
        import ray
    except ModuleNotFoundError:
        return _RayStub(), False
    return ray, True


def _sanitize_fragment(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-") or "ray"


def resolve_ray_temp_dir(
    *,
    label: str,
    unique_key: str,
    requested_root: Path | None = None,
) -> Path:
    root = requested_root
    if root is None:
        env_root = os.environ.get("RAY_TMPDIR") or os.environ.get("TMPDIR") or os.environ.get("TMP") or os.environ.get("TEMP")
        root = Path(env_root) if env_root else Path(tempfile.gettempdir())

    root = root.expanduser().resolve()
    digest = hashlib.sha1(unique_key.encode("utf-8"), usedforsecurity=False).hexdigest()[:10]
    label_fragment = _sanitize_fragment(label)[:12]
    candidate = root / f"ray-{label_fragment}-{digest}"

    if len(candidate.as_posix()) > _SAFE_RAY_BASE_PATH_LEN:
        candidate = root / f"ray-{digest}"
    if len(candidate.as_posix()) > _SAFE_RAY_BASE_PATH_LEN:
        candidate = Path("/tmp") / f"ray-{digest}"

    candidate.mkdir(parents=True, exist_ok=True)
    return candidate.resolve()


def configure_process_temp_dir(temp_dir: Path) -> Path:
    temp_dir = temp_dir.expanduser().resolve()
    temp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(temp_dir)
    os.environ["TMP"] = str(temp_dir)
    os.environ["TEMP"] = str(temp_dir)
    os.environ["RAY_TMPDIR"] = str(temp_dir)
    return temp_dir


def cleanup_temp_paths(paths: Iterable[Path | str | None]) -> list[Path]:
    cleaned: list[Path] = []
    seen: set[Path] = set()
    for path_value in paths:
        if path_value is None:
            continue
        path = Path(path_value).expanduser().resolve()
        if path in seen or not path.exists():
            continue
        staging.remove_tree(path)
        cleaned.append(path)
        seen.add(path)
    return cleaned


def _resolve_python_paths(python_paths: Iterable[Path | str] | None = None) -> tuple[str, ...]:
    ordered_paths: list[str] = []
    seen: set[str] = set()

    def _append_path(path_value: Path | str) -> None:
        path = Path(path_value).expanduser().resolve()
        if not path.exists():
            return
        path_str = str(path)
        if path_str in seen:
            return
        ordered_paths.append(path_str)
        seen.add(path_str)

    for path in _DEFAULT_PYTHON_PATHS:
        _append_path(path)
    for path in python_paths or ():
        _append_path(path)

    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    for raw_fragment in existing_pythonpath.split(os.pathsep):
        fragment = raw_fragment.strip()
        if not fragment or fragment in seen:
            continue
        ordered_paths.append(fragment)
        seen.add(fragment)

    return tuple(ordered_paths)


def build_runtime_env(*, python_paths: Iterable[Path | str] | None = None) -> dict[str, object]:
    env_vars = {
        "PYTHONPATH": os.pathsep.join(_resolve_python_paths(python_paths)),
        "RAY_ENABLE_UV_RUN_RUNTIME_ENV": os.environ.get("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0"),
        "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": os.environ.get("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0"),
    }
    return {"env_vars": env_vars}


def init_ray(
    *,
    num_workers: int,
    temp_dir: Path,
    python_paths: Iterable[Path | str] | None = None,
) -> None:
    import ray

    temp_dir = configure_process_temp_dir(temp_dir)
    ray.init(
        num_cpus=num_workers,
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=True,
        _temp_dir=str(temp_dir),
        runtime_env=build_runtime_env(python_paths=python_paths),
    )
