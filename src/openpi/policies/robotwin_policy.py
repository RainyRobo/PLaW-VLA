# Copyright 2026 PLaW-VLA authors. SPDX-License-Identifier: Apache-2.0
"""RoboTwin policy wrapper with per-task normalization.

Wraps an underlying :class:`openpi.policies.policy.Policy` and dynamically swaps
the per-task ``norm_stats`` referenced by the embedded
:class:`openpi.transforms.Normalize` / :class:`openpi.transforms.Unnormalize`
transforms based on a ``__robotwin_task_id__`` field that the RoboTwin client
appends to the observation dict.

This lets one server use the task statistics stored with its checkpoint. The
``assets/`` folder under a checkpoint typically contains one
``norm_stats.json`` per ``<task>-<task_config>-<seed>`` subdirectory, and the
correct entry must be selected at inference time because state/action
distributions differ across tasks.

The websocket server calls ``infer`` synchronously on its event-loop thread.
Calls to this wrapper must remain serial because they mutate the underlying
``Normalize``/``Unnormalize`` instances.
"""

from __future__ import annotations

from collections.abc import Sequence
import logging
import os
from typing import Any

from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.policies import policy as _policy
from openpi.shared import normalize as _normalize

_TASK_ID_KEYS = ("__robotwin_task_id__", "task_id", "task_name")


def _find_norm_transforms(policy: _policy.Policy) -> tuple[_transforms.Normalize, _transforms.Unnormalize]:
    """Locate the (single) Normalize/Unnormalize instance in the policy pipeline."""

    def _find_one(composite, cls):
        candidates = [t for t in composite.transforms if isinstance(t, cls)]
        if not candidates:
            raise RuntimeError(f"Policy pipeline does not contain a {cls.__name__} transform.")
        if len(candidates) > 1:
            raise RuntimeError(f"Policy pipeline contains multiple {cls.__name__} transforms; ambiguous.")
        return candidates[0]

    return (
        _find_one(policy._input_transform, _transforms.Normalize),  # noqa: SLF001
        _find_one(policy._output_transform, _transforms.Unnormalize),  # noqa: SLF001
    )


def _scan_norm_stats(assets_dir: str) -> dict[str, dict[str, _normalize.NormStats]]:
    """Recursively scan ``assets_dir`` and load all per-task norm_stats.

    Returns a mapping from asset_id (relative path under ``assets_dir``) to its
    ``{state, actions} -> NormStats`` dict.
    """
    out: dict[str, dict[str, _normalize.NormStats]] = {}
    if not os.path.isdir(assets_dir):
        return out
    for root, _dirs, files in os.walk(assets_dir):
        if "norm_stats.json" not in files:
            continue
        asset_id = os.path.relpath(root, assets_dir)
        try:
            stats = _normalize.load(root)
        except FileNotFoundError:
            continue
        out[asset_id] = stats
    return out


def _build_task_index(norm_stats_map: dict[str, dict[str, _normalize.NormStats]]) -> dict[str, list[str]]:
    """Build a fast lookup from task base-name to candidate asset ids.

    asset_id is typically ``<prefix>/<task_name>-<task_config>_collect_<n>-<seed>``;
    we index by the trailing ``<task_name>-<task_config>...`` directory name and
    by just ``<task_name>`` so a client can route by task name alone.
    """
    index: dict[str, list[str]] = {}
    for asset_id in sorted(norm_stats_map):
        leaf = os.path.basename(asset_id)
        index.setdefault(leaf, []).append(asset_id)
        task_part = leaf.split("-", 1)[0]
        if task_part and task_part != leaf:
            index.setdefault(task_part, []).append(asset_id)
    return index


class RobotwinPolicy(_base_policy.BasePolicy):
    """Routes per-task ``norm_stats`` into a single underlying Policy."""

    def __init__(
        self,
        base_policy: _policy.Policy,
        norm_stats_map: dict[str, dict[str, _normalize.NormStats]],
        *,
        default_asset_id: str | None = None,
        prefer_task_config: str | None = None,
    ):
        if not norm_stats_map:
            raise ValueError("RobotwinPolicy requires at least one norm_stats entry.")
        self._policy = base_policy
        self._norm_stats_map = norm_stats_map
        self._task_index = _build_task_index(norm_stats_map)
        self._prefer_task_config = prefer_task_config
        self._normalize, self._unnormalize = _find_norm_transforms(base_policy)
        # Track which asset id is currently bound to the (mutable) transforms.
        self._current_asset_id: str | None = None
        if default_asset_id is None:
            default_asset_id = next(iter(norm_stats_map))
        self._apply_asset_id(default_asset_id)
        logging.info(
            "RobotwinPolicy initialised with %d tasks (default=%s).",
            len(norm_stats_map),
            default_asset_id,
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return self._policy.metadata

    @property
    def available_asset_ids(self) -> Sequence[str]:
        return tuple(self._norm_stats_map.keys())

    def _apply_asset_id(self, asset_id: str) -> None:
        if asset_id == self._current_asset_id:
            return
        stats = self._norm_stats_map[asset_id]
        # ``Normalize`` / ``Unnormalize`` are frozen dataclasses; bypass with object.__setattr__.
        object.__setattr__(self._normalize, "norm_stats", stats)
        object.__setattr__(self._unnormalize, "norm_stats", stats)
        self._current_asset_id = asset_id
        logging.info("Switched RobotWin norm_stats to asset_id=%s", asset_id)

    def _resolve_asset_id(self, hint: str, task_config: str | None = None) -> str | None:
        # An explicit asset ID takes precedence over task-name routing.
        if hint in self._norm_stats_map:
            return hint
        # Indexed lookup (task name, task-config combo, etc.).
        candidates = self._task_index.get(hint)
        if not candidates:
            return None
        preference = task_config or self._prefer_task_config
        if preference:
            scene_mode = {"demo_clean": "clean", "demo_randomized": "randomized"}.get(preference)
            if scene_mode is None:
                scored = [c for c in candidates if preference in c]
            else:
                def matches_scene_mode(asset_id: str) -> bool:
                    suffix = os.path.basename(asset_id).partition("-")[2]
                    tokens = suffix.replace("-", "_").split("_")
                    declared_modes = {"clean", "randomized"}.intersection(tokens)
                    if declared_modes:
                        return declared_modes == {scene_mode}
                    # Some task assets use the scene directory with a bare
                    # task name; embodiment-specific suffixes retain the mode.
                    parent_mode = {"clean": "clean", "aug": "randomized"}.get(
                        os.path.basename(os.path.dirname(asset_id))
                    )
                    return parent_mode == scene_mode

                scored = [c for c in candidates if matches_scene_mode(c)]
            if scored:
                candidates = scored
            else:
                raise ValueError(f"No normalization assets for task {hint!r} with task_config={preference!r}.")
        if len(candidates) != 1:
            raise ValueError(f"Ambiguous normalization assets for task {hint!r}; supply ROBOTWIN_ASSET_ID.")
        return candidates[0]

    @override
    def infer(self, obs: dict, *, noise=None) -> dict:  # type: ignore[misc]
        obs = dict(obs)
        hint: str | None = None
        for key in _TASK_ID_KEYS:
            if key in obs:
                value = obs.pop(key)
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="ignore")
                if value:
                    hint = str(value)
                    break
        task_config = obs.pop("__robotwin_task_config__", None)

        if hint is not None:
            asset_id = self._resolve_asset_id(hint, task_config=task_config)
            if asset_id is None:
                raise ValueError(f"No normalization assets match RoboTwin task {hint!r}.")
            else:
                self._apply_asset_id(asset_id)

        debug = os.environ.get("ROBOTWIN_INFER_DEBUG") == "1"
        if debug:
            import numpy as _np

            state = obs.get("state")
            if state is not None:
                arr = _np.asarray(state).reshape(-1)
                logging.info(
                    "INFER state[16]=%s task=%s asset=%s",
                    _np.array2string(arr[:16], precision=3, suppress_small=True),
                    hint,
                    self._current_asset_id,
                )

        if noise is None:
            result = self._policy.infer(obs)
        else:
            result = self._policy.infer(obs, noise=noise)

        if debug:
            import numpy as _np

            actions = result.get("actions")
            if actions is not None:
                arr = _np.asarray(actions)
                first = arr[0] if arr.ndim >= 2 else arr
                logging.info(
                    "INFER action[0,:16]=%s    shape=%s",
                    _np.array2string(first.reshape(-1)[:16], precision=3, suppress_small=True),
                    arr.shape,
                )
        return result
