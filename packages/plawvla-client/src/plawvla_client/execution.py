"""Lightweight helpers for executing canonical end-effector trajectories.

Canonical poses are flattened per arm as ``xyz + quaternion(wxyz) +
open_fraction``.  The functions in this module intentionally depend only on
NumPy and the Python typing/dataclass libraries so they can be used by thin
robot clients without importing the training stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from typing_extensions import Literal, Protocol, runtime_checkable


CANONICAL_ARM_DIM = 8
ROBOSUITE_OSC_ARM_DIM = 7
LIBERO_GRIPPER_WIDTH_SCALE = 0.04
_QUATERNION_EPS = 1e-8
TimedSampleSelection = Literal["at_or_before", "nearest"]


def _finite_array(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _quaternion(value: np.ndarray, *, name: str = "quaternion") -> np.ndarray:
    quaternion = _finite_array(value, name=name)
    if quaternion.ndim == 0 or quaternion.shape[-1] != 4:
        raise ValueError(f"{name} must have shape [..., 4], got {quaternion.shape}.")
    return quaternion


def normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    """Normalize one or more ``wxyz`` quaternions.

    Zero (or numerically zero) quaternions are rejected rather than silently
    replaced with an identity rotation.
    """

    quaternion = _quaternion(quaternion)
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm <= _QUATERNION_EPS):
        raise ValueError("Quaternion norm must be greater than zero.")
    return quaternion / norm


def quaternion_conjugate(quaternion: np.ndarray) -> np.ndarray:
    """Return the conjugate of normalized ``wxyz`` quaternion(s)."""

    result = normalize_quaternion(quaternion).copy()
    result[..., 1:] *= -1.0
    return result


def quaternion_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Hamilton product of broadcast-compatible ``wxyz`` quaternions."""

    lhs = normalize_quaternion(lhs)
    rhs = normalize_quaternion(rhs)
    try:
        lhs, rhs = np.broadcast_arrays(lhs, rhs)
    except ValueError as error:
        raise ValueError(
            f"Quaternion shapes {lhs.shape} and {rhs.shape} are not broadcast-compatible."
        ) from error

    lw, lx, ly, lz = np.moveaxis(lhs, -1, 0)
    rw, rx, ry, rz = np.moveaxis(rhs, -1, 0)
    product = np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )
    return normalize_quaternion(product)


def quaternion_delta(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the shortest local rotation taking ``current`` to ``target``.

    The result follows ``conjugate(current) * target``.  Antipodal target
    quaternions are aligned with the current quaternion before multiplication.
    """

    current = normalize_quaternion(current)
    target = normalize_quaternion(target)
    try:
        current, target = np.broadcast_arrays(current, target)
    except ValueError as error:
        raise ValueError(
            f"Quaternion shapes {current.shape} and {target.shape} are not broadcast-compatible."
        ) from error
    target = np.where(np.sum(current * target, axis=-1, keepdims=True) < 0.0, -target, target)
    delta = quaternion_multiply(quaternion_conjugate(current), target)
    return np.where(delta[..., :1] < 0.0, -delta, delta)


def quaternion_to_rotvec(quaternion: np.ndarray) -> np.ndarray:
    """Convert normalized ``wxyz`` quaternion(s) to shortest rotation vectors."""

    quaternion = normalize_quaternion(quaternion)
    quaternion = np.where(quaternion[..., :1] < 0.0, -quaternion, quaternion)
    w = np.clip(quaternion[..., :1], -1.0, 1.0)
    xyz = quaternion[..., 1:]
    sin_half = np.linalg.norm(xyz, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(sin_half, w)
    scale = np.divide(angle, sin_half, out=np.full_like(angle, 2.0), where=sin_half > _QUATERNION_EPS)
    return xyz * scale


def rotvec_to_quaternion(rotation_vector: np.ndarray) -> np.ndarray:
    """Convert rotation vector(s) to normalized ``wxyz`` quaternion(s)."""

    rotation_vector = _finite_array(rotation_vector, name="rotation_vector")
    if rotation_vector.ndim == 0 or rotation_vector.shape[-1] != 3:
        raise ValueError(
            f"rotation_vector must have shape [..., 3], got {rotation_vector.shape}."
        )
    angle = np.linalg.norm(rotation_vector, axis=-1, keepdims=True)
    half_angle = 0.5 * angle
    scale = np.divide(
        np.sin(half_angle),
        angle,
        out=np.full_like(angle, 0.5),
        where=angle > _QUATERNION_EPS,
    )
    return normalize_quaternion(
        np.concatenate((np.cos(half_angle), rotation_vector * scale), axis=-1)
    )


def quaternion_slerp(start: np.ndarray, end: np.ndarray, fraction: np.ndarray) -> np.ndarray:
    """Spherically interpolate between broadcast-compatible quaternions.

    Antipodal inputs use the same short path.  ``fraction`` may be a scalar or
    an array broadcast-compatible with the quaternion batch dimensions.
    """

    start = normalize_quaternion(start)
    end = normalize_quaternion(end)
    try:
        start, end = np.broadcast_arrays(start, end)
    except ValueError as error:
        raise ValueError(
            f"Quaternion shapes {start.shape} and {end.shape} are not broadcast-compatible."
        ) from error

    fraction = _finite_array(fraction, name="fraction")
    try:
        fraction = np.broadcast_to(fraction, start.shape[:-1])[..., None]
    except ValueError as error:
        raise ValueError(
            f"fraction shape {fraction.shape} is not compatible with quaternion batch shape "
            f"{start.shape[:-1]}."
        ) from error

    dot = np.sum(start * end, axis=-1, keepdims=True)
    end = np.where(dot < 0.0, -end, end)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    angle = np.arccos(dot)
    sin_angle = np.sin(angle)
    spherical = (
        np.divide(
            np.sin((1.0 - fraction) * angle),
            sin_angle,
            out=np.zeros_like(angle),
            where=sin_angle > _QUATERNION_EPS,
        )
        * start
        + np.divide(
            np.sin(fraction * angle),
            sin_angle,
            out=np.zeros_like(angle),
            where=sin_angle > _QUATERNION_EPS,
        )
        * end
    )
    linear = (1.0 - fraction) * start + fraction * end
    return normalize_quaternion(np.where(sin_angle > _QUATERNION_EPS, spherical, linear))


def libero_state_to_canonical(
    raw_state: np.ndarray,
    *,
    arm_count: Optional[int] = None,
    width_scale: float = LIBERO_GRIPPER_WIDTH_SCALE,
) -> np.ndarray:
    """Convert flattened LIBERO EEF state(s) to canonical 8D-per-arm poses.

    Each raw arm contains ``xyz + rotation_vector + two opposing fingers``.
    Opposing finger positions are collapsed as ``(left - right) / 2`` and
    divided by ``width_scale`` to produce a clipped open fraction.
    """

    raw_state = _finite_array(raw_state, name="raw_state")
    if raw_state.ndim == 0 or raw_state.shape[-1] == 0 or raw_state.shape[-1] % 8:
        raise ValueError(
            f"raw_state must have a non-empty flattened [..., arm_count * 8] shape, got "
            f"{raw_state.shape}."
        )
    inferred_arm_count = raw_state.shape[-1] // 8
    if arm_count is not None:
        if not isinstance(arm_count, int) or isinstance(arm_count, bool) or arm_count <= 0:
            raise ValueError("arm_count must be a positive integer.")
        if inferred_arm_count != arm_count:
            raise ValueError(
                f"raw_state has {inferred_arm_count} arms, expected arm_count={arm_count}."
            )
    if not np.isfinite(width_scale) or width_scale <= 0.0:
        raise ValueError("width_scale must be finite and positive.")

    arms = raw_state.reshape(raw_state.shape[:-1] + (inferred_arm_count, 8))
    width = 0.5 * (arms[..., 6:7] - arms[..., 7:8])
    canonical = np.concatenate(
        (
            arms[..., :3],
            rotvec_to_quaternion(arms[..., 3:6]),
            np.clip(width / width_scale, 0.0, 1.0),
        ),
        axis=-1,
    )
    return canonical.reshape(raw_state.shape[:-1] + (inferred_arm_count * CANONICAL_ARM_DIM,))


# A descriptive alias for callers that use "canonicalize" terminology.
canonicalize_libero_state = libero_state_to_canonical


def _validate_arm_count(arm_count: int) -> None:
    if not isinstance(arm_count, int) or isinstance(arm_count, bool) or arm_count <= 0:
        raise ValueError("arm_count must be a positive integer.")


def _canonical_arms(value: np.ndarray, *, arm_count: int, name: str) -> np.ndarray:
    value = _finite_array(value, name=name)
    expected_shape = (arm_count * CANONICAL_ARM_DIM,)
    if value.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {value.shape}.")
    arms = value.reshape(arm_count, CANONICAL_ARM_DIM).copy()
    arms[:, 3:7] = normalize_quaternion(arms[:, 3:7])
    return arms


class TimedValueBuffer:
    """Store timestamped array values and sample a relative-time window.

    ``selection="at_or_before"`` (the default) is causal: for each target
    ``query_time_s + offset_s`` it chooses the newest sample whose timestamp is
    not later than the target. ``selection="nearest"`` chooses the sample with
    minimum absolute timestamp error; exact ties choose the earlier sample.

    If a target predates the earliest available sample, both rules repeat that
    earliest value and mark the corresponding ``is_pad`` entry true. This is
    the intended episode-start behavior. Targets at or after the first
    available timestamp are never marked as padding, even if the latest value
    is reused because the query is slightly newer than the buffer.

    Values may have any NumPy-compatible dtype and shape, but all values in one
    buffer must have the same shape. Timestamps must be finite and strictly
    increasing. ``sample_offsets`` returns ``(values, is_pad)`` with the
    requested offsets kept in their supplied order.
    """

    def __init__(self, selection: TimedSampleSelection = "at_or_before") -> None:
        self._selection = self._validate_selection(selection)
        self._timestamps: List[float] = []
        self._values: List[np.ndarray] = []
        self._value_shape: Optional[tuple] = None

    @staticmethod
    def _validate_selection(selection: str) -> TimedSampleSelection:
        if selection not in ("at_or_before", "nearest"):
            raise ValueError(
                "selection must be either 'at_or_before' or 'nearest', "
                f"got {selection!r}."
            )
        return selection

    def __len__(self) -> int:
        return len(self._values)

    @property
    def selection(self) -> TimedSampleSelection:
        return self._selection

    def clear(self) -> None:
        """Remove all samples, starting a fresh episode."""

        self._timestamps.clear()
        self._values.clear()
        self._value_shape = None

    def append(self, value: np.ndarray, timestamp_s: float) -> None:
        """Append one value at a finite, strictly increasing timestamp."""

        timestamp = _finite_array(timestamp_s, name="timestamp_s")
        if timestamp.shape != ():
            raise ValueError(f"timestamp_s must be a scalar, got shape {timestamp.shape}.")
        timestamp_value = float(timestamp)
        if self._timestamps and timestamp_value <= self._timestamps[-1]:
            raise ValueError("timestamp_s must be strictly greater than the previous timestamp.")

        array = np.asarray(value)
        if self._value_shape is not None and array.shape != self._value_shape:
            raise ValueError(
                f"value shape must remain {self._value_shape}, got {array.shape}."
            )
        if self._value_shape is None:
            self._value_shape = array.shape
        self._timestamps.append(timestamp_value)
        self._values.append(array.copy())

    def sample_offsets(
        self,
        query_time_s: float,
        offsets_s: np.ndarray,
        *,
        selection: Optional[TimedSampleSelection] = None,
    ) -> tuple:
        """Sample values at ``query_time_s + offsets_s``.

        The result is ``(values, is_pad)``. ``values`` has shape
        ``[len(offsets_s), *value_shape]`` and ``is_pad`` is a boolean vector.
        See the class docstring for the two deterministic selection rules.
        """

        if not self._values:
            raise ValueError("Cannot sample an empty timed value buffer.")
        query_time = _finite_array(query_time_s, name="query_time_s")
        if query_time.shape != ():
            raise ValueError(f"query_time_s must be a scalar, got shape {query_time.shape}.")
        offsets = _finite_array(offsets_s, name="offsets_s")
        if offsets.ndim != 1 or offsets.size == 0:
            raise ValueError("offsets_s must be a non-empty one-dimensional array.")
        rule = self._selection if selection is None else self._validate_selection(selection)

        timestamps = np.asarray(self._timestamps, dtype=np.float64)
        target_times = float(query_time) + offsets
        is_pad = target_times < timestamps[0]
        if rule == "at_or_before":
            # Include a tiny roundoff allowance so arithmetic such as
            # ``1.0 + (-0.8)`` still matches a sample timestamped at ``0.2``.
            time_scale = np.maximum(1.0, np.abs(target_times))
            inclusive_targets = target_times + 4.0 * np.finfo(np.float64).eps * time_scale
            indices = np.searchsorted(timestamps, inclusive_targets, side="right") - 1
            indices = np.clip(indices, 0, timestamps.size - 1)
        else:
            right = np.searchsorted(timestamps, target_times, side="left")
            right = np.clip(right, 0, timestamps.size - 1)
            left = np.clip(right - 1, 0, timestamps.size - 1)
            left_error = np.abs(target_times - timestamps[left])
            right_error = np.abs(timestamps[right] - target_times)
            error_scale = np.maximum(
                1.0,
                np.maximum(
                    np.abs(target_times),
                    np.maximum(np.abs(timestamps[left]), np.abs(timestamps[right])),
                ),
            )
            tie_tolerance = 4.0 * np.finfo(np.float64).eps * error_scale
            # Prefer the right sample only when it is meaningfully closer.
            # Numerically equal-distance ties therefore remain deterministic
            # and causal by selecting the earlier (left) sample.
            indices = np.where(right_error < left_error - tie_tolerance, right, left)

        sampled = np.stack([self._values[int(index)] for index in indices], axis=0)
        return sampled, is_pad.astype(bool)


class TimedHistoryBuffer(TimedValueBuffer):
    """History-oriented name for :class:`TimedValueBuffer`."""


@runtime_checkable
class ExecutionAdapter(Protocol):
    """Protocol for converting canonical targets to robot commands."""

    arm_count: int

    def adapt(self, current_pose: np.ndarray, target_pose: np.ndarray) -> np.ndarray:
        """Convert live canonical state and a canonical target to a command."""


@dataclass(frozen=True)
class AbsoluteEefTrajectory:
    """Timed canonical absolute EEF targets with interpolation."""

    target_times: np.ndarray
    targets: np.ndarray
    arm_count: int = 1
    initial_pose: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        _validate_arm_count(self.arm_count)
        target_times = _finite_array(self.target_times, name="target_times")
        targets = _finite_array(self.targets, name="targets")
        if target_times.ndim != 1 or target_times.size == 0:
            raise ValueError("target_times must be a non-empty one-dimensional array.")
        if np.any(target_times <= 0.0):
            raise ValueError("target_times must be positive.")
        if np.any(np.diff(target_times) <= 0.0):
            raise ValueError("target_times must be strictly increasing.")
        expected_shape = (target_times.size, self.arm_count * CANONICAL_ARM_DIM)
        if targets.shape != expected_shape:
            raise ValueError(f"targets must have shape {expected_shape}, got {targets.shape}.")

        targets = targets.copy()
        arms = targets.reshape(target_times.size, self.arm_count, CANONICAL_ARM_DIM)
        arms[..., 3:7] = normalize_quaternion(arms[..., 3:7])
        initial_pose = (
            None
            if self.initial_pose is None
            else _canonical_arms(self.initial_pose, arm_count=self.arm_count, name="initial_pose").reshape(-1)
        )
        object.__setattr__(self, "target_times", target_times.copy())
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "initial_pose", initial_pose)

    def sample(self, time: float) -> np.ndarray:
        """Sample at relative ``time``, clamping outside the target interval."""

        time_array = _finite_array(time, name="time")
        if time_array.shape != ():
            raise ValueError(f"time must be a scalar, got shape {time_array.shape}.")
        time_value = float(time_array)
        if time_value <= self.target_times[0]:
            if self.initial_pose is None or time_value >= self.target_times[0]:
                return self.targets[0].copy()
            if time_value <= 0.0:
                return self.initial_pose.copy()
            return _interpolate_canonical_pose(
                self.initial_pose,
                self.targets[0],
                time_value / self.target_times[0],
                self.arm_count,
            )
        if time_value >= self.target_times[-1]:
            return self.targets[-1].copy()

        upper = int(np.searchsorted(self.target_times, time_value, side="right"))
        lower = upper - 1
        fraction = (time_value - self.target_times[lower]) / (
            self.target_times[upper] - self.target_times[lower]
        )
        return _interpolate_canonical_pose(
            self.targets[lower],
            self.targets[upper],
            fraction,
            self.arm_count,
        )

    def __call__(self, time: float) -> np.ndarray:
        return self.sample(time)


def _interpolate_canonical_pose(
    start_pose: np.ndarray,
    end_pose: np.ndarray,
    fraction: float,
    arm_count: int,
) -> np.ndarray:
    start = _canonical_arms(start_pose, arm_count=arm_count, name="start_pose")
    end = _canonical_arms(end_pose, arm_count=arm_count, name="end_pose")
    sampled = np.empty_like(start)
    sampled[:, :3] = start[:, :3] + fraction * (end[:, :3] - start[:, :3])
    sampled[:, 3:7] = quaternion_slerp(start[:, 3:7], end[:, 3:7], fraction)
    sampled[:, 7] = start[:, 7] + fraction * (end[:, 7] - start[:, 7])
    return sampled.reshape(arm_count * CANONICAL_ARM_DIM)


@dataclass(frozen=True)
class RobosuiteOSCAdapter:
    """Convert canonical absolute targets to normalized robosuite OSC commands."""

    arm_count: int = 1
    position_scale: float = 0.05
    rotation_scale: float = 0.5

    def __post_init__(self) -> None:
        _validate_arm_count(self.arm_count)
        if not np.isfinite(self.position_scale) or self.position_scale <= 0.0:
            raise ValueError("position_scale must be finite and positive.")
        if not np.isfinite(self.rotation_scale) or self.rotation_scale <= 0.0:
            raise ValueError("rotation_scale must be finite and positive.")

    def adapt(self, current_pose: np.ndarray, target_pose: np.ndarray) -> np.ndarray:
        """Create clipped OSC commands from the supplied *live* canonical pose."""

        current = _canonical_arms(current_pose, arm_count=self.arm_count, name="current_pose")
        target = _canonical_arms(target_pose, arm_count=self.arm_count, name="target_pose")
        position = np.clip(
            (target[:, :3] - current[:, :3]) / self.position_scale, -1.0, 1.0
        )
        rotation = np.clip(
            quaternion_to_rotvec(quaternion_delta(current[:, 3:7], target[:, 3:7]))
            / self.rotation_scale,
            -1.0,
            1.0,
        )
        # Canonical gripper values are open-high; robosuite commands are close-high.
        gripper = np.where(target[:, 7:8] >= 0.5, -1.0, 1.0)
        return np.concatenate((position, rotation, gripper), axis=-1).reshape(
            self.arm_count * ROBOSUITE_OSC_ARM_DIM
        )

    def command(self, current_pose: np.ndarray, target_pose: np.ndarray) -> np.ndarray:
        """Alias for :meth:`adapt`."""

        return self.adapt(current_pose, target_pose)

    def __call__(self, current_pose: np.ndarray, target_pose: np.ndarray) -> np.ndarray:
        return self.adapt(current_pose, target_pose)
