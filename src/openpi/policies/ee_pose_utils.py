from __future__ import annotations

import numpy as np


def collapse_opposing_gripper_fingers(fingers: np.ndarray) -> np.ndarray:
    fingers = np.asarray(fingers, dtype=np.float32)
    if fingers.shape[-1] != 2:
        raise ValueError(f"Expected two-finger gripper state, got shape {fingers.shape}.")
    return 0.5 * (fingers[..., 0] - fingers[..., 1])


def rotation_vector_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float32)
    if rotation.shape[-1] != 3:
        raise ValueError(f"Expected axis-angle rotation vector with shape [..., 3], got {rotation.shape}.")

    angle = np.linalg.norm(rotation, axis=-1, keepdims=True)
    half_angle = 0.5 * angle
    sin_half = np.sin(half_angle)

    axis = np.zeros_like(rotation, dtype=np.float32)
    np.divide(rotation, angle, out=axis, where=angle > 1e-8)

    quat = np.concatenate([np.cos(half_angle), axis * sin_half], axis=-1)
    small_angle = (angle[..., 0] <= 1e-8)
    if np.any(small_angle):
        quat = quat.copy()
        quat[small_angle] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return quat.astype(np.float32)


def quaternion_to_rotation_vector(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float32)
    if quaternion.shape[-1] != 4:
        raise ValueError(f"Expected quaternion with shape [..., 4], got {quaternion.shape}.")

    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    normalized = np.divide(
        quaternion,
        norm,
        out=np.zeros_like(quaternion, dtype=np.float32),
        where=norm > 1e-8,
    )
    w = np.clip(normalized[..., :1], -1.0, 1.0)
    xyz = normalized[..., 1:]
    xyz_norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(xyz_norm, w)

    axis = np.zeros_like(xyz, dtype=np.float32)
    np.divide(xyz, xyz_norm, out=axis, where=xyz_norm > 1e-8)
    rotation = axis * angle

    small_angle = (xyz_norm[..., 0] <= 1e-8)
    if np.any(small_angle):
        rotation = rotation.copy()
        rotation[small_angle] = 0.0
    return rotation.astype(np.float32)


def normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float32)
    if quaternion.shape[-1] != 4:
        raise ValueError(f"Expected quaternion with shape [..., 4], got {quaternion.shape}.")
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    normalized = np.divide(
        quaternion,
        norm,
        out=np.zeros_like(quaternion, dtype=np.float32),
        where=norm > 1e-8,
    )
    invalid = norm[..., 0] <= 1e-8
    if np.any(invalid):
        normalized = normalized.copy()
        normalized[invalid] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return normalized.astype(np.float32)


def canonicalize_quaternion_sign(quaternion: np.ndarray) -> np.ndarray:
    quaternion = normalize_quaternion(quaternion)
    sign = np.where(quaternion[..., :1] < 0.0, -1.0, 1.0).astype(np.float32)
    return (quaternion * sign).astype(np.float32)


def align_quaternion_sign(reference: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    reference = normalize_quaternion(reference)
    quaternion = normalize_quaternion(quaternion)
    sign = np.where(np.sum(reference * quaternion, axis=-1, keepdims=True) < 0.0, -1.0, 1.0).astype(np.float32)
    return (quaternion * sign).astype(np.float32)


def quaternion_conjugate(quaternion: np.ndarray) -> np.ndarray:
    quaternion = normalize_quaternion(quaternion)
    conjugate = quaternion.copy()
    conjugate[..., 1:] *= -1.0
    return conjugate.astype(np.float32)


def quaternion_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lhs = normalize_quaternion(lhs)
    rhs = normalize_quaternion(rhs)
    lw, lx, ly, lz = np.moveaxis(lhs, -1, 0)
    rw, rx, ry, rz = np.moveaxis(rhs, -1, 0)
    product = np.stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        axis=-1,
    )
    return canonicalize_quaternion_sign(product)


def quaternion_delta(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    current = canonicalize_quaternion_sign(current)
    target = align_quaternion_sign(current, target)
    return quaternion_multiply(quaternion_conjugate(current), target)


def quaternion_apply_delta(current: np.ndarray, delta: np.ndarray) -> np.ndarray:
    current = canonicalize_quaternion_sign(current)
    delta = canonicalize_quaternion_sign(delta)
    return quaternion_multiply(current, delta)
