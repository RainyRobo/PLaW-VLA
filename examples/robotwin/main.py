# Adapted from RoboTwin (Copyright (c) 2025 Tianxing Chen; MIT) and
# LingBot-VLA (Copyright 2026 Robbyant Team; Apache-2.0).
# Modified for PLaW-VLA by the PLaW-VLA authors, 2026.
# See LICENSE, NOTICE, and LICENSES/MIT-RoboTwin.txt.
# Simulator paths and rendering settings must precede third-party imports.
# ruff: noqa: E402
import dataclasses
import os
from pathlib import Path
import sys

import cv2

# Project layout:
#   <repo>/examples/robotwin/          <- this file
#   <repo>/third_party/robotwin/       <- RoboTwin codebase (envs, task_config, description, ...)
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_CALLER_DIR = Path.cwd()
robotwin_root = _PROJECT_ROOT / "third_party" / "robotwin"
if not robotwin_root.is_dir():
    raise FileNotFoundError(
        f"RobotWin checkout not found at {robotwin_root}; make sure third_party/robotwin/ is initialised."
    )

# RoboTwin tasks resolve ``task_config/*.yml`` and simulator resources from
# the upstream checkout's root.
if str(robotwin_root) not in sys.path:
    sys.path.insert(0, str(robotwin_root))
os.chdir(robotwin_root)

# Sibling helpers live in the same directory as this script. Put it on sys.path so
# we can import them as top-level modules (no ``examples.`` package prefix needed).
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import sapien

# SAPIEN 3.0.0b1 bundles an OIDN build that does not support all GPUs.
# Preserve upstream rendering and allow selecting a compatible denoiser.
if "ROBOTWIN_DENOISER" in os.environ:
    _set_denoiser = sapien.render.set_ray_tracing_denoiser
    sapien.render.set_ray_tracing_denoiser = lambda _: _set_denoiser(os.environ["ROBOTWIN_DENOISER"])

import argparse
from collections import deque
from datetime import datetime
from datetime import timezone
import importlib
import json
import math
import re
import traceback

import imageio
import numpy as np
from plawvla_client.execution import ExecutionAdapter
from plawvla_client.execution import normalize_quaternion
from plawvla_client.policy_contract import parse_policy_contract
from plawvla_client.websocket_client_policy import WebsocketClientPolicy
import yaml

# ---------------------------------------------------------------------------
# PLaW-VLA RoboTwin server contract
# ---------------------------------------------------------------------------
# Canonical bimanual EEF layout expected by this client.
#
#   left_xyz, left_quat(qw, qx, qy, qz), left_gripper,    -> 8 floats
#   right_xyz, right_quat(qw, qx, qy, qz), right_gripper. -> 8 floats
#                                                  total = 16 floats
#
# Audited against the pinned RoboTwin commit 2eeec322:
# * get_obs()["endpose"] and take_action(..., action_type="ee") use the same
#   world-frame gripper-centre pose;
# * SAPIEN/transforms3d quaternions are scalar-first (wxyz);
# * gripper values are normalized [0, 1], with 1=open.
#
# RoboTwin's EE action is already an absolute IK/planner target.  It must not
# pass through the robosuite/LIBERO OSC adapter or its 0.05 m / 0.5 rad scales.
_EE_QUAT_SLICES = (slice(3, 7), slice(11, 15))  # left arm quat, right arm quat
_EE_BIMANUAL_DIM = 16
_ROBOTWIN_EXECUTION_TIMING = "one_target_per_take_action"

@dataclasses.dataclass(frozen=True)
class ServerContract:
    """Validated action, camera, and temporal layout of the policy server.

    The client reads ``policy_metadata`` and ``input_spec`` before simulation
    begins. Camera keys and history offsets determine the observation payload;
    the action layout determines how returned targets are sent to the robot.
    """

    policy_config_name: str
    action_type: str
    native_action_dim: int
    action_horizon: int  # max chunk length the model emits per inference
    family: str
    image_keys: tuple[str, ...]  # all camera keys we must include in the payload
    temporal_image_key: str  # the single camera the model consumes as history
    state_key: str  # payload key for the 16-D EE state
    prompt_key: str  # payload key for the language instruction
    history_step_offsets: tuple[int, ...]
    future_step_offsets: tuple[int, ...]
    action_target_time_offsets_s: tuple[float, ...]
    execution_timing: str

    @property
    def history_buffer_capacity(self) -> int:
        return _history_buffer_capacity(self.history_step_offsets)


def parse_server_contract(metadata: dict, *, execution_timing: str) -> ServerContract:
    """Validate server metadata and return the corresponding observation layout.

    Raises ``RuntimeError`` for unsupported actions, cameras, or temporal keys.
    """
    if not isinstance(metadata, dict):
        raise RuntimeError(f"Server metadata must be a dict, got {type(metadata).__name__}.")

    try:
        policy_contract = parse_policy_contract(metadata)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "RoboTwin requires the explicit canonical absolute-EEF wire contract. "
            "The server must publish output_action_dim plus input_spec fields for "
            "absolute_eef_target, wxyz, open_fraction, two arms, and time offsets."
        ) from exc

    action_type = metadata.get("robotwin_action_type")
    if action_type != "ee":
        raise RuntimeError(
            "This client targets the EE-action server (e.g. stage3_finetuning_robotwin); "
            f"server published robotwin_action_type={action_type!r}. "
            "Re-launch the server with an EE-action checkpoint, or extend this client to "
            "support the qpos space (currently unimplemented)."
        )
    native_action_dim = int(metadata.get("robotwin_native_action_dim") or 0)
    if native_action_dim != _EE_BIMANUAL_DIM:
        raise RuntimeError(
            f"Server published robotwin_native_action_dim={native_action_dim}, but this "
            f"client only knows how to serialize the canonical bimanual EE pose layout "
            f"({_EE_BIMANUAL_DIM}-D = 2 arms x [xyz, qwqxqyqz, grip]). "
            "Update _encode_robotwin_ee_state / normalize_robotwin_ee_action to match."
        )

    action_horizon = int(metadata.get("action_horizon") or 0)
    if action_horizon <= 0:
        raise RuntimeError(
            f"Server published an invalid action_horizon={action_horizon!r}; expected a positive integer."
        )

    if policy_contract.output_action_dim != native_action_dim:
        raise RuntimeError(
            f"Server output_action_dim={policy_contract.output_action_dim} does not match "
            f"robotwin_native_action_dim={native_action_dim}; padded model dimensions "
            "must never leak onto the execution wire."
        )
    if policy_contract.action_arm_count != 2:
        raise RuntimeError(
            f"RoboTwin requires two canonical arms, got action_arm_count={policy_contract.action_arm_count}."
        )
    if policy_contract.state_gripper_format != "open_fraction":
        raise RuntimeError(
            "RoboTwin observations expose normalized open-high gripper fractions; "
            f"server published state_gripper_format={policy_contract.state_gripper_format!r}."
        )

    family = policy_contract.family
    if family != "aloha":
        raise RuntimeError(
            f"Expected an 'aloha' family server (RobotWin reuses the Aloha policy "
            f"transform); server published family={family!r}."
        )

    image_keys = policy_contract.image_keys
    temporal_image_keys = policy_contract.temporal_image_keys
    if len(temporal_image_keys) > 1:
        raise RuntimeError(
            f"This client only knows how to maintain temporal history for a single "
            f"camera; server published {len(temporal_image_keys)} temporal_image_keys="
            f"{temporal_image_keys}. Extend make_policy_obs / eval_policy to keep one "
            "deque per temporal key."
        )

    state_key = policy_contract.state_key
    prompt_key = policy_contract.prompt_key
    if not isinstance(state_key, str):
        raise RuntimeError(
            f"Server input_spec is missing 'state_key' (got {state_key!r}); "
            "this client cannot send observations without it."
        )
    if not isinstance(prompt_key, str):
        raise RuntimeError(f"Server input_spec is missing 'prompt_key' (got {prompt_key!r}).")

    action_target_time_offsets_s = policy_contract.action_target_time_offsets_s
    if len(action_target_time_offsets_s) != action_horizon:
        raise RuntimeError(
            "action_target_time_offsets_s must exactly match the action horizon; "
            f"got {len(action_target_time_offsets_s)} offsets for horizon {action_horizon}."
        )
    action_period_s = action_target_time_offsets_s[0]
    if not math.isfinite(action_period_s) or action_period_s <= 0:
        raise RuntimeError("The first action target time must be finite and positive.")
    expected_action_times = tuple(action_period_s * step for step in range(1, action_horizon + 1))
    if any(
        not math.isclose(actual, expected, abs_tol=1e-6)
        for actual, expected in zip(action_target_time_offsets_s, expected_action_times)
    ):
        raise RuntimeError(
            "one_target_per_take_action requires a uniform waypoint cadence starting "
            f"at one period; got action_target_time_offsets_s={action_target_time_offsets_s}."
        )

    def time_offsets_to_waypoint_steps(offsets: tuple[float, ...], *, name: str) -> tuple[int, ...]:
        steps = tuple(int(round(offset / action_period_s)) for offset in offsets)
        if any(not math.isclose(offset, step * action_period_s, abs_tol=1e-6) for offset, step in zip(offsets, steps)):
            raise RuntimeError(
                f"{name}={offsets} is not aligned to the declared one-target-per-waypoint "
                f"period {action_period_s}s. RoboTwin timing interpolation is not implemented."
            )
        return steps

    history_step_offsets = time_offsets_to_waypoint_steps(
        policy_contract.history_time_offsets_s, name="history_time_offsets_s"
    )
    future_step_offsets = time_offsets_to_waypoint_steps(
        policy_contract.world_future_time_offsets_s, name="world_future_time_offsets_s"
    )

    if execution_timing != _ROBOTWIN_EXECUTION_TIMING:
        raise RuntimeError(
            "RoboTwin's upstream take_action() plans a variable-length joint path, so "
            "wall-clock/simulator-step timing cannot be inferred safely. The server "
            "target timestamps cannot be treated as a fixed simulator frequency. Set "
            f"robotwin_execution_timing={_ROBOTWIN_EXECUTION_TIMING!r} in the client "
            "deployment config only when the checkpoint dataset represents one upstream "
            "EE waypoint per policy target. No implicit 10 Hz assumption is allowed."
        )

    # Reject malformed offsets up-front so downstream code can assume monotone
    # non-positive history offsets.
    if not history_step_offsets:
        raise RuntimeError("Server published an empty history_step_offsets.")
    if max(history_step_offsets) > 0:
        raise RuntimeError(
            f"history_step_offsets must include only non-positive values for history-only "
            f"inference, got {history_step_offsets}."
        )
    if list(history_step_offsets) != sorted(set(history_step_offsets)):
        raise RuntimeError(f"history_step_offsets must be strictly increasing; got {history_step_offsets}.")
    if history_step_offsets[-1] != 0:
        raise RuntimeError(
            f"history_step_offsets must end with the current-frame offset 0; got {history_step_offsets}."
        )

    # ``temporal_image_keys`` may be empty when the checkpoint disables the
    # world model (e.g. a vanilla pi0 stage). In that case we fall back to the
    # first image key as the "current frame" target.
    if temporal_image_keys:
        temporal_image_key = temporal_image_keys[0]
        if temporal_image_key not in image_keys:
            raise RuntimeError(
                f"Server published temporal_image_keys={temporal_image_keys} but the "
                f"temporal key is not part of image_keys={image_keys}."
            )
    else:
        # Pure single-frame mode (no history). We still need a primary camera key.
        if not image_keys:
            raise RuntimeError("Server input_spec has no image_keys; cannot serve.")
        temporal_image_key = image_keys[0]

    supported_cameras = {alias for alias, _ in _ROBOTWIN_CAMERA_SOURCES}
    if not image_keys or not set(image_keys).issubset(supported_cameras):
        raise RuntimeError(f"Unsupported camera keys {image_keys}; expected cameras from {sorted(supported_cameras)}.")

    return ServerContract(
        policy_config_name=str(metadata.get("policy_config_name") or "<unknown>"),
        action_type=action_type,
        native_action_dim=native_action_dim,
        action_horizon=action_horizon,
        family=family,
        image_keys=image_keys,
        temporal_image_key=temporal_image_key,
        state_key=state_key,
        prompt_key=prompt_key,
        history_step_offsets=history_step_offsets,
        future_step_offsets=future_step_offsets,
        action_target_time_offsets_s=action_target_time_offsets_s,
        execution_timing=execution_timing,
    )


def _encode_robotwin_ee_state(endpose, *, expected_dim: int = _EE_BIMANUAL_DIM) -> np.ndarray:
    """Serialize a RoboTwin observation into the canonical 16-D EE-pose layout.

    Accepts the dict produced by ``TASK_ENV.get_obs()['endpose']``::

        {'left_endpose':  [x, y, z, qw, qx, qy, qz],   # SAPIEN's Pose.q is w-first
         'left_gripper':  float,
         'right_endpose': [x, y, z, qw, qx, qy, qz],
         'right_gripper': float}

    The returned vector is laid out as
    ``[lx,ly,lz, lqw,lqx,lqy,lqz, lgrip, rx,ry,rz, rqw,rqx,rqy,rqz, rgrip]``
    which is exactly what the server's
    ``transforms._canonical_ee_arm_slices`` indexes into when applying
    ``DeltaActions(ee_pose=True)`` and ``AbsoluteActions(ee_pose=True)``.

    ``expected_dim`` should match ``ServerContract.native_action_dim`` so a
    misconfigured ckpt is caught at the first observation rather than silently
    propagating mis-aligned tensors through the policy server.
    """
    required_keys = ("left_endpose", "left_gripper", "right_endpose", "right_gripper")
    missing = tuple(key for key in required_keys if key not in endpose)
    if missing:
        raise ValueError(f"RoboTwin endpose observation is missing keys: {missing}.")
    state = np.asarray(
        [*endpose["left_endpose"], endpose["left_gripper"], *endpose["right_endpose"], endpose["right_gripper"]],
        dtype=np.float64,
    )
    if state.shape != (expected_dim,):
        raise ValueError(f"Expected {expected_dim}-D EE state (xyz+quat(w-first)+grip x 2), got shape={state.shape}.")
    if not np.isfinite(state).all():
        raise ValueError("RoboTwin EEF observations must contain only finite values.")
    state = state.copy()
    for quat_slice in _EE_QUAT_SLICES:
        state[quat_slice] = normalize_quaternion(state[quat_slice])
    grippers = state[[7, 15]]
    if np.any(grippers < 0.0) or np.any(grippers > 1.0):
        raise ValueError(
            f"RoboTwin observation grippers must be open fractions in [0, 1], got {grippers.tolist()}."
        )
    return state.astype(np.float32)


def _history_buffer_capacity(history_step_offsets) -> int:
    """Number of consecutive frames needed to satisfy ``history_step_offsets``.

    The deque is indexed so the newest entry (offset ``0``) is at the end and an entry
    at deque index ``i`` corresponds to history offset ``i - (capacity - 1)``. The
    capacity must therefore cover ``-min(offsets)`` past frames plus the current one.

    For offsets ``(-5, -4, -3, -2, -1, 0)``, capacity is six frames. A coarser
    schedule such as ``(-10, -5, 0)`` needs eleven frames even though only three
    are sent to the server. ``_stack_temporal_history`` selects the frames at
    the requested offsets from that window.
    """
    offsets = tuple(int(o) for o in history_step_offsets)
    if not offsets:
        raise ValueError("history_step_offsets must not be empty.")
    if max(offsets) > 0:
        raise ValueError(f"history_step_offsets must all be <= 0 (history-only), got {offsets}.")
    return -min(offsets) + 1


def _stack_temporal_history(
    buffer: "deque[np.ndarray]",
    history_step_offsets,
) -> np.ndarray:
    """Build the ``[len(history_step_offsets), H, W, 3]`` ``cam_high`` stack the server expects.

    The deque is treated as a rolling window of the most recent
    ``_history_buffer_capacity(history_step_offsets)`` head-camera observations
    (newest at the end). We then pick frames at the *numerical* step offsets the
    server published, so a stride-1 schedule like ``(-5,-4,-3,-2,-1,0)`` returns
    the last 6 frames and a stride-5 schedule like ``(-10,-5,0)`` returns every
    5th frame -- exactly the cadence the model was trained with.

    See ``SplitTemporalFrames`` (src/plawvla/transforms.py): when the client sends a
    history-only stack, the last frame is treated as the current observation and
    all preceding frames as history. We pre-pad with the oldest available frame so
    indices below ``-len(buffer)`` are valid -- this matches LeRobot's episode-
    boundary clamping behaviour at training.
    """
    offsets = tuple(int(o) for o in history_step_offsets)
    capacity = _history_buffer_capacity(offsets)

    frames = list(buffer)
    if not frames:
        raise ValueError("temporal history buffer is empty -- seed it before inference.")
    while len(frames) < capacity:
        frames.insert(0, frames[0])
    # frames is now exactly `capacity` long; index 0 = oldest, index (capacity-1) = current.
    # Frame at history offset `o (<= 0)` lives at index `(capacity - 1) + o`.
    indexed = [frames[capacity - 1 + offset] for offset in offsets]
    return np.stack(indexed, axis=0)


def _build_history_pad_mask(history_step_offsets, num_real_frames: int) -> np.ndarray:
    """Build the ``cam_high_is_pad`` mask matching training-time semantics.

    At training, LeRobot tags out-of-bound history slots with ``is_pad=True`` when
    sampling near the episode start; the server's
    ``transforms.split_temporal_valid_mask`` turns those into the attention mask
    ``image_mask["base_0_rgb_history"]`` consumed by the model.

    A slot ``i`` is padded iff the absolute offset ``|history_step_offsets[i]|`` is
    >= the number of distinct frames captured so far this episode. With
    ``num_real_frames=1`` (right after ``setup_demo``) only offset ``0`` is real;
    every prior slot is a duplicate of the initial frame and gets ``True``.

    Args:
        history_step_offsets: Step offsets the server published (all <= 0).
        num_real_frames: Distinct frames captured so far this episode
            (1 right after reset, +1 per ``take_action`` call).

    Returns:
        Bool array of shape ``(len(history_step_offsets),)``:
        ``True`` = padded duplicate, ``False`` = real observation.
    """
    offsets = tuple(int(o) for o in history_step_offsets)
    if not offsets:
        return np.zeros((0,), dtype=bool)
    real = max(0, int(num_real_frames))
    return np.array([(-int(offset)) >= real for offset in offsets], dtype=bool)


def _env_send_cam_high_is_pad() -> bool:
    """Whether to send the history-padding mask; enabled by default.

    Matches training-time semantics (LeRobot tags out-of-bound history slots
    with ``is_pad=True`` near the episode start). Set
    ``ROBOTWIN_SEND_CAM_HIGH_IS_PAD=0`` to disable.
    """
    return os.environ.get("ROBOTWIN_SEND_CAM_HIGH_IS_PAD", "1").lower() in {"1", "true", "yes"}


# Map RoboTwin simulator cameras to the keys used by the shared ALOHA transform.
_ROBOTWIN_CAMERA_SOURCES: tuple[tuple[str, str], ...] = (
    ("cam_high", "head_camera"),
    ("cam_left_wrist", "left_camera"),
    ("cam_right_wrist", "right_camera"),
)


def make_policy_obs(
    observation: dict,
    prompt: str,
    *,
    contract: ServerContract,
    task_id: str | None,
    task_config: str | None,
    cam_history: "deque[np.ndarray] | None" = None,
    num_real_frames: int | None = None,
    asset_id_override: str | None = None,
) -> dict:
    """Build a payload matching the RoboTwin server's ``input_spec``.

    All payload keys are taken from ``contract`` (parsed from server
    metadata): the temporal camera key, the wrist camera keys, the state and
    prompt keys, the history offsets and the EE state width. The only fixed
    contract on the RoboTwin side is the simulator-to-camera mapping in
    ``_ROBOTWIN_CAMERA_SOURCES`` (RoboTwin's ``observation['observation']``
    always exposes ``head_camera`` / ``left_camera`` / ``right_camera``).

    ``asset_id_override`` pins a specific per-task ``norm_stats`` entry on the
    server side. Otherwise, ``RobotwinPolicy`` resolves the task name and
    task config; ambiguous matches require an explicit asset id.

    The temporal stack is sampled from ``cam_history`` at the *numerical* step
    offsets in ``contract.history_step_offsets``. The caller is responsible
    for sizing the deque to ``contract.history_buffer_capacity`` and pushing
    exactly one frame per executed action -- this is how a server-side
    stride > 1 schedule (e.g. ``(-10,-5,0)``) is honored without sampling the
    same dense window the action loop produces.
    """
    obs_root = observation["observation"]

    # Validate up-front that the server actually asked for the cameras we know
    # how to populate, and that the temporal camera is one of them. If the
    # contract publishes a key we don't recognise, fail loudly -- silently
    # dropping a camera would result in zero-filled inputs at the server.
    sim_lookup = {alias: obs_root[sim_key]["rgb"] for alias, sim_key in _ROBOTWIN_CAMERA_SOURCES}
    for image_key in contract.image_keys:
        if image_key not in sim_lookup:
            raise RuntimeError(
                f"Server input_spec requires image key {image_key!r}, but this RobotWin "
                f"client only knows how to source the following Aloha-style cameras: "
                f"{sorted(sim_lookup.keys())}. Extend _ROBOTWIN_CAMERA_SOURCES if your "
                "simulator exposes the requested camera under a different name."
            )

    history_step_offsets = contract.history_step_offsets

    payload: dict[str, object] = {}

    for image_key in contract.image_keys:
        rgb = np.ascontiguousarray(sim_lookup[image_key], dtype=np.uint8)
        if image_key == contract.temporal_image_key and cam_history is not None and len(history_step_offsets) > 1:
            payload[image_key] = _stack_temporal_history(cam_history, history_step_offsets)
            if num_real_frames is not None and _env_send_cam_high_is_pad():
                # Picked up by ``transforms.split_temporal_valid_mask`` on the
                # server; length matches ``len(history_step_offsets)`` since
                # our payload is history-only. The pad-key naming follows
                # LeRobot's ``f"{key}_is_pad"`` convention.
                payload[f"{image_key}_is_pad"] = _build_history_pad_mask(history_step_offsets, num_real_frames)
        else:
            payload[image_key] = rgb

    payload[contract.state_key] = _encode_robotwin_ee_state(
        obs_root["endpose"] if "endpose" in obs_root else observation["endpose"],
        expected_dim=contract.native_action_dim,
    )
    payload[contract.prompt_key] = prompt

    # Precise (override) > task name. The server's RobotwinPolicy
    # looks up the hint against (a) full asset id, (b) full leaf, (c) bare
    # task name -- in that priority order.
    hint = asset_id_override or task_id
    if hint:
        payload["__robotwin_task_id__"] = hint
    if task_config:
        payload["__robotwin_task_config__"] = task_config
    return payload


@dataclasses.dataclass(frozen=True)
class RobotwinAbsoluteEefAdapter(ExecutionAdapter):
    """Adapt canonical absolute EEF targets to RoboTwin's native EE waypoint.

    This adapter is intentionally an identity mapping for position, orientation
    frame, and gripper semantics. It only validates the audited boundary and
    normalizes quaternions. In particular, it contains no LIBERO OSC scale.
    """

    arm_count: int = 2

    def __post_init__(self) -> None:
        if self.arm_count != 2:
            raise ValueError("The audited RoboTwin client supports exactly two arms.")

    def adapt(self, current_pose: np.ndarray, target_pose: np.ndarray) -> np.ndarray:
        current = np.asarray(current_pose, dtype=np.float64).reshape(-1)
        target = np.asarray(target_pose, dtype=np.float64).reshape(-1)
        expected_dim = self.arm_count * 8
        if current.shape != (expected_dim,) or target.shape != (expected_dim,):
            raise ValueError(
                f"RoboTwin canonical poses must both have shape ({expected_dim},); "
                f"got current={current.shape}, target={target.shape}."
            )
        if not np.isfinite(current).all() or not np.isfinite(target).all():
            raise ValueError("RoboTwin canonical poses must contain only finite values.")
        out = target.copy()
        for quat_slice in _EE_QUAT_SLICES:
            out[quat_slice] = normalize_quaternion(out[quat_slice])
        grippers = out[[7, 15]]
        if np.any(grippers < 0.0) or np.any(grippers > 1.0):
            raise ValueError(
                f"RoboTwin native grippers require open fractions in [0, 1], got {grippers.tolist()}."
            )
        return out


def write_json(data: dict, fpath: Path) -> None:
    """Write data to a JSON file.

    Creates parent directories if they don't exist.

    Args:
        data (dict): The dictionary to write.
        fpath (Path): The path to the output JSON file.
    """
    fpath.parent.mkdir(exist_ok=True, parents=True)
    temporary = fpath.with_name(fpath.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")
    temporary.replace(fpath)


def add_title_bar(img, text, font_scale=0.8, thickness=2):
    """Add a black title bar with text above the image"""
    h, w, _ = img.shape
    bar_height = 40

    # Create black background bar
    title_bar = np.zeros((bar_height, w, 3), dtype=np.uint8)

    # Calculate text position to center it
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_x = (w - text_w) // 2
    text_y = (bar_height + text_h) // 2 - 5

    cv2.putText(
        title_bar, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
    )

    return np.vstack([title_bar, img])


def save_rollout_video(observations, save_path, fps=15):
    """Save the actual head and wrist camera observations from a rollout."""
    if not observations:
        return
    print(f"Saving rollout video with {len(observations)} frames...")

    final_frames = []

    for obs in observations:
        cam_high = obs["observation.images.cam_high"]
        cam_left = obs["observation.images.cam_left_wrist"]
        cam_right = obs["observation.images.cam_right_wrist"]

        base_h = cam_high.shape[0]

        def resize_h(img, h):
            if img.shape[0] != h:
                w = int(img.shape[1] * h / img.shape[0])
                img = cv2.resize(img, (w, h))
            img = np.ascontiguousarray(img)
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            return img

        row_real = np.hstack([resize_h(cam_high, base_h), resize_h(cam_left, base_h), resize_h(cam_right, base_h)])

        row_real = np.ascontiguousarray(row_real)

        final_frames.append(add_title_bar(row_real, "Observation (Head / Left wrist / Right wrist)"))

    imageio.mimsave(save_path, final_frames, fps=fps)
    print(f"Rollout video saved to: {save_path}")


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except Exception as exc:
        raise SystemExit(f"Unable to initialize RoboTwin task {task_name}: {exc}") from exc
    return env_instance


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def main(usr_args):
    from envs import CONFIGS_PATH

    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    save_root = usr_args["save_root"]
    policy_name = usr_args["policy_name"]
    # Read the instruction split from the evaluation config.
    instruction_type = usr_args.get("instruction_type") or "unseen"
    save_dir = None

    with open(f"./task_config/{task_config}.yml", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting
    args["save_root"] = save_root

    # Rollout-video modes: ``none`` skips encoding, ``failed`` records failed
    # episodes, and ``all`` records every episode. Results are recorded in metrics
    # independently of videos. The simulator's separate recorder is
    # disabled to avoid writing duplicate videos inside its checkout.
    _video_mode = os.environ.get("ROBOTWIN_VIDEO_MODE", "none").strip().lower()
    if _video_mode not in ("none", "failed", "all"):
        raise SystemExit(f"ROBOTWIN_VIDEO_MODE must be one of: none, failed, all (got: {_video_mode!r})")
    args["video_mode"] = _video_mode
    args["eval_video_log"] = False
    print(f"[main] Rollout video mode: {_video_mode}", flush=True)

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise ValueError("No embodiment files")
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("embodiment items should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    save_dir = Path(save_root) / task_name / task_config / current_time
    save_dir.mkdir(parents=True, exist_ok=True)

    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print(
        "\033[94mHead Camera Config:\033[0m "
        + str(args["camera"]["head_camera_type"])
        + ", "
        + str(args["camera"]["collect_head_camera"])
    )
    print(
        "\033[94mWrist Camera Config:\033[0m "
        + str(args["camera"]["wrist_camera_type"])
        + ", "
        + str(args["camera"]["collect_wrist_camera"])
    )
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name

    seed = int(usr_args["seed"])

    # Upstream eval_policy.py: seed 0 starts at episode 100000.
    st_seed = 100000 * (1 + seed)
    np.random.seed(st_seed)
    args["expert_check"] = bool(usr_args.get("expert_check", True))
    suc_nums = []
    test_num = usr_args["test_num"]

    server_host = os.environ.get("POLICY_SERVER_HOST", "localhost")
    server_port = int(usr_args["port"])
    model = WebsocketClientPolicy(
        host=server_host,
        port=server_port,
        connect_timeout=usr_args["connect_timeout"],
        inference_timeout=usr_args["inference_timeout"],
    )

    try:
        # Parse + validate the full server contract once, then thread it through
        # every downstream call site. This is where we enforce alignment with the
        # training pipeline: the parser refuses to run against a checkpoint whose
        # action space / camera layout / temporal schedule we don't know how to
        # serve. See :class:`ServerContract` for the full list of fields.
        metadata = model.get_server_metadata() or {}
        contract = parse_server_contract(
            metadata,
            execution_timing=str(usr_args.get("robotwin_execution_timing") or ""),
        )

        # Cap the per-chunk execution budget at the server's action horizon.
        replan_steps_raw = usr_args.get("replan_steps")
        replan_steps_value = int(replan_steps_raw) if replan_steps_raw is not None else min(5, contract.action_horizon)
        if replan_steps_value < 1:
            raise ValueError("replan_steps must be a positive integer.")
        replan_steps_value = max(1, min(replan_steps_value, contract.action_horizon))

        # An explicit asset id selects one normalization entry when task/config
        # routing would be ambiguous. Set it in the config or ROBOTWIN_ASSET_ID.
        asset_id_override = usr_args.get("asset_id") or os.environ.get("ROBOTWIN_ASSET_ID") or None

        # Report the gaps between published history offsets, in executed actions.
        offsets = contract.history_step_offsets
        if len(offsets) >= 2:
            gaps = [offsets[i + 1] - offsets[i] for i in range(len(offsets) - 1)]
            history_stride_str = (
                f"{gaps[0]} step{'s' if abs(gaps[0]) != 1 else ''}"
                if all(g == gaps[0] for g in gaps)
                else f"non-uniform gaps={gaps}"
            )
        else:
            history_stride_str = "(current frame only)"

        print(
            "[client] server contract:\n"
            f"  policy_config_name        = {contract.policy_config_name}\n"
            f"  family                    = {contract.family}\n"
            f"  action_type / native_dim  = {contract.action_type} / {contract.native_action_dim}\n"
            f"  action_horizon            = {contract.action_horizon}\n"
            f"  image_keys                = {contract.image_keys}\n"
            f"  temporal_image_key        = {contract.temporal_image_key}\n"
            f"  state_key / prompt_key    = {contract.state_key!r} / {contract.prompt_key!r}\n"
            f"  history_step_offsets      = {offsets}  (stride={history_stride_str})\n"
            f"  history_buffer_capacity   = {contract.history_buffer_capacity}  "
            f"(= max(-offsets)+1; covers dense one-frame-per-action pushes)\n"
            f"  future_step_offsets       = {contract.future_step_offsets}  "
            "(advertised by server; we only send history at inference)\n"
            f"  action_target_times_s     = {contract.action_target_time_offsets_s}\n"
            f"  execution_timing          = {contract.execution_timing}\n"
            f"  replan_steps (per-chunk exec) = {replan_steps_value} (capped at action_horizon)\n"
            f"  asset_id_override         = {asset_id_override!r}"
        )

        st_seed, suc_num = eval_policy(
            task_name,
            TASK_ENV,
            args,
            model,
            st_seed,
            test_num=test_num,
            instruction_type=instruction_type,
            task_config=task_config,
            replan_steps=replan_steps_value,
            contract=contract,
            asset_id_override=asset_id_override,
        )
        suc_nums.append(suc_num)

        file_path = os.path.join(save_dir, "_result.txt")
        with open(file_path, "w") as file:
            file.write(f"Timestamp: {current_time}\n\n")
            file.write(f"Instruction Type: {instruction_type}\n\n")
            file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

        print(f"Data has been saved to {file_path}")
    finally:
        try:
            model.close()
        finally:
            TASK_ENV.close_env()


def format_obs(observation, prompt):
    """Build the single-frame observation used by ``save_rollout_video``.

    ``make_policy_obs`` builds the separate temporal payload for inference.
    """
    return {
        "observation.images.cam_high": observation["observation"]["head_camera"]["rgb"],
        "observation.images.cam_left_wrist": observation["observation"]["left_camera"]["rgb"],
        "observation.images.cam_right_wrist": observation["observation"]["right_camera"]["rgb"],
        "observation.state": _encode_robotwin_ee_state(observation["endpose"]),
        "task": prompt,
    }


def eval_policy(
    task_name,
    TASK_ENV,
    args,
    model,
    st_seed,
    test_num=100,
    instruction_type=None,
    task_config: str | None = None,
    replan_steps: int = 5,
    contract: ServerContract | None = None,
    asset_id_override: str | None = None,
):
    from description.utils.generate_episode_instructions import generate_episode_descriptions
    from envs.utils.create_actor import UnStableError

    if contract is None:
        raise RuntimeError("eval_policy requires a ServerContract parsed from server metadata.")
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")
    # Effective per-chunk execution budget (already capped by ``contract.action_horizon``
    # at the call site; clamp again here so direct callers of eval_policy stay safe).
    replan_steps = max(1, min(int(replan_steps), contract.action_horizon))
    # Retain consecutive observations through the oldest requested history
    # offset. This can require more frames than the number sent to the server.
    history_buffer_capacity = contract.history_buffer_capacity
    execution_adapter: ExecutionAdapter = RobotwinAbsoluteEefAdapter(
        arm_count=contract.native_action_dim // 8
    )

    expert_check = args.get("expert_check", True)
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0

    now_seed = st_seed
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True

    while succ_seed < test_num:
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError:
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                print(f"error occurs ! {e}")
                traceback.print_exc()
                continue
        else:
            # Upstream tasks populate language placeholders during the scripted
            # demonstration. Recreate the same scene for the policy rollout;
            # do not discard or substitute the requested evaluation seed.
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
            finally:
                TASK_ENV.close_env()

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
        instruction = np.random.choice(results[0][instruction_type])
        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

        succ = False

        prompt = TASK_ENV.get_instruction()

        # ``video_mode == "none"`` means we never call ``save_rollout_video``,
        # so don't waste memory accumulating per-frame obs (each entry holds 3
        # RGB frames -> tens-to-hundreds of MB per episode on long horizons).
        # For ``"failed"`` we still need the history because success is only
        # known at episode end.
        video_mode = args.get("video_mode", "none")
        need_obs_history = video_mode != "none"
        full_obs_list = []

        # Seed the temporal history deque with copies of the first frame (matches
        # how LeRobot clamps out-of-bound offsets at the episode boundary).
        # ``num_real_frames`` lets us emit a matching ``<temporal_key>_is_pad``
        # mask.
        #
        # The deque is sized to ``history_buffer_capacity`` (= -min(offsets)+1),
        # NOT to ``len(history_step_offsets)``. For stride=1 these are equal;
        # for stride>1 the deque must store the in-between frames too so that
        # ``_stack_temporal_history`` can index them at the correct step gaps.
        #
        # Select the temporal camera published in the server contract.
        initial_obs = TASK_ENV.get_obs()
        temporal_sim_key = dict(_ROBOTWIN_CAMERA_SOURCES)[contract.temporal_image_key]
        initial_temporal_frame = np.ascontiguousarray(
            initial_obs["observation"][temporal_sim_key]["rgb"], dtype=np.uint8
        )
        cam_history: deque[np.ndarray] = deque(maxlen=history_buffer_capacity)
        for _ in range(history_buffer_capacity):
            cam_history.append(initial_temporal_frame.copy())
        num_real_frames: int = 1

        if need_obs_history:
            full_obs_list.append(format_obs(initial_obs, prompt))

        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            observation = TASK_ENV.get_obs()
            payload = make_policy_obs(
                observation,
                prompt,
                contract=contract,
                task_id=task_name,
                task_config=task_config,
                cam_history=cam_history,
                num_real_frames=num_real_frames,
                asset_id_override=asset_id_override,
            )
            # Stateless server: one obs in, one (action_horizon, action_dim)
            # chunk out.
            ret = model.infer(payload)
            actions = np.asarray(ret["actions"])
            expected_action_shape = (contract.action_horizon, contract.native_action_dim)
            if actions.shape != expected_action_shape:
                raise ValueError(
                    f"Expected canonical absolute EEF targets with shape "
                    f"{expected_action_shape}, got {actions.shape}."
                )
            if not np.isfinite(actions).all():
                raise ValueError("Policy returned nonfinite RoboTwin actions.")
            # Respect the per-chunk replanning budget.
            remaining_steps = TASK_ENV.step_lim - TASK_ENV.take_action_cnt
            actions = actions[: min(replan_steps, remaining_steps)]

            for raw_step in actions:
                live_observation = TASK_ENV.get_obs()
                live_pose = _encode_robotwin_ee_state(
                    live_observation["endpose"],
                    expected_dim=contract.native_action_dim,
                )
                ee_action = execution_adapter.adapt(live_pose, raw_step)
                TASK_ENV.take_action(ee_action, action_type=contract.action_type)

                next_obs = TASK_ENV.get_obs()
                # Advance the temporal history one *action step* per executed
                # target. This is waypoint-index timing, not simulator wall
                # time: upstream take_action() plans and executes a variable
                # number of inner physics steps. parse_server_contract() only
                # permits this path when the server explicitly declares
                # one_target_per_take_action.
                cam_history.append(
                    np.ascontiguousarray(next_obs["observation"][temporal_sim_key]["rgb"], dtype=np.uint8)
                )
                num_real_frames += 1
                if need_obs_history:
                    full_obs_list.append(format_obs(next_obs, prompt))

                if TASK_ENV.eval_success:
                    succ = True
                    break

            if succ:
                break

        # Rollout-video policy (controlled by ROBOTWIN_VIDEO_MODE):
        #   * mode == "all"    -> always encode <n>_<prompt>_<True|False>.mp4
        #   * mode == "failed" -> encode only on failure
        #   * mode == "none"   -> never write any video file (no placeholder).
        # ``calc_stat.py`` reads ``res.json`` (written below) for success rates,
        # so we don't need a per-episode filename marker.
        should_save_video = video_mode == "all" or (video_mode == "failed" and not succ)
        if should_save_video:
            vis_dir = Path(args["save_root"]) / f"stseed-{st_seed}" / "visualization" / task_name
            vis_dir.mkdir(parents=True, exist_ok=True)
            safe_prompt = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(prompt)).strip("_")[:120] or "task"
            video_name = f"{TASK_ENV.test_num}_{safe_prompt}_{succ}.mp4"
            out_img_file = vis_dir / video_name
            save_rollout_video(
                observations=full_obs_list,
                save_path=str(out_img_file),
                fps=15,  # Suggest adjusting fps based on simulation step
            )
        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        save_dir = Path(args["save_root"]) / f"stseed-{st_seed}" / "metrics" / task_name
        save_dir.mkdir(parents=True, exist_ok=True)
        out_json_file = save_dir / "res.json"
        write_json(
            {
                "succ_num": float(TASK_ENV.suc),
                "total_num": float(TASK_ENV.test_num),
                "succ_rate": float(TASK_ENV.suc / TASK_ENV.test_num),
            },
            out_json_file,
        )

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc / TASK_ENV.test_num * 100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        now_seed += 1

    return now_seed, TASK_ENV.suc


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    parser.add_argument("--port", type=int, default=8001, help="remote policy socket port.")
    parser.add_argument("--save_root", type=str, default="results/default_vis_path")
    parser.add_argument("--test_num", type=int, default=100)
    parser.add_argument("--connect_timeout", type=float, default=30.0)
    parser.add_argument("--inference_timeout", type=float, default=60.0)
    args = parser.parse_args()

    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = _CALLER_DIR / config_path
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        parser.error("--config must contain a YAML mapping")

    # Parse overrides
    def parse_override_pairs(pairs):
        if len(pairs) % 2:
            parser.error("--overrides expects --key value pairs")
        override_dict = {}
        for i in range(0, len(pairs), 2):
            if not pairs[i].startswith("--") or len(pairs[i]) == 2:
                parser.error("--overrides expects --key value pairs")
            key = pairs[i].removeprefix("--")
            value = pairs[i + 1]
            try:
                import ast

                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    # Apply CLI settings to the evaluation config consumed by ``main``.
    for _key in ("save_root", "port", "test_num", "connect_timeout", "inference_timeout"):
        config[_key] = getattr(args, _key)
    save_root = Path(config["save_root"]).expanduser()
    if not save_root.is_absolute():
        save_root = _CALLER_DIR / save_root
    config["save_root"] = str(save_root.resolve())
    if args.test_num < 1:
        parser.error("--test_num must be a positive integer")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if not isinstance(config.get("seed"), int) or config["seed"] < 0:
        parser.error("seed must be a nonnegative integer")
    for name in ("connect_timeout", "inference_timeout"):
        if not math.isfinite(config[name]) or config[name] <= 0:
            parser.error(f"--{name} must be finite and positive")

    return config


if __name__ == "__main__":
    usr_args = parse_args_and_config()
    main(usr_args)
