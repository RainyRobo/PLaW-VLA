# Adapted from RoboTwin (Copyright (c) 2025 Tianxing Chen; MIT) and
# LingBot-VLA (Copyright 2026 Robbyant Team; Apache-2.0).
# Modified for PLaW-VLA by the PLaW-VLA authors, 2026.
# See LICENSE, NOTICE, and LICENSES/MIT-RoboTwin.txt.
import sys
import os
import dataclasses
import subprocess
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2
from pathlib import Path

# Project layout:
#   <repo>/examples/robotwin/          <- this file
#   <repo>/third_party/robotwin/       <- RoboTwin codebase (envs, task_config, description, ...)
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
robowin_root = _PROJECT_ROOT / "third_party" / "robotwin"
if not robowin_root.is_dir():
    raise FileNotFoundError(
        f"RobotWin checkout not found at {robowin_root}; make sure third_party/robotwin/ is initialised."
    )

# RoboTwin tasks resolve ``task_config/*.yml`` and simulator resources from
# the upstream checkout's root.
if str(robowin_root) not in sys.path:
    sys.path.insert(0, str(robowin_root))
os.chdir(robowin_root)

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

import numpy as np
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import argparse

import imageio
from scipy.spatial.transform import Rotation as R
import json

from openpi_client.websocket_client_policy import WebsocketClientPolicy


# ---------------------------------------------------------------------------
# openpi RoboTwin server contract
# ---------------------------------------------------------------------------
# Canonical bimanual EEF layout expected by this client. It matches
# ``_canonical_ee_arm_slices`` in src/openpi/transforms.py; ServerContract
# validates the server's published action layout.
#
#   left_xyz, left_quat(qw, qx, qy, qz), left_gripper,    -> 8 floats
#   right_xyz, right_quat(qw, qx, qy, qz), right_gripper. -> 8 floats
#                                                  total = 16 floats
#
# The quaternion order is ``[qw, qx, qy, qz]`` (w-first, SAPIEN
# convention). RoboTwin's ``robot.get_left_ee_pose()`` returns
# ``[x, y, z, qw, qx, qy, qz]`` (see ``code_gen/prompt.py`` in the upstream
# RoboTwin repo), which is what ``_encode_robotwin_ee_state`` concatenates
# without re-ordering. The training-time stack consumed by
# ``DeltaActions(ee_pose=True)`` / ``AbsoluteActions(ee_pose=True)`` expects
# the same w-first layout for quaternion composition.
_EE_QUAT_SLICES = (slice(3, 7), slice(11, 15))  # left arm quat, right arm quat
_EE_BIMANUAL_DIM = 16

# Connect directly to the local policy server rather than through an HTTP proxy.
for _proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
    os.environ.pop(_proxy_var, None)


@dataclasses.dataclass(frozen=True)
class ServerContract:
    """Validated action, camera, and temporal layout of the policy server.

    The client reads ``policy_metadata`` and ``input_spec`` before simulation
    begins. Camera keys and history offsets determine the observation payload;
    the action layout determines how returned targets are sent to the robot.
    """

    policy_config_name: str
    action_type: str  # "ee" (only one supported today on this client)
    native_action_dim: int  # 16 for bimanual EE pose
    action_horizon: int  # max chunk length the model emits per inference
    family: str  # "aloha" (RoboTwin uses the shared ALOHA policy transform)
    image_keys: tuple[str, ...]  # all camera keys we must include in the payload
    temporal_image_key: str  # the single camera the model consumes as history
    state_key: str  # payload key for the 16-D EE state
    prompt_key: str  # payload key for the language instruction
    history_step_offsets: tuple[int, ...]
    future_step_offsets: tuple[int, ...]

    @property
    def history_buffer_capacity(self) -> int:
        return _history_buffer_capacity(self.history_step_offsets)


def parse_server_contract(metadata: dict) -> ServerContract:
    """Validate server metadata and return the corresponding observation layout.

    Raises ``RuntimeError`` for unsupported actions, cameras, or temporal keys.
    """
    if not isinstance(metadata, dict):
        raise RuntimeError(f"Server metadata must be a dict, got {type(metadata).__name__}.")

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

    spec = metadata.get("input_spec")
    if not isinstance(spec, dict):
        raise RuntimeError(
            "Server metadata is missing the 'input_spec' block published by openpi.serving.policy_input_spec."
        )

    family = str(spec.get("family") or "")
    if family != "aloha":
        raise RuntimeError(
            f"Expected an 'aloha' family server (RobotWin reuses the Aloha policy "
            f"transform); server published family={family!r}."
        )

    image_keys = tuple(str(k) for k in spec.get("image_keys") or ())
    temporal_image_keys = tuple(str(k) for k in spec.get("temporal_image_keys") or ())
    if len(temporal_image_keys) > 1:
        raise RuntimeError(
            f"This client only knows how to maintain temporal history for a single "
            f"camera; server published {len(temporal_image_keys)} temporal_image_keys="
            f"{temporal_image_keys}. Extend make_policy_obs / eval_policy to keep one "
            "deque per temporal key."
        )

    state_key = spec.get("state_key")
    prompt_key = spec.get("prompt_key")
    if not isinstance(state_key, str):
        raise RuntimeError(
            f"Server input_spec is missing 'state_key' (got {state_key!r}); "
            "this client cannot send observations without it."
        )
    if not isinstance(prompt_key, str):
        raise RuntimeError(f"Server input_spec is missing 'prompt_key' (got {prompt_key!r}).")

    history_step_offsets = tuple(int(x) for x in spec.get("history_step_offsets") or ())
    future_step_offsets = tuple(int(x) for x in spec.get("future_step_offsets") or ())

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
    state = np.array(
        list(endpose["left_endpose"])
        + [endpose["left_gripper"]]
        + list(endpose["right_endpose"])
        + [endpose["right_gripper"]],
        dtype=np.float32,
    )
    if state.shape != (expected_dim,):
        raise ValueError(f"Expected {expected_dim}-D EE state (xyz+quat(w-first)+grip x 2), got shape={state.shape}.")
    return state


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

    See ``SplitTemporalFrames`` (src/openpi/transforms.py): when the client sends a
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


def normalize_robotwin_ee_action(
    action: np.ndarray,
    *,
    expected_dim: int = _EE_BIMANUAL_DIM,
) -> np.ndarray:
    """Project a 16-D EEF action into a form RoboTwin's IK solver accepts.

    Re-normalises each arm's quaternion so the IK solver does not get a
    near-but-not-quite unit quat from the policy (which would silently bias
    the end-effector orientation). Quaternion slices are
    ``[3:7]`` (left arm, ``[qw,qx,qy,qz]``) and ``[11:15]`` (right arm); see
    ``_EE_QUAT_SLICES`` and the module-level layout comment.

    ``expected_dim`` should match ``ServerContract.native_action_dim`` so a
    misconfigured ckpt is caught at the first inference rather than silently
    producing IK-invalid actions.
    """
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    if action.shape != (expected_dim,):
        raise ValueError(f"Expected {expected_dim}-D EE action (xyz+quat(w-first)+grip x 2), got shape={action.shape}.")
    out = action.copy()
    for quat_slice in _EE_QUAT_SLICES:
        q = out[quat_slice]
        norm = float(np.linalg.norm(q))
        if norm > 1e-8:
            out[quat_slice] = q / norm
    return out


def write_json(data: dict, fpath: Path) -> None:
    """Write data to a JSON file.

    Creates parent directories if they don't exist.

    Args:
        data (dict): The dictionary to write.
        fpath (Path): The path to the output JSON file.
    """
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with open(fpath, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


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


def quaternion_to_euler(quat):
    """
    Convert quaternion to Euler angles (roll, pitch, yaw) (radians).

    ``quat`` is the canonical openpi/SAPIEN bimanual EE layout ``[qw, qx, qy, qz]``
    (w-first). scipy's ``Rotation.from_quat`` expects ``[qx, qy, qz, qw]``
    (w-last), so we have to re-order before calling it -- DO NOT pass the slice
    directly or every plotted euler angle is silently wrong.
    """
    qw, qx, qy, qz = quat
    rotation = R.from_quat([qx, qy, qz, qw])
    euler = rotation.as_euler("xyz", degrees=False)
    return euler


def visualize_action_step(action_history, step_idx, window=50):
    """
    Plot dual-arm action curves:
    Subplot 1: Left arm XYZ Position + Gripper
    Subplot 2: Left arm Euler angles (Roll, Pitch, Yaw) - converted from quaternion
    Subplot 3: Right arm XYZ Position + Gripper
    Subplot 4: Right arm Euler angles (Roll, Pitch, Yaw) - converted from quaternion

    Input data format (canonical openpi bimanual EE-pose layout, w-first quats):
        [left_x, left_y, left_z, left_qw, left_qx, left_qy, left_qz, left_gripper,
         right_x, right_y, right_z, right_qw, right_qx, right_qy, right_qz, right_gripper]
    Total 16 dimensions
    """
    # Create four subplots, sharing the X-axis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 8), dpi=100, sharex=True)

    # 1. Determine slice range
    start = max(0, step_idx - window)
    end = step_idx + 1

    # 2. Get data subset
    history_subset = np.array(action_history)[start:end]

    # 3. Generate X-axis based on actual data length
    actual_len = len(history_subset)
    x_axis = range(start, start + actual_len)

    if actual_len > 0 and history_subset.shape[1] >= 16:
        # Convert quaternions to Euler angles
        left_euler = []
        right_euler = []

        for action in history_subset:
            left_quat = action[3:7]  # [qw, qx, qy, qz] - SAPIEN/openpi w-first
            left_rpy = quaternion_to_euler(left_quat)
            left_euler.append(left_rpy)

            right_quat = action[11:15]  # [qw, qx, qy, qz]
            right_rpy = quaternion_to_euler(right_quat)
            right_euler.append(right_rpy)

        left_euler = np.array(left_euler)
        right_euler = np.array(right_euler)

        # --- Left Arm ---
        # Subplot 1: Left Arm Translation (XYZ) + Gripper
        ax1.plot(x_axis, history_subset[:, 0], label="left_x", color="r", linewidth=1.5)
        ax1.plot(x_axis, history_subset[:, 1], label="left_y", color="g", linewidth=1.5)
        ax1.plot(x_axis, history_subset[:, 2], label="left_z", color="b", linewidth=1.5)
        ax1.plot(x_axis, history_subset[:, 7], label="left_grip", color="orange", linestyle=":", linewidth=2, alpha=0.8)
        ax1.set_ylabel("Position (m)")
        ax1.legend(loc="upper right", fontsize="x-small", ncol=4)
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f"Step {step_idx}: Left Arm Position & Gripper")

        # Subplot 2: Left Arm Euler Angles (Roll, Pitch, Yaw)
        ax2.plot(x_axis, left_euler[:, 0], label="left_roll", color="c", linewidth=1.5)
        ax2.plot(x_axis, left_euler[:, 1], label="left_pitch", color="m", linewidth=1.5)
        ax2.plot(x_axis, left_euler[:, 2], label="left_yaw", color="y", linewidth=1.5)
        ax2.set_ylabel("Rotation (rad)")
        ax2.legend(loc="upper right", fontsize="x-small", ncol=3)
        ax2.grid(True, alpha=0.3)
        ax2.set_title("Left Arm Rotation (RPY from Quaternion)")

        # --- Right Arm ---
        # Subplot 3: Right Arm Translation (XYZ) + Gripper
        ax3.plot(x_axis, history_subset[:, 8], label="right_x", color="r", linewidth=1.5, linestyle="--")
        ax3.plot(x_axis, history_subset[:, 9], label="right_y", color="g", linewidth=1.5, linestyle="--")
        ax3.plot(x_axis, history_subset[:, 10], label="right_z", color="b", linewidth=1.5, linestyle="--")
        ax3.plot(
            x_axis, history_subset[:, 15], label="right_grip", color="orange", linestyle=":", linewidth=2, alpha=0.8
        )
        ax3.set_ylabel("Position (m)")
        ax3.legend(loc="upper right", fontsize="x-small", ncol=4)
        ax3.grid(True, alpha=0.3)
        ax3.set_title("Right Arm Position & Gripper")

        # Subplot 4: Right Arm Euler Angles (Roll, Pitch, Yaw)
        ax4.plot(x_axis, right_euler[:, 0], label="right_roll", color="c", linewidth=1.5, linestyle="--")
        ax4.plot(x_axis, right_euler[:, 1], label="right_pitch", color="m", linewidth=1.5, linestyle="--")
        ax4.plot(x_axis, right_euler[:, 2], label="right_yaw", color="y", linewidth=1.5, linestyle="--")
        ax4.set_ylabel("Rotation (rad)")
        ax4.legend(loc="upper right", fontsize="x-small", ncol=3)
        ax4.grid(True, alpha=0.3)
        ax4.set_title("Right Arm Rotation (RPY from Quaternion)")

    # Set X-axis display range to maintain sliding window effect
    ax1.set_xlim(max(0, step_idx - window), max(window, step_idx))
    ax3.set_xlabel("Step")
    ax4.set_xlabel("Step")

    plt.tight_layout()
    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.asarray(canvas.buffer_rgba())
    img = img[:, :, :3]

    # Convert to uint8
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)

    plt.close(fig)
    return img


def save_comparison_video(real_obs_list, imagined_video, action_history, save_path, fps=15):
    if not real_obs_list:
        return

    n_real = len(real_obs_list)
    if imagined_video is not None:
        imagined_video = np.concatenate(imagined_video, 0)
        n_imagined = len(imagined_video)
    else:
        n_imagined = 0
    n_frames = n_real  # Based on real observation frames

    print(f"Saving video: Real {n_real} frames, Imagined {n_imagined} frames...")

    final_frames = []

    for i in range(n_frames):
        obs = real_obs_list[i]
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

        row_real = add_title_bar(row_real, "Real Observation (High / Left / Right)")

        target_width = row_real.shape[1]

        if imagined_video is not None and i < n_imagined:
            img_frame = imagined_video[i]
            if img_frame.dtype != np.uint8 and img_frame.max() <= 1.0001:
                img_frame = (img_frame * 255).astype(np.uint8)
            elif img_frame.dtype != np.uint8:
                img_frame = img_frame.astype(np.uint8)

            h = int(img_frame.shape[0] * target_width / img_frame.shape[1])
            row_imagined = cv2.resize(img_frame, (target_width, h))
        else:
            row_imagined = np.zeros((300, target_width, 3), dtype=np.uint8)
            cv2.putText(
                row_imagined,
                "Coming soon",
                (target_width // 2 - 100, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (100, 100, 100),
                2,
            )

        row_imagined = np.ascontiguousarray(row_imagined)
        row_imagined = add_title_bar(row_imagined, "Imagined Video Stream")
        full_frame = np.vstack([row_real, row_imagined])
        full_frame = np.ascontiguousarray(full_frame)
        final_frames.append(full_frame)

    imageio.mimsave(save_path, final_frames, fps=fps)
    print(f"Combined video saved to: {save_path}")


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e


def get_camera_config(camera_type):
    camera_config_path = os.path.join(robowin_root, "task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(usr_args):
    from envs import CONFIGS_PATH

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    save_root = usr_args["save_root"]
    policy_name = usr_args["policy_name"]
    # Read the instruction split from the evaluation config.
    instruction_type = usr_args.get("instruction_type") or "unseen"
    save_dir = None
    video_save_dir = None
    video_size = None

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting
    args["save_root"] = save_root

    # Comparison-video modes: ``none`` skips encoding, ``failed`` records failed
    # episodes, and ``all`` records every episode. Results are recorded in metrics
    # independently of videos. The simulator's separate recorder is
    # disabled to avoid writing duplicate videos inside its checkout.
    _video_mode = os.environ.get("ROBOTWIN_VIDEO_MODE", "none").strip().lower()
    if _video_mode not in ("none", "failed", "all"):
        raise SystemExit(f"ROBOTWIN_VIDEO_MODE must be one of: none, failed, all (got: {_video_mode!r})")
    args["video_mode"] = _video_mode
    args["eval_video_log"] = False
    print(
        f"[main] ROBOTWIN_VIDEO_MODE={_video_mode}  "
        f"(in-sim ffmpeg: OFF, comparison-video: "
        f"{'ALWAYS' if _video_mode == 'all' else ('FAILURES ONLY' if _video_mode == 'failed' else 'NEVER')})",
        flush=True,
    )

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise ValueError("No embodiment files")
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
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

    if args["eval_video_log"]:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

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
        + f", "
        + str(args["camera"]["collect_head_camera"])
    )
    print(
        "\033[94mWrist Camera Config:\033[0m "
        + str(args["camera"]["wrist_camera_type"])
        + f", "
        + str(args["camera"]["collect_wrist_camera"])
    )
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]

    st_seed = int(seed)
    np.random.seed(st_seed)
    args["expert_check"] = bool(usr_args.get("expert_check", False))
    suc_nums = []
    test_num = usr_args["test_num"]

    server_host = os.environ.get("POLICY_SERVER_HOST", "localhost")
    server_port = int(usr_args.get("port") or os.environ.get("POLICY_SERVER_PORT", "8000"))
    model = WebsocketClientPolicy(host=server_host, port=server_port)

    # Parse + validate the full server contract once, then thread it through
    # every downstream call site. This is where we enforce alignment with the
    # training pipeline: the parser refuses to run against a checkpoint whose
    # action space / camera layout / temporal schedule we don't know how to
    # serve. See :class:`ServerContract` for the full list of fields.
    metadata = model.get_server_metadata() or {}
    contract = parse_server_contract(metadata)

    # Cap the per-chunk execution budget at the server's action horizon.
    replan_steps_raw = usr_args.get("replan_steps", usr_args.get("pi05_step"))
    replan_steps_value = int(replan_steps_raw) if replan_steps_raw is not None else min(5, contract.action_horizon)
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
        video_size=video_size,
        instruction_type=instruction_type,
        task_config=task_config,
        replan_steps=replan_steps_value,
        contract=contract,
        asset_id_override=asset_id_override,
    )
    suc_nums.append(suc_num)

    file_path = os.path.join(save_dir, f"_result.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    print(f"Data has been saved to {file_path}")


def format_obs(observation, prompt):
    """Build the single-frame observation used by ``save_comparison_video``.

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
    video_size=None,
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
    history_step_offsets = contract.history_step_offsets
    # Retain consecutive observations through the oldest requested history
    # offset. This can require more frames than the number sent to the server.
    history_buffer_capacity = contract.history_buffer_capacity

    expert_check = args.get("expert_check", False)
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []

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
            except UnStableError as e:
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
            suc_test_seed_list.append(now_seed)
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

        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False

        prompt = TASK_ENV.get_instruction()

        # ``video_mode == "none"`` means we never call ``save_comparison_video``,
        # so don't waste memory accumulating per-frame obs (each entry holds 3
        # RGB frames -> tens-to-hundreds of MB per episode on long horizons).
        # For ``"failed"`` we still need the history because success is only
        # known at episode end.
        video_mode = args.get("video_mode", "none")
        need_obs_history = video_mode != "none"
        full_obs_list = []
        full_action_history = []

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
            # The model may have padded its action vector beyond
            # ``native_action_dim`` (e.g. PI0 pads to 32 internally); only the
            # first ``contract.native_action_dim`` cols are real EE values.
            actions = actions[:, : contract.native_action_dim]
            # Respect the per-chunk replanning budget.
            actions = actions[:replan_steps]

            for raw_step in actions:
                ee_action = normalize_robotwin_ee_action(raw_step, expected_dim=contract.native_action_dim)
                full_action_history.append(ee_action.copy())
                TASK_ENV.take_action(ee_action, action_type=contract.action_type)

                next_obs = TASK_ENV.get_obs()
                # Advance the temporal history one *action step* per executed
                # action. The model's ``action_time_step_s`` defines one "step";
                # each ee-action we execute corresponds to exactly one such
                # step, so we always push 1 frame per ``take_action``. The deque
                # keeps every in-between frame even when the server published a
                # stride>1 ``history_step_offsets`` schedule, and
                # ``_stack_temporal_history`` then samples at the published
                # indices to recreate the training cadence (e.g. picking every
                # 5th frame for a stride-5 schedule).
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

        # Comparison-video policy (controlled by ROBOTWIN_VIDEO_MODE):
        #   * mode == "all"    -> always encode <n>_<prompt>_<True|False>.mp4
        #   * mode == "failed" -> encode only on failure
        #   * mode == "none"   -> never write any video file (no placeholder).
        # ``calc_stat.py`` reads ``res.json`` (written below) for success rates,
        # so we don't need a per-episode filename marker.
        should_save_video = video_mode == "all" or (video_mode == "failed" and not succ)
        if should_save_video:
            vis_dir = Path(args["save_root"]) / f"stseed-{st_seed}" / "visualization" / task_name
            vis_dir.mkdir(parents=True, exist_ok=True)
            video_name = f"{TASK_ENV.test_num}_{prompt.replace(' ', '_')}_{succ}.mp4"
            out_img_file = vis_dir / video_name
            save_comparison_video(
                real_obs_list=full_obs_list,
                imagined_video=None,  # gen_video_list,
                action_history=full_action_history,
                save_path=str(out_img_file),
                fps=15,  # Suggest adjusting fps based on simulation step
            )
        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

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
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        if len(pairs) % 2:
            parser.error("--overrides expects --key value pairs")
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
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
    for _key in ("save_root", "port", "test_num"):
        config[_key] = getattr(args, _key)

    return config


if __name__ == "__main__":
    usr_args = parse_args_and_config()
    main(usr_args)
