import dataclasses
from collections.abc import Sequence
from typing import Literal

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.policies import ee_pose_utils


RAW_STATE_DIM = 8
RAW_ACTION_DIM = 7
CANONICAL_STATE_DIM = 8
CANONICAL_ACTION_DIM = 8
LIBERO_GRIPPER_OPENNESS_SCALE = 0.04
LiberoStateGripperFormat = Literal["physical_width", "open_fraction"]
LiberoActionGripperFormat = Literal["signed_command", "binary_target", "absolute_physical_width"]


def make_libero_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)

    if image.ndim == 4 and image.shape[1] == 3:  
        image = einops.rearrange(image, "t c h w -> t h w c")
    elif image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")

    return image


def _is_canonical_libero_state(state: np.ndarray) -> bool:
    """Check if state is already in canonical format (quaternion orientation)."""
    if state.shape[-1] != CANONICAL_STATE_DIM:
        return False
    # Canonical state has unit-quaternion orientation; raw state has rotation-vector.
    quat = state[..., 3:7]
    quat_norm = np.sum(np.square(quat), axis=-1)
    return bool(np.all(np.abs(quat_norm - 1.0) < 0.5))


def _canonicalize_libero_state(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    if state.shape[-1] != RAW_STATE_DIM:
        raise ValueError(f"Expected LIBERO state dim {RAW_STATE_DIM}, got {state.shape}.")
    position = state[..., :3]
    orientation = ee_pose_utils.rotation_vector_to_quaternion(state[..., 3:6])
    gripper = ee_pose_utils.collapse_opposing_gripper_fingers(state[..., 6:8])[..., None]
    gripper = np.clip(gripper / LIBERO_GRIPPER_OPENNESS_SCALE, 0.0, 1.0)
    return np.concatenate([position, orientation, gripper], axis=-1)


def _coerce_canonical_libero_state(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    if state.shape[-1] < CANONICAL_STATE_DIM:
        raise ValueError(f"Expected canonical LIBERO state dim >= {CANONICAL_STATE_DIM}, got {state.shape}.")
    if state.shape[-1] > CANONICAL_STATE_DIM:
        state = state[..., :CANONICAL_STATE_DIM]
    return state


def _canonicalize_stored_libero_state(
    state: np.ndarray,
    *,
    gripper_format: LiberoStateGripperFormat = "physical_width",
) -> np.ndarray:
    """Normalize an 8D EEF dataset state into the shared canonical space.

    ``libero_v3_eef`` already stores xyz+quaternion, but its last dimension is
    the physical scalar finger opening (roughly 0..0.04 metres).  Online raw
    LIBERO observations reach the same 0..1 open-high convention through
    :func:`_canonicalize_libero_state`; stored canonical states must do so too.
    """
    state = _coerce_canonical_libero_state(state).copy()
    if gripper_format == "physical_width":
        state[..., 7] = state[..., 7] / LIBERO_GRIPPER_OPENNESS_SCALE
    elif gripper_format != "open_fraction":
        raise ValueError(f"Unsupported LIBERO state gripper format: {gripper_format!r}.")
    state[..., 7] = np.clip(state[..., 7], 0.0, 1.0)
    return state


def _canonicalize_libero_actions(actions: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.shape[-1] != RAW_ACTION_DIM:
        raise ValueError(f"Expected LIBERO action dim {RAW_ACTION_DIM}, got {actions.shape}.")
    position = actions[..., :3]
    orientation = ee_pose_utils.rotation_vector_to_quaternion(actions[..., 3:6])
    gripper = actions[..., 6:7]
    return np.concatenate([position, orientation, gripper], axis=-1)


def _libero_raw_gripper_to_canonical_target(
    gripper: np.ndarray,
    *,
    gripper_actions_are_binary_targets: bool = False,
) -> np.ndarray:
    gripper = np.asarray(gripper, dtype=np.float32)
    if gripper_actions_are_binary_targets:
        if np.any((gripper < 0.0) | (gripper > 1.0)):
            raise ValueError(
                "Expected LIBERO gripper target values in [0, 1] when "
                "gripper_actions_are_binary_targets=True."
            )
        return gripper.astype(np.float32)
    return np.where(gripper < 0.0, 1.0, 0.0).astype(np.float32)


def _canonical_ee_deltas_to_absolute(
    actions: np.ndarray,
    state: np.ndarray,
    *,
    gripper_actions_are_binary_targets: bool = False,
) -> np.ndarray:
    """Convert 8D EEF commands into canonical absolute pose targets.

    Position and quaternion are deltas.  The final value is an absolute
    gripper command/target, not a delta: legacy LIBERO uses -1=open,+1=close,
    while converted datasets may use 1=open,0=close.
    """
    actions = np.asarray(actions, dtype=np.float32)
    state = _coerce_canonical_libero_state(state)
    if actions.shape[-1] != CANONICAL_ACTION_DIM:
        raise ValueError(
            f"Expected canonical LIBERO action dim {CANONICAL_ACTION_DIM}, got {actions.shape}."
        )

    delta_position = actions[..., :3]
    delta_quaternion = ee_pose_utils.canonicalize_quaternion_sign(actions[..., 3:7])
    target_position = state[..., None, :3] + delta_position
    target_orientation = ee_pose_utils.quaternion_apply_delta(
        state[..., None, 3:7], delta_quaternion
    )
    target_gripper = _libero_raw_gripper_to_canonical_target(
        actions[..., 7:8],
        gripper_actions_are_binary_targets=gripper_actions_are_binary_targets,
    )
    return np.concatenate([target_position, target_orientation, target_gripper], axis=-1)


def _command_to_libero_absolute_actions(
    actions: np.ndarray,
    state: np.ndarray,
    *,
    gripper_actions_are_binary_targets: bool = False,
) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    state = _coerce_canonical_libero_state(state)
    if actions.shape[-1] != RAW_ACTION_DIM:
        raise ValueError(f"Expected LIBERO action dim {RAW_ACTION_DIM}, got {actions.shape}.")

    delta_position = actions[..., :3]
    delta_quaternion = ee_pose_utils.rotation_vector_to_quaternion(actions[..., 3:6])
    target_position = state[..., None, :3] + delta_position
    target_orientation = ee_pose_utils.quaternion_apply_delta(state[..., None, 3:7], delta_quaternion)
    target_gripper = _libero_raw_gripper_to_canonical_target(
        actions[..., 6:7],
        gripper_actions_are_binary_targets=gripper_actions_are_binary_targets,
    )
    return np.concatenate([target_position, target_orientation, target_gripper], axis=-1)


def _uncanonicalize_libero_actions(actions: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.shape[-1] != CANONICAL_ACTION_DIM:
        raise ValueError(f"Expected canonical LIBERO action dim {CANONICAL_ACTION_DIM}, got {actions.shape}.")
    position = actions[..., :3]
    orientation = ee_pose_utils.quaternion_to_rotation_vector(actions[..., 3:7])
    gripper = actions[..., 7:8]
    return np.concatenate([position, orientation, gripper], axis=-1)


def _absolute_to_libero_command(actions: np.ndarray, state: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    state = _coerce_canonical_libero_state(state)
    if actions.shape[-1] != CANONICAL_ACTION_DIM:
        raise ValueError(f"Expected canonical LIBERO action dim {CANONICAL_ACTION_DIM}, got {actions.shape}.")

    delta_position = actions[..., :3] - state[..., None, :3]
    delta_orientation = ee_pose_utils.quaternion_to_rotation_vector(
        ee_pose_utils.quaternion_delta(state[..., None, 3:7], actions[..., 3:7])
    )
    # LIBERO eval uses the robosuite-style command sign: negative opens, positive closes.
    gripper_command = np.where(actions[..., 7:8] >= 0.5, -1.0, 1.0).astype(np.float32)
    return np.concatenate([delta_position, delta_orientation, gripper_command], axis=-1)

@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    """Inputs for the Libero policy.

    Set ``pretrain_world_model=True`` for vision(-language) pretraining
    where no state or actions are available.  State is zeroed and actions
    are omitted; a missing wrist image is filled with zeros.
    """

    model_type: _model.ModelType = _model.ModelType.PI0
    pretrain_world_model: bool = False
    action_dim: int = 32
    enable_world_model: bool = True
    image_keys: Sequence[str] = ("observation/image",)
    canonicalize_ee_pose_gripper: bool = False
    treat_actions_as_commands: bool = False
    dataset_state_gripper_format: LiberoStateGripperFormat = "physical_width"
    dataset_action_gripper_format: LiberoActionGripperFormat = "signed_command"

    def __call__(self, data: dict) -> dict:
        # currently only support one image key, which should be the key of the image to be used for the world model
        key = self.image_keys[0] if self.image_keys else "observation/image"
        history_images = None
        future_images = None
        history_mask = None
        future_mask = None
        if self.enable_world_model and f"{key}_current" in data:
            base_image = _parse_image(data[f"{key}_current"])
            history_images = _parse_image(data[f"{key}_history"])
            future_images = _parse_image(data[f"{key}_future"])
            history_mask, future_mask = transforms.split_temporal_valid_mask(
                data,
                key,
                history_len=history_images.shape[0],
                future_len=future_images.shape[0],
            )
        else:
            base_image = _parse_image(data["observation/image"])

        if self.pretrain_world_model:
            wrist_image = _parse_image(data["observation/wrist_image"]) if "observation/wrist_image" in data else np.zeros_like(base_image)
        else:
            wrist_image = _parse_image(data["observation/wrist_image"])

        if self.pretrain_world_model:
            state = np.zeros(self.action_dim, dtype=np.float32)
        else:
            state = np.asarray(data["observation/state"], dtype=np.float32)
            if self.canonicalize_ee_pose_gripper:
                # Detect stored xyz+quaternion states directly.  Action
                # presence is not a valid discriminator because inference
                # requests contain no actions.
                if _is_canonical_libero_state(state):
                    state = _canonicalize_stored_libero_state(
                        state,
                        gripper_format=self.dataset_state_gripper_format,
                    )
                else:
                    state = _canonicalize_libero_state(state)

        images = {
            "base_0_rgb": base_image,
            "left_wrist_0_rgb": wrist_image,
            "right_wrist_0_rgb": np.zeros_like(wrist_image),
        }
        image_masks = {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_ if not self.pretrain_world_model or "observation/wrist_image" in data else np.False_,
            "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
        }
        if history_images is not None:
            images["base_0_rgb_history"] = history_images
            image_masks["base_0_rgb_history"] = history_mask
        if future_images is not None:
            images["base_0_rgb_future"] = future_images
            image_masks["base_0_rgb_future"] = future_mask

        inputs = {
            "state": state,
            "image": images,
            "image_mask": image_masks,
        }  

        if not self.pretrain_world_model:
            raw_actions = data["actions"] if "actions" in data else data.get("action")
            if raw_actions is not None:
                actions = np.asarray(raw_actions, dtype=np.float32)
            else:
                actions = None
        else:
            actions = None
        if actions is not None:
            if self.canonicalize_ee_pose_gripper:
                # Detect whether actions are already canonical (8-dim) or raw (7-dim).
                is_already_canonical = actions.shape[-1] == CANONICAL_ACTION_DIM
                if self.dataset_action_gripper_format == "absolute_physical_width":
                    if not is_already_canonical:
                        raise ValueError(
                            "absolute_physical_width LIBERO actions must already be 8D "
                            f"xyz+quaternion+gripper targets, got {actions.shape}."
                        )
                    actions = actions.copy()
                    actions[..., 7:8] = np.clip(
                        actions[..., 7:8] / LIBERO_GRIPPER_OPENNESS_SCALE,
                        0.0,
                        1.0,
                    )
                elif self.treat_actions_as_commands:
                    if is_already_canonical:
                        # EEF pose deltas plus an absolute gripper command →
                        # canonical absolute targets.  The downstream shared
                        # DeltaActions transform then produces true deltas.
                        actions = _canonical_ee_deltas_to_absolute(
                            actions,
                            state,
                            gripper_actions_are_binary_targets=self.dataset_action_gripper_format == "binary_target",
                        )
                    else:
                        actions = _command_to_libero_absolute_actions(
                            actions,
                            state,
                            gripper_actions_are_binary_targets=self.dataset_action_gripper_format == "binary_target",
                        )
                else:
                    if is_already_canonical:
                        # Already canonical deltas — pass through unchanged.
                        pass
                    else:
                        actions = _canonicalize_libero_actions(actions)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    """Outputs for the Libero policy."""

    pretrain_world_model: bool = False
    canonicalize_ee_pose_gripper: bool = False
    treat_actions_as_commands: bool = False

    def __call__(self, data: dict) -> dict:
        if self.pretrain_world_model:
            return data
        actions = np.asarray(data["actions"])
        if self.canonicalize_ee_pose_gripper:
            if self.treat_actions_as_commands:
                state = np.asarray(data["state"], dtype=np.float32)
                return {"actions": _absolute_to_libero_command(actions[:, :CANONICAL_ACTION_DIM], state)}
            return {"actions": _uncanonicalize_libero_actions(actions[:, :CANONICAL_ACTION_DIM])}
        # Only return the first 7 actions (the rest is padding).
        return {"actions": actions[:, :RAW_ACTION_DIM]}
