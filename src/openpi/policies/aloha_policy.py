# Derived from openpi (Copyright 2024 Physical Intelligence, Inc.; Apache-2.0).
# Modified for PLaW-VLA by the PLaW-VLA authors, 2026.
import dataclasses
from collections.abc import Sequence
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def make_aloha_example() -> dict:
    """Creates a random input example for the Aloha policy."""
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
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


def _lookup_image(data: dict, key: str):
    if key in data:
        return data[key]
    images = data.get("images")
    if isinstance(images, dict) and key in images:
        return images[key]
    raise KeyError(key)


def _has_image(data: dict, key: str) -> bool:
    if key in data:
        return True
    images = data.get("images")
    return isinstance(images, dict) and key in images


def _validate_native_aloha_dim(values: np.ndarray, *, native_action_dim: int, value_name: str) -> None:
    if values.shape[-1] != native_action_dim:
        raise ValueError(f"Expected native Aloha {value_name} dim {native_action_dim}, got {values.shape[-1]}.")
    if native_action_dim != 14:
        raise ValueError(
            f"adapt_to_pi=True only supports canonical 14D Aloha joint {value_name}, "
            f"got native_action_dim={native_action_dim}."
        )


@dataclasses.dataclass(frozen=True)
class AlohaInputs(transforms.DataTransformFn):
    """Inputs for the Aloha policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [native_action_dim]
    - actions: [action_horizon, native_action_dim]

    Set ``pretrain_world_model=True`` for vision(-language) pretraining
    where no state or actions are available.  State is zeroed and actions
    are omitted; missing wrist cameras are filled with zeros.
    """

    adapt_to_pi: bool = True
    pretrain_world_model: bool = False
    action_dim: int = 32
    native_action_dim: int = 14
    enable_world_model: bool = True
    image_keys: Sequence[str] = ("cam_high",)

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        if not self.pretrain_world_model:
            data = _decode_aloha(data, adapt_to_pi=self.adapt_to_pi, native_action_dim=self.native_action_dim)

        key = self.image_keys[0] if self.image_keys else "cam_high"
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
            base_image = _parse_image(_lookup_image(data, key))

        if self.pretrain_world_model:
            left_wrist = (
                _parse_image(_lookup_image(data, "cam_left_wrist"))
                if _has_image(data, "cam_left_wrist")
                else np.zeros_like(base_image)
            )
            right_wrist = (
                _parse_image(_lookup_image(data, "cam_right_wrist"))
                if _has_image(data, "cam_right_wrist")
                else np.zeros_like(base_image)
            )
        else:
            left_wrist = _parse_image(_lookup_image(data, "cam_left_wrist"))
            right_wrist = _parse_image(_lookup_image(data, "cam_right_wrist"))

        images = {
            "base_0_rgb": base_image,
            "left_wrist_0_rgb": left_wrist,
            "right_wrist_0_rgb": right_wrist,
        }
        image_masks = {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_
            if not self.pretrain_world_model or _has_image(data, "cam_left_wrist")
            else np.False_,
            "right_wrist_0_rgb": np.True_
            if not self.pretrain_world_model or _has_image(data, "cam_right_wrist")
            else np.False_,
        }
        if history_images is not None:
            images["base_0_rgb_history"] = history_images
            image_masks["base_0_rgb_history"] = history_mask
        if future_images is not None:
            images["base_0_rgb_future"] = future_images
            image_masks["base_0_rgb_future"] = future_mask

        if self.pretrain_world_model:
            state = np.zeros(self.action_dim, dtype=np.float32)
        else:
            state = data["state"]

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        if not self.pretrain_world_model and "actions" in data:
            actions = np.asarray(data["actions"])
            actions = _encode_actions_inv(
                actions,
                adapt_to_pi=self.adapt_to_pi,
                native_action_dim=self.native_action_dim,
            )
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class AlohaOutputs(transforms.DataTransformFn):
    """Outputs for the Aloha policy."""

    adapt_to_pi: bool = True
    pretrain_world_model: bool = False
    native_action_dim: int = 14

    def __call__(self, data: dict) -> dict:
        if self.pretrain_world_model:
            return data
        actions = np.asarray(data["actions"][:, : self.native_action_dim])
        return {
            "actions": _encode_actions(
                actions,
                adapt_to_pi=self.adapt_to_pi,
                native_action_dim=self.native_action_dim,
            )
        }


def _joint_flip_mask() -> np.ndarray:
    """Used to convert between aloha and pi joint angles."""
    return np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])


def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def _gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with pi0 which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = _unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return np.arcsin(np.clip(value, -1.0, 1.0))

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # pi0 gripper data is normalized (0, 1) between encoder counts (2405, 3110).
    # There are 4096 total encoder counts and aloha uses a zero of 2048.
    # Converting this to radians means that the normalized inputs are between (0.5476, 1.6296)
    return _normalize(value, min_val=0.5476, max_val=1.6296)


def _gripper_from_angular(value):
    # Convert from the gripper position used by pi0 to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # We do not scale the output since the trossen model predictions are already in radians.
    # See the comment in _gripper_to_angular for a derivation of the constant
    value = value + 0.5476

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return _normalize(value, min_val=-0.6213, max_val=1.4910)


def _gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = _unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return value - 0.5476


def _decode_aloha(data: dict, *, adapt_to_pi: bool = False, native_action_dim: int = 14) -> dict:
    # state is [left_arm_joint_angles, left_arm_gripper, right_arm_joint_angles, right_arm_gripper]
    # dim sizes: [6, 1, 6, 1]
    state = np.asarray(data["state"])
    state = _decode_state(state, adapt_to_pi=adapt_to_pi, native_action_dim=native_action_dim)
    data["state"] = state
    return data


def _decode_state(state: np.ndarray, *, adapt_to_pi: bool = False, native_action_dim: int = 14) -> np.ndarray:
    if adapt_to_pi:
        _validate_native_aloha_dim(state, native_action_dim=native_action_dim, value_name="state")
        # Flip the joints.
        state = _joint_flip_mask() * state
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        state[[6, 13]] = _gripper_to_angular(state[[6, 13]])
    return state


def _encode_actions(actions: np.ndarray, *, adapt_to_pi: bool = False, native_action_dim: int = 14) -> np.ndarray:
    if adapt_to_pi:
        _validate_native_aloha_dim(actions, native_action_dim=native_action_dim, value_name="action")
        # Flip the joints.
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular(actions[:, [6, 13]])
    return actions


def _encode_actions_inv(actions: np.ndarray, *, adapt_to_pi: bool = False, native_action_dim: int = 14) -> np.ndarray:
    if adapt_to_pi:
        _validate_native_aloha_dim(actions, native_action_dim=native_action_dim, value_name="action")
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular_inv(actions[:, [6, 13]])
    return actions
