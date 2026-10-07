# Copyright 2026 PLaW-VLA authors. SPDX-License-Identifier: Apache-2.0
import dataclasses
from collections.abc import Sequence
from typing import Literal

import einops
import numpy as np
import torch

from openpi import transforms

_AGIBOT_EE_POSE_GRIPPER_INDICES = (7, 15)
_AGIBOT_JOINT_EFFECTOR_GRIPPER_INDICES = (14, 15)


def make_agibot_example() -> dict:
    """Creates a random input example for the AGIBot policy."""
    return {
        "top_head": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "hand_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "hand_right": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "state": np.random.rand(20),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    """Convert an image to uint8 HWC format."""
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()

    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)

    if image.ndim == 4 and image.shape[1] == 3:
        image = einops.rearrange(image, "t c h w -> t h w c")
    elif image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")

    return image


def _invert_unit_interval_gripper(values: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    remapped = np.asarray(values, dtype=np.float32).copy()
    for index in indices:
        if index < remapped.shape[-1]:
            remapped[..., index] = 1.0 - np.clip(remapped[..., index], 0.0, 1.0)
    return remapped


def _reverse_gripper_direction(values: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    remapped = np.asarray(values, dtype=np.float32).copy()
    for index in indices:
        if index < remapped.shape[-1]:
            remapped[..., index] = -remapped[..., index]
    return remapped


def _canonicalize_agibot_state(values: np.ndarray, *, state_semantics: str) -> np.ndarray:
    if state_semantics == "ee_pose":
        return _invert_unit_interval_gripper(values, _AGIBOT_EE_POSE_GRIPPER_INDICES)
    return _reverse_gripper_direction(values, _AGIBOT_JOINT_EFFECTOR_GRIPPER_INDICES)


def _canonicalize_agibot_actions(values: np.ndarray, *, state_semantics: str) -> np.ndarray:
    if state_semantics == "ee_pose":
        return _invert_unit_interval_gripper(values, _AGIBOT_EE_POSE_GRIPPER_INDICES)
    return _invert_unit_interval_gripper(values, _AGIBOT_JOINT_EFFECTOR_GRIPPER_INDICES)


def _uncanonicalize_agibot_actions(values: np.ndarray, *, state_semantics: str) -> np.ndarray:
    if state_semantics == "ee_pose":
        return _invert_unit_interval_gripper(values, _AGIBOT_EE_POSE_GRIPPER_INDICES)
    return _invert_unit_interval_gripper(values, _AGIBOT_JOINT_EFFECTOR_GRIPPER_INDICES)


@dataclasses.dataclass(frozen=True)
class AGIBotInputs(transforms.DataTransformFn):
    """Inputs for AGIBot robot policies.

    Accepts data in two formats:
    - Flat keys from training pipeline (after RepackTransform + SplitTemporalFrames):
        top_head_current, top_head_history, top_head_future, hand_left, hand_right, state, actions
    - Nested dict from inference:
        images: {top_head, hand_left, hand_right}, state, actions

    Set ``pretrain_world_model=True`` for vision(-language) pretraining
    where no state or actions are available.
    """

    action_dim: int
    pretrain_world_model: bool = False
    state_mask: np.ndarray | None = None
    action_mask: np.ndarray | None = None
    enable_world_model: bool = True
    image_keys: Sequence[str] = ("top_head",)
    # Number of native action dimensions to keep in output (e.g. 22 for Go1/Go2).
    native_action_dim: int = 22
    default_prompt: str = ""
    state_semantics: Literal["joint_effector_position", "ee_pose"] = "joint_effector_position"
    canonicalize_gripper_openness: bool = False

    def _lookup_image(self, data: dict, key: str) -> np.ndarray | None:
        """Look up an image from flat keys or nested images dict."""
        if key in data:
            return data[key]
        if "images" in data and key in data["images"]:
            return data["images"][key]
        return None

    def _require_or_zero(self, data: dict, key: str, ref_image: np.ndarray) -> tuple[np.ndarray, np.bool_]:
        image = self._lookup_image(data, key)
        if image is not None:
            return _parse_image(image), np.True_
        return np.zeros_like(ref_image), np.False_

    def __call__(self, data: dict) -> dict:
        key = self.image_keys[0] if self.image_keys else "top_head"
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
            image = self._lookup_image(data, key)
            if image is None:
                raise ValueError(f"Camera {key} not found in data")
            base_image = _parse_image(image)

        left_wrist, left_mask = self._require_or_zero(data, "hand_left", base_image)
        right_wrist, right_mask = self._require_or_zero(data, "hand_right", base_image)

        if self.pretrain_world_model:
            state = np.zeros(self.action_dim, dtype=np.float32)
        else:
            state = np.array(data["state"], copy=True)
            if self.canonicalize_gripper_openness:
                state = _canonicalize_agibot_state(state, state_semantics=self.state_semantics)
            if self.state_mask is not None:
                state[np.asarray(self.state_mask, dtype=bool)] = 0
            state = state.squeeze()

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist,
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": left_mask,
                "right_wrist_0_rgb": right_mask,
            },
        }
        if history_images is not None:
            inputs["image"]["base_0_rgb_history"] = history_images
            inputs["image_mask"]["base_0_rgb_history"] = history_mask
        if future_images is not None:
            inputs["image"]["base_0_rgb_future"] = future_images
            inputs["image_mask"]["base_0_rgb_future"] = future_mask

        if not self.pretrain_world_model and "actions" in data:
            actions = np.array(data["actions"], copy=True)
            if self.canonicalize_gripper_openness:
                actions = _canonicalize_agibot_actions(actions, state_semantics=self.state_semantics)
            if self.action_mask is not None:
                action_mask = np.asarray(self.action_mask, dtype=bool)
                actions[:, action_mask[:actions.shape[1]]] = 0
            inputs["actions"] = actions.squeeze()

        if "prompt" in data and data["prompt"]:
            inputs["prompt"] = data["prompt"]
        elif self.default_prompt:
            inputs["prompt"] = self.default_prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class AGIBotOutputs(transforms.DataTransformFn):
    """Outputs for AGIBot robot policies."""

    native_action_dim: int = 22
    pretrain_world_model: bool = False
    state_semantics: Literal["joint_effector_position", "ee_pose"] = "joint_effector_position"
    canonicalize_gripper_openness: bool = False

    def __call__(self, data: dict) -> dict:
        if self.pretrain_world_model:
            return data
        actions = np.asarray(data["actions"][:, :self.native_action_dim], dtype=np.float32)
        if self.canonicalize_gripper_openness:
            actions = _uncanonicalize_agibot_actions(actions, state_semantics=self.state_semantics)
        return {"actions": actions}
