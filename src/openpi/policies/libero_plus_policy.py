# Copyright 2026 PLaW-VLA authors. SPDX-License-Identifier: Apache-2.0
import dataclasses
from collections.abc import Sequence

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.policies import libero_policy


def make_libero_plus_example() -> dict:
    """Creates a random input example for the Libero Plus policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/front_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
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


@dataclasses.dataclass(frozen=True)
class LiberoPlusInputs(transforms.DataTransformFn):
    """Inputs for the Libero Plus policy."""

    model_type: _model.ModelType = _model.ModelType.PI0
    pretrain_world_model: bool = False
    action_dim: int = 32
    enable_world_model: bool = True
    image_keys: Sequence[str] = ("observation/front_image",)
    canonicalize_ee_pose_gripper: bool = False
    treat_actions_as_commands: bool = True

    def __call__(self, data: dict) -> dict:
        key = self.image_keys[0] if self.image_keys else "observation/front_image"
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
            base_image = _parse_image(data["observation/front_image"])

        if self.pretrain_world_model:
            wrist_image = (
                _parse_image(data["observation/wrist_image"])
                if "observation/wrist_image" in data
                else np.zeros_like(base_image)
            )
        else:
            wrist_image = _parse_image(data["observation/wrist_image"])

        if self.pretrain_world_model:
            state = np.zeros(self.action_dim, dtype=np.float32)
        else:
            state = np.asarray(data["observation/state"], dtype=np.float32)
            if self.canonicalize_ee_pose_gripper:
                state = libero_policy._canonicalize_libero_state(state)

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
                if self.treat_actions_as_commands:
                    actions = libero_policy._command_to_libero_absolute_actions(actions, state)
                else:
                    actions = libero_policy._canonicalize_libero_actions(actions)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoPlusOutputs(transforms.DataTransformFn):
    """Outputs for the Libero Plus policy."""

    pretrain_world_model: bool = False
    canonicalize_ee_pose_gripper: bool = False
    treat_actions_as_commands: bool = True

    def __call__(self, data: dict) -> dict:
        if self.pretrain_world_model:
            return data
        actions = np.asarray(data["actions"])
        if self.canonicalize_ee_pose_gripper:
            if self.treat_actions_as_commands:
                state = np.asarray(data["state"], dtype=np.float32)
                commands = libero_policy._absolute_to_libero_command(actions[:, :libero_policy.CANONICAL_ACTION_DIM], state)
                commands[..., 6:7] = np.where(commands[..., 6:7] > 0.5, 1.0, -1.0).astype(np.float32)
                return {"actions": commands}
            return {"actions": libero_policy._uncanonicalize_libero_actions(actions[:, :libero_policy.CANONICAL_ACTION_DIM])}
        return {"actions": actions[:, :libero_policy.RAW_ACTION_DIM]}
