# Copyright 2026 PLaW-VLA authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Literal

import einops
import numpy as np

from plawvla import transforms
from plawvla.models import model as _model

CANONICAL_STATE_DIM = 16
CANONICAL_ACTION_DIM = 16

CAM_HIGH_KEY = "observation.images.cam_high"
CAM_LEFT_WRIST_KEY = "observation.images.cam_left_wrist"
CAM_RIGHT_WRIST_KEY = "observation.images.cam_right_wrist"
STATE_KEY = "observation.state"
STATE_MASK_KEY = "observation.state_mask"
ACTION_KEY = "actions"
ACTION_MASK_KEY = "actions_mask"
IMAGE_MASK_KEY = "observation.image_mask"


def make_intern_a1_example() -> dict:
    """Create a canonical InternData-A1 input example."""
    return {
        CAM_HIGH_KEY: np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        CAM_LEFT_WRIST_KEY: np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        CAM_RIGHT_WRIST_KEY: np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        STATE_KEY: np.random.randn(CANONICAL_STATE_DIM).astype(np.float32),
        IMAGE_MASK_KEY: np.array([True, True, True], dtype=bool),
        "prompt": "pick up the object",
    }


def _parse_image(image) -> np.ndarray:
    if hasattr(image, "cpu") and hasattr(image, "numpy"):
        image = image.cpu().numpy()

    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)

    if image.ndim == 4 and image.shape[1] == 3:
        image = einops.rearrange(image, "t c h w -> t h w c")
    elif image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")

    return image


def _lookup_key(data: Mapping[str, object], key: str) -> object | None:
    if key in data:
        return data[key]

    if key.startswith("observation.images.") and "images" in data:
        short_key = key.removeprefix("observation.images.")
        images = data["images"]
        if isinstance(images, Mapping) and short_key in images:
            return images[short_key]

    short_aliases = {
        STATE_KEY: "state",
        STATE_MASK_KEY: "state_mask",
        ACTION_KEY: "actions",
        ACTION_MASK_KEY: "actions_mask",
        IMAGE_MASK_KEY: "image_mask",
    }
    alias = short_aliases.get(key)
    if alias is not None and alias in data:
        return data[alias]

    return None


def _normalize_vector_mask(mask: object | None, expected_dim: int) -> np.ndarray | None:
    if mask is None:
        return None

    value = np.asarray(mask, dtype=bool).reshape(-1)
    if value.size != expected_dim:
        raise ValueError(f"Expected mask with {expected_dim} entries, got {value.size}.")
    return value


def _apply_mask(values: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return values
    if values.ndim == 1:
        return np.where(mask, values, 0)
    return np.where(mask[np.newaxis, :], values, 0)


def _decode_prompt(prompt: object) -> str:
    if isinstance(prompt, bytes):
        return prompt.decode("utf-8")
    return str(prompt)


@dataclasses.dataclass(frozen=True)
class InternA1Inputs(transforms.DataTransformFn):
    """Transform canonical InternData-A1 samples into model inputs."""

    model_type: _model.ModelType = _model.ModelType.PI0
    action_dim: int = CANONICAL_ACTION_DIM
    pretrain_world_model: bool = False
    enable_world_model: bool = False
    image_keys: Sequence[str] = (CAM_HIGH_KEY,)
    default_prompt: str = ""
    state_semantics: Literal["joint_position", "ee_pose"] = "joint_position"

    def _model_camera_names(self) -> tuple[str, str, str]:
        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                return ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
            case _model.ModelType.PI0_FAST:
                return ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

    def _load_base_images(self, data: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        base_key = self.image_keys[0] if self.image_keys else CAM_HIGH_KEY
        if self.enable_world_model and f"{base_key}_current" in data:
            base_image = _parse_image(data[f"{base_key}_current"])
            history_images = _parse_image(data[f"{base_key}_history"])
            future_images = _parse_image(data[f"{base_key}_future"])
            return base_image, history_images, future_images

        base_image = _lookup_key(data, base_key)
        if base_image is None:
            raise ValueError(f"Required camera {base_key!r} is missing.")
        return _parse_image(base_image), None, None

    def __call__(self, data: dict) -> dict:
        base_image, history_images, future_images = self._load_base_images(data)
        history_mask = None
        future_mask = None
        if history_images is not None and future_images is not None:
            base_key = self.image_keys[0] if self.image_keys else CAM_HIGH_KEY
            history_mask, future_mask = transforms.split_temporal_valid_mask(
                data,
                base_key,
                history_len=history_images.shape[0],
                future_len=future_images.shape[0],
            )
        image_presence = _normalize_vector_mask(_lookup_key(data, IMAGE_MASK_KEY), 3)
        if image_presence is not None and not image_presence[0]:
            raise ValueError("observation.images.cam_high cannot be masked out.")

        left_image = _lookup_key(data, CAM_LEFT_WRIST_KEY)
        right_image = _lookup_key(data, CAM_RIGHT_WRIST_KEY)

        left_valid = bool(image_presence[1]) if image_presence is not None else left_image is not None
        right_valid = bool(image_presence[2]) if image_presence is not None else right_image is not None

        left_wrist = _parse_image(left_image) if left_image is not None and left_valid else np.zeros_like(base_image)
        right_wrist = _parse_image(right_image) if right_image is not None and right_valid else np.zeros_like(base_image)

        image_names = self._model_camera_names()
        images = {
            image_names[0]: base_image,
            image_names[1]: left_wrist,
            image_names[2]: right_wrist,
        }
        image_masks = {
            image_names[0]: np.True_,
            image_names[1]: np.bool_(left_valid),
            image_names[2]: np.bool_(right_valid),
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
            state = _lookup_key(data, STATE_KEY)
            if state is None:
                raise ValueError(f"Required state key {STATE_KEY!r} is missing.")
            state = np.asarray(state, dtype=np.float32)
            state_mask = _normalize_vector_mask(_lookup_key(data, STATE_MASK_KEY), state.shape[-1])
            state = _apply_mask(state, state_mask)

        inputs = {
            "state": state,
            "image": images,
            "image_mask": image_masks,
        }

        if not self.pretrain_world_model:
            actions = _lookup_key(data, ACTION_KEY)
            if actions is not None:
                actions = np.asarray(actions, dtype=np.float32)
                action_mask = _normalize_vector_mask(_lookup_key(data, ACTION_MASK_KEY), actions.shape[-1])
                inputs["actions"] = _apply_mask(actions, action_mask)

        prompt = _lookup_key(data, "prompt")
        if prompt:
            inputs["prompt"] = _decode_prompt(prompt)
        elif self.default_prompt:
            inputs["prompt"] = self.default_prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class InternA1Outputs(transforms.DataTransformFn):
    """Return canonical 16D InternData-A1 actions."""

    action_dim: int = CANONICAL_ACTION_DIM
    pretrain_world_model: bool = False
    state_semantics: Literal["joint_position", "ee_pose"] = "joint_position"

    def __call__(self, data: dict) -> dict:
        if self.pretrain_world_model:
            return data
        actions = np.asarray(data["actions"], dtype=np.float32)[..., : self.action_dim]
        return {"actions": actions}
