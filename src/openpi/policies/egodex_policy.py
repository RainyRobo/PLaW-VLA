# Copyright 2026 PLaW-VLA authors. SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_egodex_example() -> dict:
    """Creates a random input example for the EgoDex policy."""
    return {
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "Pick up the bottle and place it in the bin.",
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
class EgoDexDeltaActions(transforms.DataTransformFn):
    """Convert EgoDex future absolute chunks into current-camera-frame deltas."""

    camera_extrinsic_key: str = "camera_extrinsic"

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data
        if "state" not in data:
            raise KeyError("EgoDexDeltaActions requires `state` in the input example.")
        if self.camera_extrinsic_key not in data:
            raise KeyError(f"EgoDexDeltaActions requires `{self.camera_extrinsic_key}` in the input example.")

        from openpi.datasets.specs import egodex as egodex_spec

        state = np.asarray(data["state"], dtype=np.float32)
        future_actions = np.asarray(data["actions"], dtype=np.float32)
        camera_extrinsic = np.asarray(data[self.camera_extrinsic_key], dtype=np.float32)
        absolute_chunk = np.concatenate([state[np.newaxis, :], future_actions], axis=0)
        data["actions"] = egodex_spec.convert_to_delta_actions(
            absolute_chunk,
            absolute_chunk.shape[0],
            camera_extrinsic,
        ).astype(np.float32, copy=False)
        return data


@dataclasses.dataclass(frozen=True)
class EgoDexInputs(transforms.DataTransformFn):
    """Transform EgoDex data into the model input format.

    EgoDex has a single egocentric camera. The converted dataset provides the
    current absolute 48 DoF hand state and a future `(16, 48)` action chunk in
    absolute space. If a training config explicitly enables delta actions, that
    conversion happens before this policy transform. The egocentric image is
    mapped to base_0_rgb while wrist cameras are masked out. When
    ``pretrain_world_model=True``, state is zeroed to the model action
    dimension and actions are omitted.
    """

    model_type: _model.ModelType = _model.ModelType.PI0
    action_dim: int = 48
    pretrain_world_model: bool = False
    enable_world_model: bool = False
    image_keys: Sequence[str] = ()

    def __call__(self, data: dict) -> dict:
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
            base_image = _parse_image(data[key])

        placeholder = np.zeros_like(base_image)
        if self.pretrain_world_model:
            state = np.zeros(self.action_dim, dtype=np.float32)
        else:
            state = (
                np.asarray(data["state"], dtype=np.float32)
                if "state" in data
                else np.zeros(self.action_dim, dtype=np.float32)
            )

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, placeholder, placeholder)
                image_masks = (np.True_, np.False_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (base_image, placeholder, placeholder)
                image_masks = (np.True_, np.False_, np.False_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        image_dict = dict(zip(names, images, strict=True))
        mask_dict = dict(zip(names, image_masks, strict=True))

        if history_images is not None:
            image_dict["base_0_rgb_history"] = history_images
            mask_dict["base_0_rgb_history"] = history_mask
        if future_images is not None:
            image_dict["base_0_rgb_future"] = future_images
            mask_dict["base_0_rgb_future"] = future_mask

        inputs = {
            "state": state,
            "image": image_dict,
            "image_mask": mask_dict,
        }

        if not self.pretrain_world_model and "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            prompt = data["prompt"].decode("utf-8") if isinstance(data["prompt"], bytes) else data["prompt"]
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class EgoDexOutputs(transforms.DataTransformFn):
    """Outputs for the EgoDex policy."""

    action_dim: int = 48
    pretrain_world_model: bool = False

    def __call__(self, data: dict) -> dict:
        if self.pretrain_world_model:
            return data
        return {"actions": np.asarray(data["actions"][:, :self.action_dim])}
