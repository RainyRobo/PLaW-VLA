import torch

from openpi.models.model import Observation
from openpi.models_pytorch.preprocessing_pytorch import preprocess_observation_pytorch


def _make_observation(image: torch.Tensor) -> Observation[torch.Tensor]:
    return Observation(
        images={
            "base_0_rgb": image.clone(),
            "left_wrist_0_rgb": image.clone(),
            "right_wrist_0_rgb": image.clone(),
        },
        image_masks={
            "base_0_rgb": torch.ones((image.shape[0],), dtype=torch.bool),
            "left_wrist_0_rgb": torch.ones((image.shape[0],), dtype=torch.bool),
            "right_wrist_0_rgb": torch.ones((image.shape[0],), dtype=torch.bool),
        },
        state=torch.zeros((image.shape[0], 8), dtype=torch.float32),
    )


def test_preprocess_observation_pytorch_converts_nhwc_to_nchw():
    image = torch.arange(2 * 224 * 224 * 3, dtype=torch.float32).reshape(2, 224, 224, 3)

    processed = preprocess_observation_pytorch(_make_observation(image), train=False)

    assert processed.images["base_0_rgb"].shape == (2, 3, 224, 224)
    torch.testing.assert_close(processed.images["base_0_rgb"], image.permute(0, 3, 1, 2))


def test_preprocess_observation_pytorch_preserves_nchw_layout():
    image = torch.arange(2 * 3 * 224 * 224, dtype=torch.float32).reshape(2, 3, 224, 224)

    processed = preprocess_observation_pytorch(_make_observation(image), train=False)

    assert processed.images["base_0_rgb"].shape == (2, 3, 224, 224)
    torch.testing.assert_close(processed.images["base_0_rgb"], image)
