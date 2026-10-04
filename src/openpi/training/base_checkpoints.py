"""Public base checkpoints.

Add another π₀.₅ JAX checkpoint by appending one entry, then select it with
``BASE_CHECKPOINT=<name>``. Stage I downloads ``jax_uri`` and converts it into
``checkpoints/<pytorch_dirname>`` using the Stage I architecture (π₀.₅
PaliGemma plus the Gemma 300M action expert). Stage II and Stage III load the
previous stage's ``model.safetensors`` rather than this table.

``pi05`` must stay true. A π₀ checkpoint does not have the projection layers
Stage I loads.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class BaseCheckpoint:
    """One downloadable foundation checkpoint."""

    name: str
    # Source checkpoint URI. Must be an OpenPI π₀.₅ JAX checkpoint (a directory
    # that contains params/).
    jax_uri: str
    # Directory written by examples/convert_jax_model_to_pytorch.py, under checkpoints/.
    pytorch_dirname: str
    # Stage I is a π₀.₅ model. Entries with pi05=False are rejected.
    pi05: bool = True


BASE_CHECKPOINTS: dict[str, BaseCheckpoint] = {
    "pi05_base": BaseCheckpoint(
        name="pi05_base",
        jax_uri="gs://openpi-assets/checkpoints/pi05_base",
        pytorch_dirname="pi05_base_pytorch",
        pi05=True,
    ),
}

# LeRobot dataset used by the released three-stage recipe.
DEFAULT_DATASET_REPO = "RainyBot/libero_v3_eef"
DEFAULT_DATASET_DIR = "data/libero_v3_eef"
VJEPA2_REPO = "facebook/vjepa2-vitl-fpc64-256"
PALIGEMMA_TOKENIZER = "gs://big_vision/paligemma_tokenizer.model"


def get_base_checkpoint(name: str = "pi05_base") -> BaseCheckpoint:
    try:
        return BASE_CHECKPOINTS[name]
    except KeyError as exc:
        known = ", ".join(BASE_CHECKPOINTS)
        raise KeyError(f"Unknown base checkpoint {name!r}. Known checkpoints: {known}.") from exc
