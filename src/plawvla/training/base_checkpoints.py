# Copyright 2026 PLaW-VLA authors. SPDX-License-Identifier: Apache-2.0
"""Public initialization checkpoints.

Add another π₀.₅ JAX checkpoint by appending one entry, then select it with
``BASE_CHECKPOINT=<name>``. Stage I downloads ``jax_uri`` and converts it into
``checkpoints/<pytorch_dirname>`` using the Stage I architecture (π₀.₅
PaliGemma plus the Gemma 300M action expert). Stage II and Stage III load the
previous stage's ``model.safetensors`` rather than this table.

``pi05`` must stay true. A π₀ checkpoint does not have the projection layers
Stage I loads.

``STAGE3_INITIALIZATIONS`` is the stable registry used by the direct LIBERO
fine-tuning entrypoint.  The registry, rather than a network existence check,
decides what ``auto`` means for a given repository release.  Until the
official PLaW-VLA pretrained checkpoint is published, ``auto`` resolves to
``pi05``.  Publishing the checkpoint only requires filling its source and
marking that registry entry released; the user-facing command stays unchanged.
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
    # Directory written by scripts/setup/convert_jax_model_to_pytorch.py, under checkpoints/.
    pytorch_dirname: str
    # Stage I is a π₀.₅ model. Entries with pi05=False are rejected.
    pi05: bool = True


@dataclasses.dataclass(frozen=True)
class Stage3Initialization:
    """One supported initialization source for direct Stage III fine-tuning."""

    name: str
    weight_load_mode: str
    released: bool
    description: str
    # π₀.₅ entries delegate to BASE_CHECKPOINTS so their JAX conversion remains
    # shared with Stage I.
    base_checkpoint: str | None = None
    # Official PLaW-VLA weights may be published as a generic filesystem URI or
    # as a Hugging Face model repository. Exactly one is set when released.
    pytorch_dirname: str | None = None
    uri: str | None = None
    hf_repo_id: str | None = None


BASE_CHECKPOINTS: dict[str, BaseCheckpoint] = {
    "pi05_base": BaseCheckpoint(
        name="pi05_base",
        jax_uri="gs://openpi-assets/checkpoints/pi05_base",
        pytorch_dirname="pi05_base_pytorch",
        pi05=True,
    ),
}

STAGE3_INITIALIZATIONS: dict[str, Stage3Initialization] = {
    "pi05": Stage3Initialization(
        name="pi05",
        weight_load_mode="foundation",
        released=True,
        base_checkpoint="pi05_base",
        pytorch_dirname="pi05_base_pytorch",
        description=(
            "Public π₀.₅ foundation weights. PaliGemma and the action expert are loaded; "
            "PLaW-VLA world-model modules use recipe initialization."
        ),
    ),
    "plawvla": Stage3Initialization(
        name="plawvla",
        weight_load_mode="full",
        released=False,
        description=(
            "Official PLaW-VLA pretrained base checkpoint with the complete world model. "
            "This entry is reserved for the forthcoming weight release."
        ),
        pytorch_dirname="plawvla_base_pytorch",
        # Fill `uri` or `hf_repo_id` and set released=True when published.
    ),
}

# LeRobot dataset used for Stage III LIBERO fine-tuning.
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


def get_stage3_initialization(name: str) -> Stage3Initialization:
    try:
        return STAGE3_INITIALIZATIONS[name]
    except KeyError as exc:
        known = ", ".join(("auto", *STAGE3_INITIALIZATIONS))
        raise KeyError(f"Unknown Stage III base source {name!r}. Known sources: {known}.") from exc


def resolve_stage3_initialization(requested: str = "auto") -> Stage3Initialization:
    """Resolve the reproducible initialization selected by this code release."""

    if requested == "auto":
        official = get_stage3_initialization("plawvla")
        return official if official.released else get_stage3_initialization("pi05")

    spec = get_stage3_initialization(requested)
    if not spec.released:
        raise ValueError(
            f"Stage III base source {requested!r} is not released in this repository version. "
            "Use STAGE3_BASE_SOURCE=pi05 or leave STAGE3_BASE_SOURCE=auto. "
            "The same auto command will prefer the official PLaW-VLA base after its registry entry is released."
        )
    return spec
