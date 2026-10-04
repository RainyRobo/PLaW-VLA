"""Configuration helpers for V-JEPA 2 variants."""

from __future__ import annotations

import dataclasses
from typing import Literal

Variant = Literal["vitl-256", "vith-256", "vitg-256", "vitg-384"]


@dataclasses.dataclass(frozen=True)
class Config:
    hf_repo_id: str
    crop_size: int
    frames_per_clip: int
    tubelet_size: int
    patch_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    mlp_ratio: float


_VARIANTS: dict[Variant, Config] = {
    "vitl-256": Config(
        hf_repo_id="facebook/vjepa2-vitl-fpc64-256",
        crop_size=256,
        frames_per_clip=64,
        tubelet_size=2,
        patch_size=16,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        mlp_ratio=4.0,
    ),
    "vith-256": Config(
        hf_repo_id="facebook/vjepa2-vith-fpc64-256",
        crop_size=256,
        frames_per_clip=64,
        tubelet_size=2,
        patch_size=16,
        hidden_size=1280,
        num_hidden_layers=32,
        num_attention_heads=16,
        mlp_ratio=4.0,
    ),
    "vitg-256": Config(
        hf_repo_id="facebook/vjepa2-vitg-fpc64-256",
        crop_size=256,
        frames_per_clip=64,
        tubelet_size=2,
        patch_size=16,
        hidden_size=1408,
        num_hidden_layers=40,
        num_attention_heads=22,
        mlp_ratio=48 / 11,
    ),
    "vitg-384": Config(
        hf_repo_id="facebook/vjepa2-vitg-fpc64-384",
        crop_size=384,
        frames_per_clip=64,
        tubelet_size=2,
        patch_size=16,
        hidden_size=1408,
        num_hidden_layers=40,
        num_attention_heads=22,
        mlp_ratio=48 / 11,
    ),
}


def get_config(variant: Variant) -> Config:
    """Return the predefined configuration for a V-JEPA 2 variant."""
    try:
        return _VARIANTS[variant]
    except KeyError as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"Unknown V-JEPA 2 variant: {variant}") from exc


def known_variants() -> tuple[Variant, ...]:
    """Return the canonical variant names in a stable order."""
    return tuple(_VARIANTS)


def resolve_repo_id(variant: Variant, override: str | None = None) -> str:
    """Resolve the Hugging Face repo id or local override for a variant."""
    if override:
        return override
    return get_config(variant).hf_repo_id


def resolve_hidden_size(variant: Variant) -> int:
    """Return the encoder hidden size for a variant."""
    return get_config(variant).hidden_size


def spatial_tokens_per_temporal_bin(variant: Variant) -> int:
    """Return the number of spatial tokens emitted per temporal bin."""
    config = get_config(variant)
    return (config.crop_size // config.patch_size) * (config.crop_size // config.patch_size)
