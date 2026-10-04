import dataclasses
from typing import TYPE_CHECKING, Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
import openpi.models.vjepa2 as _vjepa2
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"
    world_model_expert_variant: _gemma.Variant = "gemma_300m"
    # Attention backend for the PyTorch Gemma/PaliGemma path. `sdpa` keeps the
    # custom block masks compatible while still allowing PyTorch to pick a fused kernel.
    attn_implementation: Literal["eager", "sdpa"] = "sdpa"
    # World model loss dropout alpha. 0 is no dropout, 1 is full dropout.
    wm_loss_dropout_alpha: float = 0.0
    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore
    # When False, the world model components (V-JEPA2 encoder, world model expert, etc.) are not initialized.
    enable_world_model: bool = False
    # V-JEPA2 encoder selection.
    # - `vjepa2_variant` picks one of the canonical checkpoints registered in
    #   `openpi.models.vjepa2`.
    # - `vjepa2_model_name_override`, when non-empty, takes precedence and is
    #   passed directly to `AutoModel.from_pretrained` (HF id or local path).
    vjepa2_variant: _vjepa2.Variant = "vitl-256"
    vjepa2_model_name_override: str | None = None
    # Optional input-side projector that maps V-JEPA2 history tokens to the
    # world-model expert width when the encoder hidden size does not match.
    vjepa2_enable_input_projector: bool = False
    # Optional MLP hidden width for the input projector. None -> LayerNorm + Linear.
    vjepa2_input_projector_hidden_dim: int | None = None
    # Learnable future-query table. vitl-256 uses 256 tokens per tubelet, so six
    # future frames need 768 slots. Checkpoints with a shorter table are copied
    # into the leading rows when training starts.
    wm_slot_max_len: int = 512
    # Optional inference-time future slot count for the world-model branch. When unset,
    # the PyTorch model infers the count from observation metadata if available.
    wm_inference_num_future_frames: int | None = None

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if not 0.0 <= self.wm_loss_dropout_alpha <= 1.0:
            raise ValueError(
                f"wm_loss_dropout_alpha must be in [0, 1], got {self.wm_loss_dropout_alpha}"
            )
        if self.attn_implementation not in {"eager", "sdpa"}:
            raise ValueError(
                "attn_implementation must be one of {'eager', 'sdpa'}, "
                f"got {self.attn_implementation!r}"
            )
        if self.wm_inference_num_future_frames is not None and self.wm_inference_num_future_frames < 0:
            raise ValueError(
                "wm_inference_num_future_frames must be >= 0, "
                f"got {self.wm_inference_num_future_frames}"
            )

        try:
            _vjepa2.get_config(self.vjepa2_variant)
        except ValueError as exc:
            raise ValueError(
                f"Unknown vjepa2_variant {self.vjepa2_variant!r}. "
                f"Known variants: {list(_vjepa2.known_variants())}."
            ) from exc

        if not isinstance(self.vjepa2_enable_input_projector, bool):
            raise ValueError(
                "vjepa2_enable_input_projector must be a bool, "
                f"got {type(self.vjepa2_enable_input_projector).__name__}"
            )
        if self.vjepa2_input_projector_hidden_dim is not None:
            if (
                not isinstance(self.vjepa2_input_projector_hidden_dim, int)
                or self.vjepa2_input_projector_hidden_dim <= 0
            ):
                raise ValueError(
                    "vjepa2_input_projector_hidden_dim must be None or a positive int, "
                    f"got {self.vjepa2_input_projector_hidden_dim!r}"
                )
        if self.vjepa2_enable_input_projector and not self.enable_world_model:
            raise ValueError(
                "vjepa2_enable_input_projector=True requires enable_world_model=True; "
                "the projector only exists inside the world-model branch."
            )

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
