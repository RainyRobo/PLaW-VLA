import logging
import math
import os
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812
import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing
from openpi.models_pytorch.world_model_pytorch import (
    VJepa2Adapter,
    WorldModelConfig,
    WorldModelEmbeddings,
    WorldModelFutureSlotBuilder,
    WorldModelPredictorHead,
)

def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def shared_bernoulli(alpha: float, device: torch.device) -> bool:
    """Draw one Bernoulli outcome and use it on every DDP rank.

    Independent draws put the expensive world-model branch and the cheap
    action-only branch on different ranks in the same step. The step then
    waits for the slowest rank, so dropout never reduces wall-clock time.
    """
    if alpha <= 0.0:
        return False
    if alpha >= 1.0:
        return True

    decision = torch.zeros((), dtype=torch.int32, device=device)
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    if not distributed or torch.distributed.get_rank() == 0:
        decision.fill_(int(torch.rand((), device=device) < alpha))
    if distributed:
        torch.distributed.broadcast(decision, src=0)
    return bool(int(decision.item()))


def make_att_2d_masks(pad_masks, att_masks, read_masks=None):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)
    if read_masks is None:
        read_masks = pad_masks
    elif read_masks.ndim != 2:
        raise ValueError(read_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    query_masks = pad_masks[:, :, None]
    key_masks = read_masks[:, None, :]
    return att_2d_masks & query_masks & key_masks


def _compute_wm_loss_per_batch(
    wm_pred: Tensor,
    wm_target: Tensor,
    wm_target_mask: Tensor,
    future_token_loss_normalizer: int | Tensor,
) -> Tensor:
    wm_valid_mask = wm_target_mask.to(dtype=wm_pred.dtype)
    wm_token_mse = F.mse_loss(wm_pred, wm_target.to(dtype=wm_pred.dtype), reduction="none").mean(dim=-1)
    normalizer = torch.as_tensor(
        future_token_loss_normalizer,
        device=wm_pred.device,
        dtype=wm_pred.dtype,
    ).clamp_min(1.0)
    return (wm_token_mse * wm_valid_mask).sum(dim=1) / normalizer


def _compute_last_step_collapse_metrics(
    wm_pred: Tensor,
    wm_target: Tensor,
    wm_target_mask: Tensor,
    *,
    tokens_per_temporal_bin: int,
) -> dict[str, Tensor]:
    if wm_pred.ndim != 3:
        raise ValueError(f"Expected wm_pred to be 3D, got shape {tuple(wm_pred.shape)}")
    if wm_target.shape != wm_pred.shape:
        raise ValueError(f"Expected wm_target shape {tuple(wm_pred.shape)}, got {tuple(wm_target.shape)}")
    if wm_target_mask.shape != wm_pred.shape[:2]:
        raise ValueError(
            f"Expected wm_target_mask shape {tuple(wm_pred.shape[:2])}, got {tuple(wm_target_mask.shape)}"
        )
    if tokens_per_temporal_bin <= 0:
        raise ValueError(f"tokens_per_temporal_bin must be > 0, got {tokens_per_temporal_bin}")

    zero = wm_pred.new_zeros(())
    metrics = {
        "wm_last_step_abs_pred_var": zero,
        "wm_last_step_abs_target_var": zero,
        "wm_last_step_abs_var_ratio": zero,
        "wm_last_step_abs_collapse_score": zero,
        "wm_last_step_eligible_count": zero,
        "wm_last_step_eligible_frac": zero,
    }
    if wm_pred.shape[1] == 0:
        return metrics
    if wm_pred.shape[1] % tokens_per_temporal_bin != 0:
        raise ValueError(
            f"Future token count {wm_pred.shape[1]} must be divisible by {tokens_per_temporal_bin}."
        )

    eligible_count = wm_target_mask[:, -tokens_per_temporal_bin:].all(dim=1).sum().to(dtype=torch.float32)
    eligible_frac = eligible_count / max(1, wm_pred.shape[0])
    metrics["wm_last_step_eligible_count"] = eligible_count
    metrics["wm_last_step_eligible_frac"] = eligible_frac

    abs_pred = wm_pred.detach().to(dtype=torch.float32)
    abs_target = wm_target.detach().to(dtype=torch.float32, device=wm_pred.device)

    last_step_pred = abs_pred[:, -tokens_per_temporal_bin:, :]
    last_step_target = abs_target[:, -tokens_per_temporal_bin:, :]
    last_step_mask = wm_target_mask[:, -tokens_per_temporal_bin:]
    eligible = last_step_mask.all(dim=1)
    if int(eligible.sum().item()) < 2:
        return metrics

    pred_flat = last_step_pred.reshape(last_step_pred.shape[0], -1)[eligible]
    target_flat = last_step_target.reshape(last_step_target.shape[0], -1)[eligible]

    pred_var = pred_flat.var(dim=0, unbiased=False).mean()
    target_var = target_flat.var(dim=0, unbiased=False).mean()
    model_mse = F.mse_loss(pred_flat, target_flat)
    const_mse = F.mse_loss(target_flat.mean(dim=0, keepdim=True).expand_as(target_flat), target_flat)

    eps = torch.finfo(torch.float32).eps
    var_ratio = pred_var / target_var.clamp_min(eps) if float(target_var.item()) > eps else zero
    collapse_score = 1.0 - model_mse / const_mse.clamp_min(eps) if float(const_mse.item()) > eps else zero

    metrics.update(
        {
            "wm_last_step_abs_pred_var": pred_var.to(device=wm_pred.device),
            "wm_last_step_abs_target_var": target_var.to(device=wm_pred.device),
            "wm_last_step_abs_var_ratio": var_ratio.to(device=wm_pred.device),
            "wm_last_step_abs_collapse_score": torch.as_tensor(
                collapse_score,
                dtype=torch.float32,
                device=wm_pred.device,
            ),
        }
    )
    return metrics


def _zero_last_step_collapse_metrics(device: torch.device) -> dict[str, Tensor]:
    zero = torch.zeros((), dtype=torch.float32, device=device)
    return {
        "wm_last_step_abs_pred_var": zero,
        "wm_last_step_abs_target_var": zero,
        "wm_last_step_abs_var_ratio": zero,
        "wm_last_step_abs_collapse_score": zero,
        "wm_last_step_eligible_count": zero,
        "wm_last_step_eligible_frac": zero,
    }


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05
        self.training_stage = config.training_stage
        self.enable_world_model = getattr(config, "enable_world_model", False)
        self.attn_implementation = getattr(config, "attn_implementation", "sdpa")
        
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        if self.enable_world_model:
            world_model_expert_config = _gemma.get_config(config.world_model_expert_variant)
            self.world_model_config = WorldModelConfig(
                expected_embedding_dim=world_model_expert_config.width,
                device=config.device,
                variant=config.vjepa2_variant,
                model_name_override=getattr(config, "vjepa2_model_name_override", None),
                enable_input_projector=getattr(config, "vjepa2_enable_input_projector", False),
                input_projector_hidden_dim=getattr(config, "vjepa2_input_projector_hidden_dim", None),
            )
            self.world_model_adapter: VJepa2Adapter | None = self.world_model_config.build_adapter()
        else:
            world_model_expert_config = None
            self.world_model_config = None
            self.world_model_adapter = None

        if self.enable_world_model:
            use_adarms = [False, False, True] if self.pi05 else [False, False, False]
        else:
            use_adarms = [False, True] if self.pi05 else [False, False]

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            world_model_expert_config=world_model_expert_config,
            use_adarms=use_adarms,
            precision=config.dtype,
            attn_implementation=self.attn_implementation,
        )

        self.wm_loss_weight = getattr(config, "wm_loss_weight", 0.1)
        self.wm_slot_max_len = getattr(config, "wm_slot_max_len", 512)
        self.wm_loss_dropout_alpha = getattr(config, "wm_loss_dropout_alpha", 0.0)
        self.wm_inference_num_future_frames = getattr(config, "wm_inference_num_future_frames", None)

        if self.enable_world_model:
            self.world_future_builder = WorldModelFutureSlotBuilder(
                adapter=self.world_model_adapter,
                embed_dim=world_model_expert_config.width,
                slot_max_len=self.wm_slot_max_len,
            )
            self.world_pred_head = WorldModelPredictorHead(
                world_model_expert_config.width,
                self.world_model_adapter.embedding_dim,
            )
        else:
            self.world_future_builder = None
            self.world_pred_head = None

        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        # Cast the trainable world-model modules to the configured precision.
        # The PaliGemma backbone manages its own precision internally
        # (``to_bfloat16_for_selected_params``).  The frozen V-JEPA2
        # encoder MUST stay in fp32: it was pretrained in fp32 and produces
        # all-zero token embeddings when its parameters are converted to bf16,
        # which collapses the world-model loss to zero.
        #
        # Note: ``world_future_builder`` holds a reference to
        # ``world_model_adapter`` as a registered submodule, so calling
        # ``.to(bf16)`` on the builder would recursively cast the shared
        # encoder.  We therefore cast everything first and then restore the
        # encoder to fp32 explicitly.
        #
        # The action projections / time MLPs are deliberately left in their
        # default fp32: the embedding paths around them (``embed_suffix``,
        # ``action_out_proj``) feed them fp32 activations by design, with
        # explicit dtype casts at the expert boundary (see
        # ``_forward_action_and_wm``).
        if config.dtype == "bfloat16":
            target_dtype = torch.bfloat16
        elif config.dtype == "float32":
            target_dtype = torch.float32
        else:
            raise ValueError(f"Unsupported dtype: {config.dtype!r}")
        for module in (
            self.world_model_adapter,
            self.world_future_builder,
            self.world_pred_head,
        ):
            if module is not None:
                module.to(dtype=target_dtype)

        # V-JEPA2 was pretrained in fp32 and produces all-zero embeddings when its
        # parameters are cast to bf16.
        if self.world_model_adapter is not None and target_dtype == torch.bfloat16:
            self.world_model_adapter.encoder_module.to(dtype=torch.float32)

        torch.set_float32_matmul_precision("high")
        compile_mode = os.environ.get("PLAW_VLA_TORCH_COMPILE_MODE", "max-autotune")
        logging.info("Compiling PI0Pytorch.sample_actions with torch.compile(mode=%s)", compile_mode)
        self.sample_actions = torch.compile(self.sample_actions, mode=compile_mode)

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False
        self._gc_enabled_modules: frozenset = frozenset()
        self._world_model_info_logged = False
        
        # Training stage setup
        self.set_training_stage(self.training_stage)
        self.print_trainable_parameters_auto()

        msg = (
            "transformers_replace is not installed correctly. "
            "Install `transformers==5.0.0`, then copy "
            "`./src/openpi/models_pytorch/transformers_replace/*` into the installed "
            "`transformers/` package directory."
        )
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self, modules: list[str] | None = None):
        """Enable gradient checkpointing for memory optimization.

        Args:
            modules: List of module names to enable GC on. None means all.
                Valid names: "language_model", "vision_tower", "action_expert", "wm_expert".
        """
        if modules is None:
            modules = ["language_model", "vision_tower", "action_expert", "wm_expert"]
        self._gc_enabled_modules = frozenset(modules)

        # Language model is the single largest activation-memory consumer.
        if "language_model" in modules:
            self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        # Vision tower — second largest.
        if "vision_tower" in modules:
            self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        # World-model expert (if present).
        if "wm_expert" in modules and hasattr(self.paligemma_with_expert, "gemma_world_model_expert"):
            self.paligemma_with_expert.gemma_world_model_expert.model.gradient_checkpointing = True
        # Action expert.
        if "action_expert" in modules:
            self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        # Enable the fine-grained checkpointing flag so _apply_checkpoint wraps
        # any remaining heavyweight ops that aren't covered by per-module GC.
        self.gradient_checkpointing_enabled = True

        logging.info(
            "Enabled gradient checkpointing for modules: %s (total: %d)",
            sorted(modules),
            len(modules),
        )

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing entirely."""
        self.gradient_checkpointing_enabled = False
        self._gc_enabled_modules = frozenset()
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        if hasattr(self.paligemma_with_expert, "gemma_world_model_expert"):
            self.paligemma_with_expert.gemma_world_model_expert.model.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks, *, dtype: torch.dtype):
        """Prepare a 4D additive attention mask in the same dtype as the attention query."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        zero = torch.zeros((), dtype=dtype, device=att_2d_masks.device)
        neg_inf = torch.full((), torch.finfo(dtype).min, dtype=dtype, device=att_2d_masks.device)
        return torch.where(att_2d_masks_4d, zero, neg_inf)

    def _prefix_attention_dtype(self) -> torch.dtype:
        return self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype

    def _world_attention_dtype(self) -> torch.dtype:
        if not self.enable_world_model:
            return self._prefix_attention_dtype()
        return self.paligemma_with_expert.gemma_world_model_expert.model.layers[0].self_attn.q_proj.weight.dtype

    def _suffix_attention_dtype(self) -> torch.dtype:
        return self.paligemma_with_expert.gemma_expert.model.layers[0].self_attn.q_proj.weight.dtype

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []
        image_graph_anchor = None

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = image_embed_func(img)
            graph_term = img_emb.sum()
            image_graph_anchor = graph_term if image_graph_anchor is None else image_graph_anchor + graph_term

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = lang_embed_func(lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        if image_graph_anchor is not None:
            # Some ranks can receive batches where every image token is masked out.
            # Keep the vision tower on the autograd path with a zero-valued anchor
            # so DDP static-graph training still sees a stable parameter set.
            embs = embs + image_graph_anchor.to(dtype=embs.dtype) * 0.0
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = state_proj_func(state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = action_proj_func(noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = mlp_func(action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = time_mlp_func(time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond
    
    def _should_drop_wm_branch(self, device: torch.device) -> bool:
        if not self.training:
            return False
        if self.training_stage != "post_training":
            return False
   
        return shared_bernoulli(float(self.wm_loss_dropout_alpha), device)

    @staticmethod
    def _build_block_att_mask(batch_size: int, block_lengths: list[int], *, device: torch.device) -> Tensor:
        total_len = sum(block_lengths)
        att_mask = torch.zeros((batch_size, total_len), dtype=torch.bool, device=device)
        offset = 0
        for block_len in block_lengths:
            if block_len > 0:
                att_mask[:, offset] = True
                offset += block_len
        return att_mask

    def _run_prefix_world(
        self,
        prefix_embs: Tensor,
        prefix_pad_masks: Tensor,
        prefix_att_masks: Tensor,
        wm_embs: Tensor,
        wm_pad_masks: Tensor,
        wm_read_masks: Tensor,
        wm_att_masks: Tensor,
    ) -> Tensor:
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
            wm_embs = wm_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, wm_pad_masks], dim=1)
        read_masks = torch.cat([prefix_pad_masks, wm_read_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, wm_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks, read_masks=read_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks, dtype=wm_embs.dtype)

        def forward_func(prefix_embs, wm_embs, att_2d_masks_4d, position_ids):
            (_, wm_out, _), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, wm_embs, None],
                use_cache=False,
                adarms_cond=[None, None, None],
            )
            return wm_out

        return self._apply_checkpoint(
            forward_func,
            prefix_embs,
            wm_embs,
            att_2d_masks_4d,
            position_ids,
        )

    @staticmethod
    def _split_wm_batch(
        wm_batch,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        hist_embs = wm_batch.wm_inputs[:, : wm_batch.l_hist, :]
        hist_pad_masks = wm_batch.wm_pad_mask[:, : wm_batch.l_hist]
        hist_read_masks = wm_batch.wm_read_mask[:, : wm_batch.l_hist]
        slot_embs = wm_batch.wm_inputs[:, wm_batch.l_hist :, :]
        future_valid_mask = wm_batch.wm_target_mask
        return hist_embs, hist_pad_masks, hist_read_masks, slot_embs, future_valid_mask

    def _build_conditioned_wm_inputs(
        self,
        hist_embs: Tensor,
        hist_pad_masks: Tensor,
        hist_read_masks: Tensor,
        future_embs: Tensor,
        future_valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        future_valid_mask = future_valid_mask.to(dtype=torch.bool, device=hist_embs.device)
        wm_embs = torch.cat([hist_embs, future_embs.to(dtype=hist_embs.dtype)], dim=1)
        wm_pad_masks = torch.cat([hist_pad_masks, future_valid_mask], dim=1)
        wm_read_masks = torch.cat([hist_read_masks, future_valid_mask], dim=1)
        wm_att_masks = self._build_block_att_mask(
            hist_embs.shape[0],
            [hist_embs.shape[1], future_embs.shape[1]],
            device=hist_embs.device,
        )
        return wm_embs, wm_pad_masks, wm_read_masks, wm_att_masks

    def _align_wm_future_embeddings_for_expert(
        self, future_embs: Tensor, *, hist_embs: Tensor
    ) -> Tensor:
        """Project predicted future tokens to world-model expert width when needed.

        With ``vjepa2_enable_input_projector``, history is fed to the expert at
        ``world_model_expert`` width while ``world_pred_head`` outputs raw V-JEPA
        targets; the follow-up expert pass that concatenates history + predictions
        requires a single embedding width.
        """
        if future_embs.shape[-1] == hist_embs.shape[-1]:
            return future_embs
        adapter = self.world_model_adapter
        if adapter is None or not getattr(adapter, "enable_input_projector", False):
            raise RuntimeError(
                "World-model future embedding width "
                f"({future_embs.shape[-1]}) does not match history width ({hist_embs.shape[-1]}), "
                "but no V-JEPA input projector is configured to map between them."
            )
        return adapter.apply_input_projector(future_embs)

    def _parallel_slot_future_rollout(
        self,
        prefix_embs: Tensor,
        prefix_pad_masks: Tensor,
        prefix_att_masks: Tensor,
        wm_embs: Tensor,
        wm_pad_masks: Tensor,
        wm_read_masks: Tensor,
        wm_att_masks: Tensor,
        *,
        history_len: int,
    ) -> tuple[Tensor, Tensor]:
        if self.world_pred_head is None:
            raise ValueError("Slot-parallel future rollout requires world model components.")
        if history_len < 0 or history_len > wm_embs.shape[1]:
            raise ValueError(
                f"history_len must be within [0, {wm_embs.shape[1]}], got {history_len}."
            )

        batch_size, total_len, embed_dim = wm_embs.shape
        future_len = total_len - history_len
        if future_len == 0:
            empty = wm_embs.new_zeros((batch_size, 0, embed_dim), dtype=torch.float32)
            return empty, empty

        wm_out = self._run_prefix_world(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            wm_embs,
            wm_pad_masks,
            wm_read_masks,
            wm_att_masks,
        ).to(dtype=torch.float32)
        wm_future_out = wm_out[:, history_len:, :]
        wm_pred = self.world_pred_head(wm_future_out)
        return wm_pred, wm_pred

    def _build_inference_world_inputs(
        self,
        prefix_embs: Tensor,
        prefix_pad_masks: Tensor,
        prefix_att_masks: Tensor,
        wm_embeddings: WorldModelEmbeddings,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        wm_read_masks = wm_embeddings.read_mask if wm_embeddings.read_mask is not None else wm_embeddings.pad_mask
        hist_slice = wm_embeddings.embeddings[:, : wm_embeddings.history_token_len, :]
        hist_pad = wm_embeddings.pad_mask[:, : wm_embeddings.history_token_len]
        hist_read = wm_read_masks[:, : wm_embeddings.history_token_len]

        _, conditioned_future = self._parallel_slot_future_rollout(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            wm_embeddings.embeddings,
            wm_embeddings.pad_mask,
            wm_read_masks,
            wm_embeddings.att_mask,
            history_len=wm_embeddings.history_token_len,
        )
        conditioned_future = self._align_wm_future_embeddings_for_expert(
            conditioned_future, hist_embs=hist_slice
        )
        return self._build_conditioned_wm_inputs(
            hist_slice,
            hist_pad,
            hist_read,
            conditioned_future,
            wm_read_masks[:, wm_embeddings.history_token_len :],
        )

    def _compute_wm_last_step_collapse_metrics(
        self,
        wm_pred: Tensor,
        wm_target: Tensor,
        wm_target_mask: Tensor,
    ) -> dict[str, Tensor]:
        if self.world_model_adapter is None:
            return _zero_last_step_collapse_metrics(wm_pred.device)
        return _compute_last_step_collapse_metrics(
            wm_pred.detach(),
            wm_target.detach(),
            wm_target_mask,
            tokens_per_temporal_bin=self.world_model_adapter.spatial_tokens_per_temporal_bin,
        )

    def _forward_world_only(self, prefix_embs, prefix_pad_masks, prefix_att_masks, wm_batch):
        wm_target = wm_batch.wm_target.to(dtype=torch.float32)
        wm_pred, _ = self._parallel_slot_future_rollout(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            wm_batch.wm_inputs,
            wm_batch.wm_pad_mask,
            wm_batch.wm_read_mask,
            wm_batch.wm_att_mask,
            history_len=wm_batch.l_hist,
        )
        wm_loss_per_batch = _compute_wm_loss_per_batch(
            wm_pred,
            wm_target,
            wm_batch.wm_target_mask,
            wm_batch.future_token_loss_normalizer,
        )
        collapse_metrics = self._compute_wm_last_step_collapse_metrics(
            wm_pred,
            wm_target,
            wm_batch.wm_target_mask,
        )

        total_loss = wm_loss_per_batch[:, None, None]
        return total_loss, {
            "action_loss": torch.zeros((), device=total_loss.device),
            "wm_loss": wm_loss_per_batch.mean().detach(),
            "wm_branch_dropped": torch.zeros((), device=total_loss.device),
            **collapse_metrics,
        }
    
    def _forward_action_only(self, prefix_embs, prefix_pad_masks, prefix_att_masks, state, x_t, time, u_t):

        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)

        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
        
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks, dtype=suffix_embs.dtype)
        
        def forward_action_only(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, _, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None, suffix_embs],
                use_cache=False,
                adarms_cond=[None, None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_action_only, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = action_out_proj_func(suffix_out)
        action_loss = F.mse_loss(u_t, v_t, reduction="none")

        return action_loss, {
            "action_loss": action_loss.mean().detach(),
            "wm_loss": torch.zeros((), device=action_loss.device),
            "wm_branch_dropped": torch.ones((), device=action_loss.device),
            **_zero_last_step_collapse_metrics(action_loss.device),
        }

    def _forward_action_and_wm(
        self,
        prefix_embs,
        prefix_pad_masks,
        prefix_att_masks,
        wm_batch,
        state,
        x_t,
        time,
        u_t,
    ):
        hist_embs, hist_pad_masks, hist_read_masks, _, future_valid_mask = self._split_wm_batch(wm_batch)
        wm_pred, conditioned_future = self._parallel_slot_future_rollout(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            wm_batch.wm_inputs,
            wm_batch.wm_pad_mask,
            wm_batch.wm_read_mask,
            wm_batch.wm_att_mask,
            history_len=wm_batch.l_hist,
        )
        conditioned_future_for_expert = self._align_wm_future_embeddings_for_expert(
            conditioned_future, hist_embs=hist_embs
        )
        wm_embs, wm_pad_masks, wm_read_masks, wm_att_masks = self._build_conditioned_wm_inputs(
            hist_embs,
            hist_pad_masks,
            hist_read_masks,
            conditioned_future_for_expert,
            future_valid_mask,
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)

        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
            wm_embs = wm_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, wm_pad_masks, suffix_pad_masks], dim=1)
        read_masks = torch.cat([prefix_pad_masks, wm_read_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, wm_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks, read_masks=read_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks, dtype=suffix_embs.dtype)

        def forward_func(prefix_embs, wm_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, _, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, wm_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, wm_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = action_out_proj_func(suffix_out)
        action_loss = F.mse_loss(u_t, v_t, reduction="none")

        wm_target = wm_batch.wm_target.to(dtype=torch.float32)
        wm_loss_per_batch = _compute_wm_loss_per_batch(
            wm_pred,
            wm_target,
            wm_batch.wm_target_mask,
            wm_batch.future_token_loss_normalizer,
        )
        collapse_metrics = self._compute_wm_last_step_collapse_metrics(
            wm_pred,
            wm_target,
            wm_batch.wm_target_mask,
        )

        total_loss = action_loss + self.wm_loss_weight * wm_loss_per_batch[:, None, None]

        return total_loss, {
            "action_loss": action_loss.mean().detach(),
            "wm_loss": wm_loss_per_batch.mean().detach(),
            "wm_branch_dropped": torch.zeros((), device=action_loss.device),
            **collapse_metrics,
        }

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)

        # branch 1 : world model alignment stage
        if self.training_stage == "wm_alignment":
            wm_batch = self.world_future_builder.build_training_inputs(observation)
            return self._forward_world_only(
                prefix_embs,
                prefix_pad_masks,
                prefix_att_masks,
                wm_batch,
            )
        
        # branch 2 : post-training stage
        if actions is None:
            raise ValueError("actions must not be None when training_stage != 'wm_alignment'")

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        if not self.enable_world_model or self.world_future_builder is None:
            return self._forward_action_only(prefix_embs, prefix_pad_masks, prefix_att_masks, state, x_t, time, u_t)

        # branch 2.1 : post-training stage dropout world model branch
        if self._should_drop_wm_branch(actions.device):
            return self._forward_action_only(prefix_embs, prefix_pad_masks, prefix_att_masks, state, x_t, time, u_t)

        # branch 2.2 : post-training stage with world model branch
        wm_batch = self.world_future_builder.build_training_inputs(observation)
        return self._forward_action_and_wm(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            wm_batch,
            state,
            x_t,
            time,
            u_t,
        )
    

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        # 1. Build VLM prefix cache once.
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_attn_dtype = self._prefix_attention_dtype()
        if prefix_embs.dtype != prefix_attn_dtype:
            prefix_embs = prefix_embs.to(dtype=prefix_attn_dtype)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks, dtype=prefix_attn_dtype)
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None] if not self.enable_world_model else [prefix_embs, None, None],
            use_cache=True,
        )

        prefix_memory_pad_masks = prefix_pad_masks
        prefix_memory_read_masks = prefix_pad_masks

        # 2. Build and cache the world-model context for the action expert.
        if self.enable_world_model:
            wm_embeddings = self.world_future_builder.build_inference_inputs(
                observation,
                future_num_frames=self.wm_inference_num_future_frames,
            )
            wm_embs, wm_pad_masks, wm_read_masks, wm_att_masks = self._build_inference_world_inputs(
                prefix_embs,
                prefix_pad_masks,
                prefix_att_masks,
                wm_embeddings,
            )
            world_attn_dtype = self._world_attention_dtype()
            if wm_embs.dtype != world_attn_dtype:
                wm_embs = wm_embs.to(dtype=world_attn_dtype)
            wm_att_2d_masks = make_att_2d_masks(wm_pad_masks, wm_att_masks, read_masks=wm_read_masks)

            prefix_pad_2d_masks = prefix_memory_pad_masks[:, None, :].expand(
                bsize,
                wm_pad_masks.shape[1],
                prefix_memory_pad_masks.shape[1],
            )
            wm_full_att_2d_masks = torch.cat([prefix_pad_2d_masks, wm_att_2d_masks], dim=2)

            prefix_offsets = torch.sum(prefix_memory_pad_masks, dim=-1)[:, None]
            wm_position_ids = prefix_offsets + torch.cumsum(wm_pad_masks, dim=1) - 1
            wm_full_att_2d_masks_4d = self._prepare_attention_masks_4d(wm_full_att_2d_masks, dtype=world_attn_dtype)

            _, past_key_values = self.paligemma_with_expert.forward(
                attention_mask=wm_full_att_2d_masks_4d,
                position_ids=wm_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, wm_embs, None],
                use_cache=True,
            )

            prefix_memory_pad_masks = torch.cat([prefix_memory_pad_masks, wm_pad_masks], dim=1)
            prefix_memory_read_masks = torch.cat([prefix_memory_read_masks, wm_read_masks], dim=1)

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_memory_pad_masks,
                prefix_memory_read_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        prefix_read_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_read_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        suffix_attn_dtype = self._suffix_attention_dtype()
        if suffix_embs.dtype != suffix_attn_dtype:
            suffix_embs = suffix_embs.to(dtype=suffix_attn_dtype)
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks, dtype=suffix_attn_dtype)
        if self.enable_world_model:
            outputs_embeds, _ = self.paligemma_with_expert.forward(
                attention_mask=full_att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, None, suffix_embs],
                use_cache=False,
                adarms_cond=[None, None, adarms_cond],
            )
            suffix_out = outputs_embeds[2]
        else:
            outputs_embeds, _ = self.paligemma_with_expert.forward(
                attention_mask=full_att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
    
    def set_training_stage(self, stage: str = "post_training") -> None:
        trainable_modules = []
        frozen_modules = []

        if stage == "wm_alignment":
            if not self.enable_world_model:
                raise ValueError("Cannot use 'wm_alignment' stage when world model is disabled")
            trainable_modules = [
                self.paligemma_with_expert.gemma_world_model_expert.model,
                self.world_pred_head,
                self.world_future_builder,
            ]
            frozen_modules = [
                # We always consume hidden states from `.model.forward()` and never
                # use the causal LM head during policy/world-model training.
                # Leaving `lm_head` trainable makes DDP see a permanently-unused
                # parameter (`index 163` on gemma_300m), which breaks
                # `find_unused_parameters=False` and prevents `static_graph=True`.
                self.paligemma_with_expert.gemma_world_model_expert.lm_head,
                self.world_model_adapter.encoder_module,
                self.paligemma_with_expert.paligemma,
                self.paligemma_with_expert.gemma_expert,
                self.action_in_proj,
                self.action_out_proj,
            ]
            if self.pi05:
                frozen_modules.extend([self.time_mlp_in, self.time_mlp_out])
                
        elif stage == "post_training":
            # The causal LM heads are never used: training reads hidden states
            # from `.model.forward()`. Leaving them trainable makes DDP treat
            # them as permanently unused parameters.
            frozen_modules = [self.paligemma_with_expert.gemma_expert.lm_head]
            if self.enable_world_model:
                frozen_modules.extend(
                    [
                        self.world_model_adapter.encoder_module,
                        self.paligemma_with_expert.gemma_world_model_expert.lm_head,
                    ]
                )

        for model in trainable_modules:
            for p in model.parameters():
                p.requires_grad = True
                
        for model in frozen_modules:
            for p in model.parameters():
                p.requires_grad = False
    
    def print_trainable_parameters_auto(self, top_k: int = 40) -> None:
        """Automatically inspect all registered parameters and report trainable status."""
        total_params = 0
        trainable_params = 0
        frozen_params = 0
        top_level_stats = {}
        role_stats = {}
        trainable_named_params = []
        wm_encoder_total = 0
        wm_encoder_trainable = 0

        role_prefixes = [
            ("VLM_EXPERT", "paligemma_with_expert.paligemma."),
            ("WORLD_EXPERT", "paligemma_with_expert.gemma_world_model_expert."),
            ("ACTION_EXPERT", "paligemma_with_expert.gemma_expert."),
            ("WM_ENCODER", "world_model_adapter."),
        ]

        def _add_role_param(role_name: str, param_numel: int, is_trainable: bool) -> None:
            if role_name not in role_stats:
                role_stats[role_name] = {"trainable": 0, "frozen": 0, "total": 0}
            role_stats[role_name]["total"] += param_numel
            if is_trainable:
                role_stats[role_name]["trainable"] += param_numel
            else:
                role_stats[role_name]["frozen"] += param_numel

        for name, param in self.named_parameters():
            numel = param.numel()
            total_params += numel

            top_level = name.split(".", 1)[0]
            if top_level not in top_level_stats:
                top_level_stats[top_level] = {"trainable": 0, "frozen": 0, "total": 0}
            top_level_stats[top_level]["total"] += numel

            if param.requires_grad:
                trainable_params += numel
                top_level_stats[top_level]["trainable"] += numel
                trainable_named_params.append((numel, name, tuple(param.shape), str(param.dtype)))
            else:
                frozen_params += numel
                top_level_stats[top_level]["frozen"] += numel

            matched_role = "OTHER"
            for role_name, prefix in role_prefixes:
                if name.startswith(prefix):
                    matched_role = role_name
                    break
            _add_role_param(matched_role, numel, param.requires_grad)

            if matched_role == "WM_ENCODER":
                wm_encoder_total += numel
                if param.requires_grad:
                    wm_encoder_trainable += numel

        overall_pct = 100.0 * trainable_params / total_params if total_params > 0 else 0.0
        line = "=" * 118
        subline = "-" * 118
        logging.info(line)
        logging.info("AUTO PARAMETER SUMMARY (FULL MODEL SCAN)")
        logging.info(line)
        logging.info(
            f"{'Total':>14s}  {'Trainable':>14s}  {'Frozen':>14s}  {'Trainable%':>10s}"
        )
        logging.info(
            f"{total_params:>14,}  {trainable_params:>14,}  {frozen_params:>14,}  {overall_pct:>9.2f}%"
        )
        logging.info(subline)
        logging.info("ROLE BREAKDOWN (ARCHITECTURE VIEW)")
        logging.info(
            f"{'Status':8s}  {'Role':32s}  {'Trainable':>14s}  {'Frozen':>14s}  {'Total':>14s}  {'Trainable%':>10s}"
        )
        logging.info(subline)
        for role_name, stats in sorted(role_stats.items(), key=lambda kv: kv[1]["total"], reverse=True):
            role_trainable_pct = 100.0 * stats["trainable"] / stats["total"] if stats["total"] > 0 else 0.0
            status = "ACTIVE" if stats["trainable"] > 0 else "FROZEN"
            logging.info(
                f"{status:8s}  {role_name:32.32s}  {stats['trainable']:>14,}  "
                f"{stats['frozen']:>14,}  {stats['total']:>14,}  {role_trainable_pct:>9.2f}%"
            )
        logging.info(subline)
        logging.info("MODULE BREAKDOWN")
        logging.info(
            f"{'Status':8s}  {'Module':32s}  {'Trainable':>14s}  {'Frozen':>14s}  {'Total':>14s}  {'Trainable%':>10s}"
        )
        logging.info(subline)

        for module_name, stats in sorted(top_level_stats.items(), key=lambda kv: kv[1]["total"], reverse=True):
            module_trainable_pct = 100.0 * stats["trainable"] / stats["total"] if stats["total"] > 0 else 0.0
            status = "ACTIVE" if stats["trainable"] > 0 else "FROZEN"
            logging.info(
                f"{status:8s}  {module_name:32.32s}  {stats['trainable']:>14,}  "
                f"{stats['frozen']:>14,}  {stats['total']:>14,}  {module_trainable_pct:>9.2f}%"
            )

        if trainable_params == 0:
            logging.warning("Reminder: no trainable parameters detected in the whole model.")

        logging.info(subline)
        logging.info("WORLD ENCODER CHECK (world_model_adapter)")
        if self.world_model_adapter is not None:
            if wm_encoder_total == 0:
                logging.warning(
                    "Reminder: world_model_adapter exists but no registered parameters were found under "
                    "'world_model_adapter.*'."
                )
            elif wm_encoder_trainable == 0:
                logging.warning(
                    "Reminder: world_model_adapter parameters are all frozen "
                    f"(0/{wm_encoder_total:,} trainable under world_model_adapter)."
                )
            else:
                logging.info(
                    "World encoder trainable parameters: "
                    f"{wm_encoder_trainable:,}/{wm_encoder_total:,} "
                    f"({100.0 * wm_encoder_trainable / wm_encoder_total:.2f}%)."
                )
        else:
            logging.info("world_model_adapter is None")

        if trainable_named_params:
            logging.info(subline)
            logging.info("TOP TRAINABLE PARAMETERS BY SIZE")
            logging.info(
                f"{'Params':>14s}  {'DType':10s}  {'Shape':22s}  Name"
            )
            logging.info(subline)
            for numel, name, shape, dtype in sorted(trainable_named_params, key=lambda x: x[0], reverse=True)[:top_k]:
                logging.info(f"{numel:>14,}  {dtype:10s}  {str(shape):22.22s}  {name}")

        logging.info(line)
