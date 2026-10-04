from typing import Any
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
from transformers import GemmaForCausalLM
from transformers import PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import GemmaConfig, modeling_gemma


def _default_rope_init(config, device=None, seq_len=None, layer_type=None):
    del seq_len
    if layer_type is not None and hasattr(config, "rope_parameters") and config.rope_parameters is not None:
        rope_params = config.rope_parameters.get(layer_type, {})
    else:
        rope_params = getattr(config, "rope_parameters", None) or {}

    base = rope_params.get("rope_theta", getattr(config, "rope_theta", 10000.0))
    partial_rotary_factor = rope_params.get("partial_rotary_factor", getattr(config, "partial_rotary_factor", 1.0))
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim))
    return inv_freq, 1.0


if "default" not in modeling_gemma.ROPE_INIT_FUNCTIONS:
    modeling_gemma.ROPE_INIT_FUNCTIONS["default"] = _default_rope_init

if isinstance(modeling_gemma.GemmaForCausalLM._tied_weights_keys, list):
    modeling_gemma.GemmaForCausalLM._tied_weights_keys = {
        "lm_head.weight": "model.embed_tokens.weight",
    }

if isinstance(PaliGemmaForConditionalGeneration._tied_weights_keys, list):
    PaliGemmaForConditionalGeneration._tied_weights_keys = {
        "lm_head.weight": "model.language_model.embed_tokens.weight",
    }


_SUPPORTED_ATTENTION_IMPLS = {"eager", "sdpa"}


def _resolve_attention_implementation(module: nn.Module) -> str:
    attn_impl = getattr(getattr(module, "config", None), "_attn_implementation", None) or "eager"
    if attn_impl not in _SUPPORTED_ATTENTION_IMPLS:
        raise ValueError(
            f"Unsupported attention implementation {attn_impl!r}. "
            f"Expected one of {sorted(_SUPPORTED_ATTENTION_IMPLS)}."
        )
    return attn_impl


def _multi_stream_attention_forward(
    module: nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    scaling: float,
):
    attn_impl = _resolve_attention_implementation(module)
    if attn_impl == "eager":
        return modeling_gemma.eager_attention_forward(
            module,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling,
        )

    key_states = modeling_gemma.repeat_kv(key_states, module.num_key_value_groups)
    value_states = modeling_gemma.repeat_kv(value_states, module.num_key_value_groups)
    causal_mask = attention_mask[:, :, :, : key_states.shape[-2]] if attention_mask is not None else None
    if causal_mask is not None and causal_mask.dtype != query_states.dtype:
        causal_mask = causal_mask.to(dtype=query_states.dtype)
    attn_output = F.scaled_dot_product_attention(
        query_states.contiguous(),
        key_states.contiguous(),
        value_states.contiguous(),
        attn_mask=causal_mask,
        dropout_p=0.0,
        scale=scaling,
        is_causal=False,
    )
    return attn_output.transpose(1, 2).contiguous(), None


class PaliGemmaWithExpertModel(nn.Module):
    def __init__(
        self,
        vlm_config,
        action_expert_config,
        world_model_expert_config=None,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        attn_implementation: Literal["eager", "sdpa"] = "sdpa",
    ):
        super().__init__()
        self.has_world_model_expert = world_model_expert_config is not None

        if use_adarms is None:
            use_adarms = [False, False, False] if self.has_world_model_expert else [False, False]

        vlm_text_config_hf = GemmaConfig(
            hidden_size=vlm_config.width,
            intermediate_size=vlm_config.mlp_dim,
            num_attention_heads=vlm_config.num_heads,
            head_dim=vlm_config.head_dim,
            num_hidden_layers=vlm_config.depth,
            num_key_value_heads=vlm_config.num_kv_heads,
            hidden_activation="gelu_pytorch_tanh",
            dtype="float32",
            vocab_size=257152,
            use_adarms=use_adarms[0],
            adarms_cond_dim=vlm_config.width if use_adarms[0] else None,
            use_bidirectional_attention=True,
        )
        vlm_config_hf = CONFIG_MAPPING["paligemma"](
            text_config=vlm_text_config_hf,
            pad_token_id=0,
            bos_token_id=2,
            eos_token_id=1,
        )
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.vision_config.intermediate_size = 4304
        # Projector output must match the language width. gemma_2b is 2048;
        # smaller variants (including the dummy debug model) are not.
        vlm_config_hf.vision_config.projection_dim = vlm_config.width
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.dtype = "float32"

        if self.has_world_model_expert:
            world_model_expert_config_hf = CONFIG_MAPPING["gemma"](
                head_dim=world_model_expert_config.head_dim,
                hidden_size=world_model_expert_config.width,
                intermediate_size=world_model_expert_config.mlp_dim,
                num_attention_heads=world_model_expert_config.num_heads,
                num_hidden_layers=world_model_expert_config.depth,
                num_key_value_heads=world_model_expert_config.num_kv_heads,
                vocab_size=257152,
                hidden_activation="gelu_pytorch_tanh",
                dtype="float32",
                use_adarms=use_adarms[1],
                adarms_cond_dim=world_model_expert_config.width if use_adarms[1] else None,
            )

        # Action expert adarms index depends on whether world model expert exists
        action_adarms_idx = 2 if self.has_world_model_expert else 1
        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            dtype="float32",
            use_adarms=use_adarms[action_adarms_idx],
            adarms_cond_dim=action_expert_config.width if use_adarms[action_adarms_idx] else None,
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)

        if self.has_world_model_expert:
            self.gemma_world_model_expert = GemmaForCausalLM(config=world_model_expert_config_hf)
            self.gemma_world_model_expert.model.embed_tokens = None

        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        self.set_attention_implementation(attn_implementation)
        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def set_attention_implementation(self, attn_implementation: Literal["eager", "sdpa"]) -> None:
        if attn_implementation not in _SUPPORTED_ATTENTION_IMPLS:
            raise ValueError(
                f"Unsupported attention implementation {attn_implementation!r}. "
                f"Expected one of {sorted(_SUPPORTED_ATTENTION_IMPLS)}."
            )

        self.attn_implementation = attn_implementation
        self.paligemma.config.text_config._attn_implementation = attn_implementation  # noqa: SLF001
        self.paligemma.language_model.config._attn_implementation = attn_implementation  # noqa: SLF001

        if self.has_world_model_expert:
            self.gemma_world_model_expert.config._attn_implementation = attn_implementation  # noqa: SLF001
            self.gemma_world_model_expert.model.config._attn_implementation = attn_implementation  # noqa: SLF001

        self.gemma_expert.config._attn_implementation = attn_implementation  # noqa: SLF001
        self.gemma_expert.model.config._attn_implementation = attn_implementation  # noqa: SLF001

    def embed_image(self, image: torch.Tensor):
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | Any | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        n_segments = len(inputs_embeds) if inputs_embeds is not None else 0
        if n_segments not in (2, 3):
            raise ValueError(f"Expected 2 or 3 segments in inputs_embeds, got {n_segments}")

        # Normalize 2-segment inputs to 3-segment by inserting None for world model
        _was_2_segment = n_segments == 2
        if _was_2_segment:
            inputs_embeds = [inputs_embeds[0], None, inputs_embeds[1]]
            if adarms_cond is not None:
                adarms_cond = [adarms_cond[0], None, adarms_cond[1]]

        if adarms_cond is None:
            adarms_cond = [None, None, None]
        
        prefix_output = None
        world_model_output = None
        suffix_output = None
        prefix_past_key_values = None

        # prefix-only: [prefix, None, None]
        if inputs_embeds[1] is None and inputs_embeds[2] is None and inputs_embeds[0] is not None:
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0],
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state

        # world-only: [None, wm, None]
        elif inputs_embeds[0] is None and inputs_embeds[2] is None and inputs_embeds[1] is not None:
            world_model_output = self.gemma_world_model_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1],
            )
            prefix_past_key_values = world_model_output.past_key_values
            world_model_output = world_model_output.last_hidden_state

        # suffix-only: [None, None, suffix]
        elif inputs_embeds[0] is None and inputs_embeds[1] is None and inputs_embeds[2] is not None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[2],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[2],
            )
            suffix_output = suffix_output.last_hidden_state        
        
        #inputs_embeds=[prefix_embs, wm_embs, None]
        elif inputs_embeds[0] is not None and inputs_embeds[1] is not None and inputs_embeds[2] is None:
            models = [self.paligemma.language_model, self.gemma_world_model_expert.model]

            inputs_embeds = inputs_embeds[:2]
            adarms_cond = adarms_cond[:2]

            num_layers = self.paligemma.config.text_config.num_hidden_layers

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                self.training
                and (
                    (hasattr(models[0], "gradient_checkpointing") and models[0].gradient_checkpointing)
                    or (hasattr(models[1], "gradient_checkpointing") and models[1].gradient_checkpointing)
                )
            )

            # Define the complete layer computation function for gradient checkpointing
            def compute_layer_complete(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond):
                models = [self.paligemma.language_model, self.gemma_world_model_expert.model]

                query_states = []
                key_states = []
                value_states = []
                gates = []
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])  # noqa: PLW2901
                    gates.append(gate)

                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
                    query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                    query_states.append(query_state)
                    key_states.append(key_state)
                    value_states.append(value_state)

                # Concatenate and process attention
                query_states = torch.cat(query_states, dim=2)
                key_states = torch.cat(key_states, dim=2)
                value_states = torch.cat(value_states, dim=2)

                dummy_tensor = torch.zeros(
                    query_states.shape[0],
                    query_states.shape[2],
                    query_states.shape[-1],
                    device=query_states.device,
                    dtype=query_states.dtype,
                )
                cos, sin = self.paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
                query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, unsqueeze_dim=1
                )

                batch_size = query_states.shape[0]
                scaling = self.paligemma.language_model.layers[layer_idx].self_attn.scaling

                # Attention computation
                att_output, _ = _multi_stream_attention_forward(
                    self.paligemma.language_model.layers[layer_idx].self_attn,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    scaling=scaling,
                )
                # Get head_dim from the current layer, not from the model
                head_dim = self.paligemma.language_model.layers[layer_idx].self_attn.head_dim
                att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)

                # Process layer outputs
                outputs_embeds = []
                start_pos = 0
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    end_pos = start_pos + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])

                    # first residual
                    out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])  # noqa: SLF001
                    after_first_residual = out_emb.clone()
                    out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
                    # Convert to bfloat16 if the next layer (mlp) uses bfloat16
                    if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                        out_emb = out_emb.to(dtype=torch.bfloat16)

                    out_emb = layer.mlp(out_emb)
                    # second residual
                    out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)  # noqa: SLF001
                    outputs_embeds.append(out_emb)
                    start_pos = end_pos

                return outputs_embeds

            # Process all layers with gradient checkpointing if enabled
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond
                    )

            # final norm
            # Define final norm computation function for gradient checkpointing
            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms, inputs_embeds, adarms_cond, use_reentrant=False, preserve_rng_state=False
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            world_model_output  = outputs_embeds[1]
            suffix_output = None

            prefix_past_key_values = None

        # inputs_embeds=[prefix_embs, None, suffix_embs]
        elif inputs_embeds[0] is not None and inputs_embeds[1] is None and inputs_embeds[2] is not None:
            models = [self.paligemma.language_model, self.gemma_expert.model]

            active_embeds = [inputs_embeds[0], inputs_embeds[2]]
            active_cond = [adarms_cond[0], adarms_cond[2]]

            num_layers = self.paligemma.config.text_config.num_hidden_layers

            use_gradient_checkpointing = (
                self.training
                and (
                    (hasattr(models[0], "gradient_checkpointing") and models[0].gradient_checkpointing)
                    or (hasattr(models[1], "gradient_checkpointing") and models[1].gradient_checkpointing)
                )
            )

            def compute_layer_complete(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond):
                models = [self.paligemma.language_model, self.gemma_expert.model]

                query_states = []
                key_states = []
                value_states = []
                gates = []
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])  # noqa: PLW2901
                    gates.append(gate)

                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
                    query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                    query_states.append(query_state)
                    key_states.append(key_state)
                    value_states.append(value_state)

                query_states = torch.cat(query_states, dim=2)
                key_states = torch.cat(key_states, dim=2)
                value_states = torch.cat(value_states, dim=2)

                dummy_tensor = torch.zeros(
                    query_states.shape[0],
                    query_states.shape[2],
                    query_states.shape[-1],
                    device=query_states.device,
                    dtype=query_states.dtype,
                )
                cos, sin = self.paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
                query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, unsqueeze_dim=1
                )

                batch_size = query_states.shape[0]
                scaling = self.paligemma.language_model.layers[layer_idx].self_attn.scaling

                att_output, _ = _multi_stream_attention_forward(
                    self.paligemma.language_model.layers[layer_idx].self_attn,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    scaling=scaling,
                )
                head_dim = self.paligemma.language_model.layers[layer_idx].self_attn.head_dim
                att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)

                outputs_embeds = []
                start_pos = 0
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    end_pos = start_pos + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])

                    out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])  # noqa: SLF001
                    after_first_residual = out_emb.clone()
                    out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
                    if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                        out_emb = out_emb.to(dtype=torch.bfloat16)

                    out_emb = layer.mlp(out_emb)
                    out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)  # noqa: SLF001
                    outputs_embeds.append(out_emb)
                    start_pos = end_pos

                return outputs_embeds

            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    active_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        active_embeds,
                        attention_mask,
                        position_ids,
                        active_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    active_embeds = compute_layer_complete(
                        layer_idx, active_embeds, attention_mask, position_ids, active_cond
                    )

            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms, active_embeds, active_cond, use_reentrant=False, preserve_rng_state=False
                )
            else:
                outputs_embeds = compute_final_norms(active_embeds, active_cond)

            prefix_output = outputs_embeds[0]
            world_model_output = None
            suffix_output = outputs_embeds[1]

            prefix_past_key_values = None

        # inputs_embeds=[prefix_embs, wm_embs, suffix_embs]
        elif inputs_embeds[0] is not None and inputs_embeds[1] is not None and inputs_embeds[2] is not None:
            models = [self.paligemma.language_model, self.gemma_world_model_expert.model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

            # Respect the per-module checkpointing flags; do not silently force GC on.
            use_gradient_checkpointing = (
                any(
                    hasattr(module, "gradient_checkpointing") and module.gradient_checkpointing
                    for module in (
                        self.paligemma.language_model,
                        self.gemma_world_model_expert.model,
                        self.gemma_expert.model,
                    )
                )
                or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing)
            ) and self.training

            # Define the complete layer computation function for gradient checkpointing
            def compute_layer_complete(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond):
                models = [self.paligemma.language_model, self.gemma_world_model_expert.model, self.gemma_expert.model]
                query_states = []
                key_states = []
                value_states = []
                gates = []
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])  # noqa: PLW2901
                    gates.append(gate)

                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
                    query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                    query_states.append(query_state)
                    key_states.append(key_state)
                    value_states.append(value_state)

                # Concatenate and process attention
                query_states = torch.cat(query_states, dim=2)
                key_states = torch.cat(key_states, dim=2)
                value_states = torch.cat(value_states, dim=2)

                dummy_tensor = torch.zeros(
                    query_states.shape[0],
                    query_states.shape[2],
                    query_states.shape[-1],
                    device=query_states.device,
                    dtype=query_states.dtype,
                )
                cos, sin = self.paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
                query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, unsqueeze_dim=1
                )

                batch_size = query_states.shape[0]
                scaling = self.paligemma.language_model.layers[layer_idx].self_attn.scaling

                # Attention computation
                att_output, _ = _multi_stream_attention_forward(
                    self.paligemma.language_model.layers[layer_idx].self_attn,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    scaling=scaling,
                )
                # Get head_dim from the current layer, not from the model
                head_dim = self.paligemma.language_model.layers[layer_idx].self_attn.head_dim
                att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)

                # Process layer outputs
                outputs_embeds = []
                start_pos = 0
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    end_pos = start_pos + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])

                    # first residual
                    out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])  # noqa: SLF001
                    after_first_residual = out_emb.clone()
                    out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
                    # Convert to bfloat16 if the next layer (mlp) uses bfloat16
                    if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                        out_emb = out_emb.to(dtype=torch.bfloat16)

                    out_emb = layer.mlp(out_emb)
                    # second residual
                    out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)  # noqa: SLF001
                    outputs_embeds.append(out_emb)
                    start_pos = end_pos

                return outputs_embeds

            # Process all layers with gradient checkpointing if enabled
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond
                    )

                # Old code removed - now using compute_layer_complete function above

            # final norm
            # Define final norm computation function for gradient checkpointing
            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms, inputs_embeds, adarms_cond, use_reentrant=False, preserve_rng_state=False
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            world_model_output = outputs_embeds[1]
            suffix_output = outputs_embeds[2]

            prefix_past_key_values = None

        if _was_2_segment:
            return [prefix_output, suffix_output], prefix_past_key_values
        return [prefix_output, world_model_output, suffix_output], prefix_past_key_values
