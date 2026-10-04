import types

import torch

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.pi0_pytorch import shared_bernoulli
from openpi.models_pytorch.pi0_pytorch import _compute_last_step_collapse_metrics
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.models_pytorch.pi0_pytorch import _compute_wm_loss_per_batch
from openpi.models_pytorch.world_model_pytorch import WorldModelFutureSlotBuilder
from openpi.models_pytorch.world_model_pytorch import WorldModelEmbeddings
from openpi.models_pytorch.world_model_pytorch import WorldModelPredictorHead


def test_shared_bernoulli_is_rank_local_without_ddp(monkeypatch):
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor(0.0))
    assert shared_bernoulli(0.3, torch.device("cpu")) is True
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor(0.9))
    assert shared_bernoulli(0.3, torch.device("cpu")) is False
    assert shared_bernoulli(0.0, torch.device("cpu")) is False
    assert shared_bernoulli(1.0, torch.device("cpu")) is True


def test_shared_bernoulli_broadcasts_rank0_decision(monkeypatch):
    seen = {}

    def broadcast(tensor, src):
        seen["value"] = int(tensor.item())
        seen["src"] = src
        tensor.fill_(1)

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 3)
    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)

    assert shared_bernoulli(0.3, torch.device("cpu")) is True
    assert seen == {"value": 0, "src": 0}


def test_make_att_2d_masks_respects_read_masked_keys():
    att_2d = make_att_2d_masks(
        torch.tensor([[True, True, True]], dtype=torch.bool),
        torch.tensor([[False, False, False]], dtype=torch.bool),
        read_masks=torch.tensor([[True, False, True]], dtype=torch.bool),
    )

    torch.testing.assert_close(
        att_2d,
        torch.tensor(
            [[[True, False, True], [True, False, True], [True, False, True]]],
            dtype=torch.bool,
        ),
    )


def test_prepare_attention_masks_4d_preserves_requested_dtype():
    model = object.__new__(PI0Pytorch)
    att_2d = torch.tensor(
        [[[True, False], [True, True]]],
        dtype=torch.bool,
    )

    mask = PI0Pytorch._prepare_attention_masks_4d(model, att_2d, dtype=torch.bfloat16)

    assert mask.dtype == torch.bfloat16
    assert mask.shape == (1, 1, 2, 2)
    torch.testing.assert_close(mask[0, 0, 0, 0], torch.tensor(0.0, dtype=torch.bfloat16))
    torch.testing.assert_close(mask[0, 0, 0, 1], torch.tensor(torch.finfo(torch.bfloat16).min, dtype=torch.bfloat16))


def test_compute_wm_loss_per_batch_normalizes_by_original_future_token_horizon():
    wm_pred = torch.tensor(
        [
            [[1.0, 1.0], [0.0, 0.0]],
            [[1.0, 1.0], [2.0, 2.0]],
        ],
        dtype=torch.float32,
    )
    wm_target = torch.zeros_like(wm_pred)
    wm_target_mask = torch.tensor([[True, False], [True, True]], dtype=torch.bool)

    wm_loss = _compute_wm_loss_per_batch(
        wm_pred,
        wm_target,
        wm_target_mask,
        future_token_loss_normalizer=2,
    )

    torch.testing.assert_close(wm_loss, torch.tensor([0.5, 2.5], dtype=torch.float32))


def test_compute_last_step_collapse_metrics_uses_absolute_last_bin_without_spatial_pooling():
    wm_pred = torch.tensor(
        [
            [[100.0], [100.0], [0.0], [10.0]],
            [[200.0], [200.0], [4.0], [6.0]],
            [[300.0], [300.0], [8.0], [2.0]],
        ],
        dtype=torch.float32,
    )
    wm_target = wm_pred.clone()
    wm_target_mask = torch.ones((3, 4), dtype=torch.bool)

    metrics = _compute_last_step_collapse_metrics(
        wm_pred,
        wm_target,
        wm_target_mask,
        tokens_per_temporal_bin=2,
    )

    torch.testing.assert_close(metrics["wm_last_step_abs_pred_var"], torch.tensor(10.666667, dtype=torch.float32))
    torch.testing.assert_close(metrics["wm_last_step_abs_target_var"], torch.tensor(10.666667, dtype=torch.float32))
    torch.testing.assert_close(metrics["wm_last_step_abs_var_ratio"], torch.tensor(1.0, dtype=torch.float32))
    torch.testing.assert_close(metrics["wm_last_step_abs_collapse_score"], torch.tensor(1.0, dtype=torch.float32))
    torch.testing.assert_close(metrics["wm_last_step_eligible_count"], torch.tensor(3.0, dtype=torch.float32))
    torch.testing.assert_close(metrics["wm_last_step_eligible_frac"], torch.tensor(1.0, dtype=torch.float32))


def test_compute_last_step_collapse_metrics_tracks_eligible_support():
    wm_pred = torch.tensor(
        [
            [[1.0], [1.0], [1.0], [1.0]],
            [[2.0], [2.0], [2.0], [2.0]],
        ],
        dtype=torch.float32,
    )
    wm_target = wm_pred.clone()
    wm_target_mask = torch.tensor(
        [
            [True, True, True, True],
            [True, True, False, False],
        ],
        dtype=torch.bool,
    )

    metrics = _compute_last_step_collapse_metrics(
        wm_pred,
        wm_target,
        wm_target_mask,
        tokens_per_temporal_bin=2,
    )

    torch.testing.assert_close(metrics["wm_last_step_abs_pred_var"], torch.tensor(0.0, dtype=torch.float32))
    torch.testing.assert_close(metrics["wm_last_step_abs_target_var"], torch.tensor(0.0, dtype=torch.float32))
    torch.testing.assert_close(metrics["wm_last_step_abs_var_ratio"], torch.tensor(0.0, dtype=torch.float32))
    torch.testing.assert_close(metrics["wm_last_step_abs_collapse_score"], torch.tensor(0.0, dtype=torch.float32))
    torch.testing.assert_close(metrics["wm_last_step_eligible_count"], torch.tensor(1.0, dtype=torch.float32))
    torch.testing.assert_close(metrics["wm_last_step_eligible_frac"], torch.tensor(0.5, dtype=torch.float32))


def test_parallel_slot_future_rollout_restores_one_shot_slot_path():
    model = PI0Pytorch.__new__(PI0Pytorch)
    torch.nn.Module.__init__(model)
    model.world_pred_head = torch.nn.Identity()
    model.gradient_checkpointing_enabled = False

    def fake_run_prefix_world(
        self,
        prefix_embs,
        prefix_pad_masks,
        prefix_att_masks,
        wm_embs,
        wm_pad_masks,
        wm_read_masks,
        wm_att_masks,
    ):
        del prefix_embs, prefix_pad_masks, prefix_att_masks
        del wm_pad_masks, wm_read_masks, wm_att_masks
        return wm_embs.to(dtype=torch.float32) + 7.0

    model._run_prefix_world = types.MethodType(fake_run_prefix_world, model)

    prefix_embs = torch.zeros((1, 0, 1), dtype=torch.float32)
    prefix_pad_masks = torch.zeros((1, 0), dtype=torch.bool)
    prefix_att_masks = torch.zeros((1, 0), dtype=torch.bool)
    wm_embs = torch.tensor([[[1.0], [2.0], [10.0], [20.0]]], dtype=torch.float32)
    wm_pad_masks = torch.ones((1, 4), dtype=torch.bool)
    wm_read_masks = torch.ones((1, 4), dtype=torch.bool)
    wm_att_masks = torch.tensor([[True, False, True, False]], dtype=torch.bool)

    wm_pred, conditioned_future = model._parallel_slot_future_rollout(
        prefix_embs,
        prefix_pad_masks,
        prefix_att_masks,
        wm_embs,
        wm_pad_masks,
        wm_read_masks,
        wm_att_masks,
        history_len=2,
    )

    torch.testing.assert_close(wm_pred, torch.tensor([[[17.0], [27.0]]], dtype=torch.float32))
    torch.testing.assert_close(conditioned_future, wm_pred)


def test_set_training_stage_wm_alignment_freezes_unused_world_lm_head():
    class DummyCausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Linear(2, 2)
            self.lm_head = torch.nn.Linear(2, 2, bias=False)

    class DummyExpertBundle(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gemma_world_model_expert = DummyCausalLM()
            self.paligemma = torch.nn.Linear(2, 2)
            self.gemma_expert = DummyCausalLM()

    class DummyWorldAdapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._vjepa2_model = torch.nn.Linear(2, 2)

        @property
        def encoder_module(self):
            return self._vjepa2_model

    model = PI0Pytorch.__new__(PI0Pytorch)
    torch.nn.Module.__init__(model)
    model.enable_world_model = True
    model.pi05 = True
    model.paligemma_with_expert = DummyExpertBundle()
    model.world_pred_head = torch.nn.Linear(2, 2)
    model.world_future_builder = torch.nn.Linear(2, 2)
    model.world_model_adapter = DummyWorldAdapter()
    model.action_in_proj = torch.nn.Linear(2, 2)
    model.action_out_proj = torch.nn.Linear(2, 2)
    model.time_mlp_in = torch.nn.Linear(2, 2)
    model.time_mlp_out = torch.nn.Linear(2, 2)

    model.set_training_stage("wm_alignment")

    assert all(p.requires_grad for p in model.paligemma_with_expert.gemma_world_model_expert.model.parameters())
    assert all(not p.requires_grad for p in model.paligemma_with_expert.gemma_world_model_expert.lm_head.parameters())
    assert all(p.requires_grad for p in model.world_pred_head.parameters())
    assert all(p.requires_grad for p in model.world_future_builder.parameters())


def test_embed_prefix_keeps_vision_path_in_graph_when_images_are_masked():
    class DummyExpertBundle(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.image_proj = torch.nn.Linear(4, 4, bias=False)
            self.lang_embed = torch.nn.Embedding(8, 4)

        def embed_image(self, image):
            flat = image.reshape(image.shape[0], -1)
            return self.image_proj(flat).unsqueeze(1)

        def embed_language_tokens(self, tokens):
            return self.lang_embed(tokens)

    model = PI0Pytorch.__new__(PI0Pytorch)
    torch.nn.Module.__init__(model)
    model.gradient_checkpointing_enabled = False
    model.paligemma_with_expert = DummyExpertBundle()
    model.train()

    prefix_embs, _, _ = model.embed_prefix(
        images=[torch.randn(2, 1, 2, 2, dtype=torch.float32)],
        img_masks=[torch.zeros(2, dtype=torch.bool)],
        lang_tokens=torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long),
        lang_masks=torch.ones((2, 3), dtype=torch.bool),
    )

    loss = prefix_embs[:, 1:, :].sum()
    loss.backward()

    grad = model.paligemma_with_expert.image_proj.weight.grad
    assert grad is not None
    torch.testing.assert_close(grad, torch.zeros_like(grad))


def test_build_inference_world_inputs_uses_parallel_predictions():
    model = PI0Pytorch.__new__(PI0Pytorch)
    torch.nn.Module.__init__(model)
    model.world_pred_head = torch.nn.Identity()
    model.world_model_adapter = types.SimpleNamespace(spatial_tokens_per_temporal_bin=2)
    model.gradient_checkpointing_enabled = False

    def fake_run_prefix_world(
        self,
        prefix_embs,
        prefix_pad_masks,
        prefix_att_masks,
        wm_embs,
        wm_pad_masks,
        wm_read_masks,
        wm_att_masks,
    ):
        del prefix_embs, prefix_pad_masks, prefix_att_masks
        del wm_pad_masks, wm_read_masks, wm_att_masks
        return wm_embs.to(dtype=torch.float32) + 5.0

    model._run_prefix_world = types.MethodType(fake_run_prefix_world, model)

    wm_embeddings = WorldModelEmbeddings(
        embeddings=torch.tensor([[[1.0], [2.0], [3.0], [4.0]]], dtype=torch.float32),
        pad_mask=torch.tensor([[True, True, True, True]], dtype=torch.bool),
        att_mask=torch.tensor([[True, False, True, False]], dtype=torch.bool),
        read_mask=torch.tensor([[True, True, False, True]], dtype=torch.bool),
        history_token_len=2,
    )

    wm_embs, wm_pad_masks, wm_read_masks, wm_att_masks = model._build_inference_world_inputs(
        torch.zeros((1, 0, 1), dtype=torch.float32),
        torch.zeros((1, 0), dtype=torch.bool),
        torch.zeros((1, 0), dtype=torch.bool),
        wm_embeddings,
    )

    torch.testing.assert_close(wm_embs, torch.tensor([[[1.0], [2.0], [8.0], [9.0]]], dtype=torch.float32))
    torch.testing.assert_close(wm_pad_masks, torch.tensor([[True, True, False, True]], dtype=torch.bool))
    torch.testing.assert_close(wm_read_masks, torch.tensor([[True, True, False, True]], dtype=torch.bool))
    torch.testing.assert_close(wm_att_masks, torch.tensor([[True, False, True, False]], dtype=torch.bool))

def test_wm_alignment_world_only_backward_hits_input_projector_world_expert_and_predictor():
    class ProjectedAdapter(torch.nn.Module):
        tubelet_size = 2
        spatial_tokens_per_temporal_bin = 2
        enable_input_projector = True

        def __init__(self):
            super().__init__()
            self.input_projector = torch.nn.Linear(4, 6, bias=False)

        def apply_input_projector(self, tokens):
            return self.input_projector(tokens)

        def temporal_bins_for_num_frames(self, num_frames: int) -> int:
            if num_frames <= 0:
                return 0
            return 1 if num_frames < self.tubelet_size else num_frames // self.tubelet_size

        def token_count_for_num_frames(self, num_frames: int) -> int:
            return self.temporal_bins_for_num_frames(num_frames) * self.spatial_tokens_per_temporal_bin

        def frame_mask_to_temporal_mask(self, frame_mask: torch.Tensor, *, num_frames: int | None = None) -> torch.Tensor:
            frame_mask = torch.as_tensor(frame_mask, dtype=torch.bool)
            if frame_mask.ndim == 1:
                frame_mask = frame_mask[None, :]
            total_frames = frame_mask.shape[1] if num_frames is None else int(num_frames)
            if total_frames == 0:
                return torch.zeros((frame_mask.shape[0], 0), dtype=torch.bool, device=frame_mask.device)
            if total_frames < self.tubelet_size:
                return frame_mask[:, :1]
            temporal_bins = self.temporal_bins_for_num_frames(total_frames)
            consumed_frames = temporal_bins * self.tubelet_size
            return frame_mask[:, :consumed_frames].reshape(
                frame_mask.shape[0], temporal_bins, self.tubelet_size
            ).all(dim=2)

        def expand_frame_mask_to_token_mask(
            self,
            frame_mask: torch.Tensor,
            *,
            num_frames: int | None = None,
            token_count: int | None = None,
        ) -> torch.Tensor:
            token_mask = self.frame_mask_to_temporal_mask(frame_mask, num_frames=num_frames).repeat_interleave(
                self.spatial_tokens_per_temporal_bin,
                dim=1,
            )
            if token_count is not None and token_mask.shape[1] != token_count:
                raise ValueError(f"Expected token_count={token_count}, got {token_mask.shape[1]}")
            return token_mask

        def forward(self, video_frames, skip_predictor: bool = True):
            del skip_predictor
            batch_size, num_frames = video_frames.shape[:2]
            num_tokens = self.token_count_for_num_frames(num_frames)
            base = torch.arange(
                num_tokens,
                dtype=torch.float32,
                device=video_frames.device,
            ).reshape(1, num_tokens, 1).expand(batch_size, -1, 4)
            pad_mask = torch.ones((batch_size, num_tokens), dtype=torch.bool, device=video_frames.device)
            att_mask = torch.zeros((batch_size, num_tokens), dtype=torch.bool, device=video_frames.device)
            return WorldModelEmbeddings(
                embeddings=base,
                pad_mask=pad_mask,
                att_mask=att_mask,
                read_mask=pad_mask,
            )

    class Observation:
        def __init__(self):
            self.images = {
                "base_0_rgb_history": torch.zeros((2, 4, 8, 8, 3), dtype=torch.float32),
                "base_0_rgb_future": torch.zeros((2, 4, 8, 8, 3), dtype=torch.float32),
            }
            self.image_masks = {
                "base_0_rgb_history": torch.ones((2, 4), dtype=torch.bool),
                "base_0_rgb_future": torch.ones((2, 4), dtype=torch.bool),
            }

    adapter = ProjectedAdapter()
    builder = WorldModelFutureSlotBuilder(adapter=adapter, embed_dim=6, slot_max_len=8)

    model = PI0Pytorch.__new__(PI0Pytorch)
    torch.nn.Module.__init__(model)
    model.gradient_checkpointing_enabled = False
    model.world_model_adapter = adapter
    model.world_pred_head = WorldModelPredictorHead(6, 4)
    model.world_model_expert = torch.nn.Linear(6, 6)

    def fake_run_prefix_world(
        self,
        prefix_embs,
        prefix_pad_masks,
        prefix_att_masks,
        wm_embs,
        wm_pad_masks,
        wm_read_masks,
        wm_att_masks,
    ):
        del prefix_embs, prefix_pad_masks, prefix_att_masks, wm_pad_masks, wm_read_masks, wm_att_masks
        context = wm_embs.mean(dim=1, keepdim=True)
        return self.world_model_expert(wm_embs + context)

    model._run_prefix_world = types.MethodType(fake_run_prefix_world, model)

    wm_batch = builder.build_training_inputs(Observation())
    total_loss, _ = model._forward_world_only(
        torch.zeros((2, 0, 6), dtype=torch.float32),
        torch.zeros((2, 0), dtype=torch.bool),
        torch.zeros((2, 0), dtype=torch.bool),
        wm_batch,
    )

    loss = total_loss.mean()
    assert torch.isfinite(loss)
    loss.backward()

    assert adapter.input_projector.weight.grad is not None
    assert model.world_model_expert.weight.grad is not None
    assert model.world_pred_head.proj.weight.grad is not None
    assert torch.isfinite(adapter.input_projector.weight.grad).all()
    assert torch.isfinite(model.world_model_expert.weight.grad).all()
    assert torch.isfinite(model.world_pred_head.proj.weight.grad).all()
