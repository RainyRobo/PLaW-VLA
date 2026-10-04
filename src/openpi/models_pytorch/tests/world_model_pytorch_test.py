import logging
import types

import torch
import pytest
from transformers.video_utils import make_batched_videos

import openpi.models_pytorch.world_model_pytorch as world_model_pytorch
from openpi.models_pytorch.world_model_pytorch import VJepa2Adapter


_FAKE_VJEPA_HIDDEN_SIZE = 1024


class _FakePretrainedModel(torch.nn.Module):
    def __init__(self, hidden_size: int = _FAKE_VJEPA_HIDDEN_SIZE):
        super().__init__()
        self.fake_param = torch.nn.Parameter(torch.zeros(1))
        self.config = types.SimpleNamespace(
            _name_or_path="fake-vjepa2",
            tubelet_size=2,
            patch_size=16,
            crop_size=32,
            hidden_size=hidden_size,
        )

    def to(self, device):
        self.loaded_device = device
        return self


class _FakeAdapter(torch.nn.Module):
    tubelet_size = 2
    spatial_tokens_per_temporal_bin = 2
    enable_input_projector = False

    def __init__(self):
        super().__init__()
        self.recorded_shapes: list[tuple[int, ...]] = []

    def apply_input_projector(self, tokens):
        return tokens

    def temporal_bins_for_num_frames(self, num_frames: int) -> int:
        if num_frames < 0:
            raise ValueError(f"num_frames must be >= 0, got {num_frames}")
        if num_frames == 0:
            return 0
        if num_frames < self.tubelet_size:
            return 1
        return num_frames // self.tubelet_size

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
        return frame_mask[:, :consumed_frames].reshape(frame_mask.shape[0], temporal_bins, self.tubelet_size).all(dim=2)

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
        self.recorded_shapes.append(tuple(video_frames.shape))
        batch_size, num_frames = video_frames.shape[:2]
        num_tokens = self.token_count_for_num_frames(num_frames)
        embeddings = torch.arange(
            num_tokens,
            dtype=torch.float32,
            device=video_frames.device,
        ).reshape(1, num_tokens, 1).expand(batch_size, -1, 4)
        pad_mask = torch.ones((batch_size, num_tokens), dtype=torch.bool, device=video_frames.device)
        att_mask = torch.zeros((batch_size, num_tokens), dtype=torch.bool, device=video_frames.device)
        return world_model_pytorch.WorldModelEmbeddings(
            embeddings=embeddings,
            pad_mask=pad_mask,
            att_mask=att_mask,
            read_mask=pad_mask,
        )


class _GradAdapter(_FakeAdapter):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, video_frames, skip_predictor: bool = True):
        del skip_predictor
        self.recorded_shapes.append(tuple(video_frames.shape))
        batch_size, num_frames = video_frames.shape[:2]
        num_tokens = self.token_count_for_num_frames(num_frames)
        base = torch.arange(
            num_tokens,
            dtype=torch.float32,
            device=video_frames.device,
        ).reshape(1, num_tokens, 1).expand(batch_size, -1, 4)
        embeddings = base * self.scale
        pad_mask = torch.ones((batch_size, num_tokens), dtype=torch.bool, device=video_frames.device)
        att_mask = torch.zeros((batch_size, num_tokens), dtype=torch.bool, device=video_frames.device)
        return world_model_pytorch.WorldModelEmbeddings(
            embeddings=embeddings,
            pad_mask=pad_mask,
            att_mask=att_mask,
            read_mask=pad_mask,
        )


class _ProjectedAdapter(_FakeAdapter):
    enable_input_projector = True

    def __init__(self, raw_dim: int = 4, projected_dim: int = 6):
        super().__init__()
        self.input_projector = torch.nn.Linear(raw_dim, projected_dim, bias=False)

    def apply_input_projector(self, tokens):
        return self.input_projector(tokens)


def test_prepare_videos_for_processor_splits_batched_channel_last_input():
    frames = torch.zeros((4, 3, 256, 256, 3), dtype=torch.uint8)

    videos = VJepa2Adapter._prepare_videos_for_processor(frames)

    assert len(videos) == 4
    assert all(video.shape == (3, 256, 256, 3) for video in videos)
    assert all(video.ndim == 4 for video in make_batched_videos(videos))


def test_prepare_videos_for_processor_splits_batched_channel_first_input():
    frames = torch.zeros((4, 3, 3, 256, 256), dtype=torch.uint8)

    videos = VJepa2Adapter._prepare_videos_for_processor(frames)

    assert len(videos) == 4
    assert all(video.shape == (3, 3, 256, 256) for video in videos)
    assert all(video.ndim == 4 for video in make_batched_videos(videos))


def test_prepare_videos_for_processor_wraps_single_video():
    frames = torch.zeros((3, 256, 256, 3), dtype=torch.uint8)

    videos = VJepa2Adapter._prepare_videos_for_processor(frames)

    assert len(videos) == 1
    assert videos[0].shape == (3, 256, 256, 3)


def test_vjepa2_adapter_preserves_raw_patch_tokens(monkeypatch):
    class _FakeProcessor:
        def __call__(self, videos, return_tensors="pt"):
            assert return_tensors == "pt"
            return {"pixel_values_videos": torch.stack([torch.as_tensor(video) for video in videos], dim=0)}

    class _FakeTokenModel(_FakePretrainedModel):
        def forward(self, pixel_values_videos, skip_predictor=True):
            del skip_predictor
            batch_size, num_frames = pixel_values_videos.shape[:2]
            spatial_tokens = (self.config.crop_size // self.config.patch_size) ** 2
            if num_frames == 0:
                num_tokens = 0
            elif num_frames < self.config.tubelet_size:
                num_tokens = spatial_tokens
            else:
                num_tokens = (num_frames // self.config.tubelet_size) * spatial_tokens
            token_ids = torch.arange(
                batch_size * num_tokens,
                dtype=torch.float32,
                device=pixel_values_videos.device,
            ).reshape(batch_size, num_tokens, 1)
            last_hidden_state = token_ids.expand(-1, -1, _FAKE_VJEPA_HIDDEN_SIZE)
            return types.SimpleNamespace(last_hidden_state=last_hidden_state)

    monkeypatch.setattr(world_model_pytorch.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        world_model_pytorch,
        "_AutoModel",
        types.SimpleNamespace(from_pretrained=lambda model_name: _FakeTokenModel()),
    )
    monkeypatch.setattr(world_model_pytorch, "hf_hub_download", lambda **kwargs: "/tmp/video_preprocessor_config.json")
    monkeypatch.setattr(
        world_model_pytorch,
        "_VJEPA2VideoProcessor",
        types.SimpleNamespace(from_pretrained=lambda path: _FakeProcessor()),
    )

    adapter = VJepa2Adapter(
        model_name="facebook/vjepa2-vitl-fpc64-256",
        device="cuda",
        expected_embedding_dim=_FAKE_VJEPA_HIDDEN_SIZE,
    )

    frames = torch.zeros((1, 3, 8, 8, 3), dtype=torch.uint8)
    outputs = adapter(frames)

    torch.testing.assert_close(
        outputs.embeddings[..., 0],
        torch.tensor([[0.0, 1.0, 2.0, 3.0]], dtype=torch.float32),
    )
    assert outputs.embeddings.shape == (1, 4, _FAKE_VJEPA_HIDDEN_SIZE)
    torch.testing.assert_close(outputs.pad_mask, torch.ones((1, 4), dtype=torch.bool))
    torch.testing.assert_close(outputs.att_mask, torch.zeros((1, 4), dtype=torch.bool))
    torch.testing.assert_close(outputs.read_mask, torch.ones((1, 4), dtype=torch.bool))


def test_vjepa2_adapter_accepts_matching_direct_embedding_dim(monkeypatch):
    monkeypatch.setattr(world_model_pytorch.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        world_model_pytorch,
        "_AutoModel",
        types.SimpleNamespace(from_pretrained=lambda model_name: _FakePretrainedModel()),
    )
    monkeypatch.setattr(world_model_pytorch, "hf_hub_download", lambda **kwargs: "/tmp/video_preprocessor_config.json")
    monkeypatch.setattr(
        world_model_pytorch,
        "_VJEPA2VideoProcessor",
        types.SimpleNamespace(from_pretrained=lambda path: object()),
    )

    adapter = VJepa2Adapter(
        model_name="facebook/vjepa2-vitl-fpc64-256",
        device="cuda",
        expected_embedding_dim=_FAKE_VJEPA_HIDDEN_SIZE,
    )

    assert adapter.embedding_dim == _FAKE_VJEPA_HIDDEN_SIZE


def test_world_model_config_uses_shared_vjepa2_metadata(monkeypatch):
    captured = {}

    def fake_resolve_repo_id(variant, override=None):
        captured["repo_lookup"] = (variant, override)
        return "resolved-from-helper"

    def fake_resolve_hidden_size(variant):
        captured["hidden_lookup"] = variant
        return 1152

    def fake_from_pretrained(model_name):
        captured["model_name"] = model_name
        return _FakePretrainedModel(hidden_size=1152)

    monkeypatch.setattr(world_model_pytorch._vjepa2, "resolve_repo_id", fake_resolve_repo_id)
    monkeypatch.setattr(world_model_pytorch._vjepa2, "resolve_hidden_size", fake_resolve_hidden_size)
    monkeypatch.setattr(world_model_pytorch.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        world_model_pytorch,
        "_AutoModel",
        types.SimpleNamespace(from_pretrained=fake_from_pretrained),
    )
    monkeypatch.setattr(world_model_pytorch, "hf_hub_download", lambda **kwargs: "/tmp/video_preprocessor_config.json")
    monkeypatch.setattr(
        world_model_pytorch,
        "_VJEPA2VideoProcessor",
        types.SimpleNamespace(from_pretrained=lambda path: object()),
    )

    adapter = world_model_pytorch.WorldModelConfig(
        variant="vith-256",
        model_name_override=None,
        expected_embedding_dim=1152,
        device="cuda",
    ).build_adapter()

    assert captured["repo_lookup"] == ("vith-256", None)
    assert captured["hidden_lookup"] == "vith-256"
    assert captured["model_name"] == "resolved-from-helper"
    assert adapter.embedding_dim == 1152


def test_vjepa2_adapter_rejects_mismatched_embedding_dim(monkeypatch):
    monkeypatch.setattr(world_model_pytorch.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        world_model_pytorch,
        "_AutoModel",
        types.SimpleNamespace(from_pretrained=lambda model_name: _FakePretrainedModel()),
    )
    monkeypatch.setattr(world_model_pytorch, "hf_hub_download", lambda **kwargs: "/tmp/video_preprocessor_config.json")
    monkeypatch.setattr(
        world_model_pytorch,
        "_VJEPA2VideoProcessor",
        types.SimpleNamespace(from_pretrained=lambda path: object()),
    )

    with pytest.raises(ValueError, match="does not match the expected width"):
        VJepa2Adapter(
            model_name="facebook/vjepa2-vitl-fpc64-256",
            device="cuda",
            expected_embedding_dim=16,
        )


def test_vjepa2_adapter_rejects_checkpoint_mismatch_with_canonical_hidden_size(monkeypatch):
    monkeypatch.setattr(world_model_pytorch.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        world_model_pytorch,
        "_AutoModel",
        types.SimpleNamespace(from_pretrained=lambda model_name: _FakePretrainedModel(hidden_size=1024)),
    )
    monkeypatch.setattr(world_model_pytorch, "hf_hub_download", lambda **kwargs: "/tmp/video_preprocessor_config.json")
    monkeypatch.setattr(
        world_model_pytorch,
        "_VJEPA2VideoProcessor",
        types.SimpleNamespace(from_pretrained=lambda path: object()),
    )

    with pytest.raises(ValueError, match="canonical metadata"):
        VJepa2Adapter(
            model_name="facebook/vjepa2-vitl-fpc64-256",
            device="cuda",
            expected_vjepa_hidden_size=1408,
        )


def test_vjepa2_adapter_expands_frame_masks_to_raw_token_masks(monkeypatch):
    monkeypatch.setattr(world_model_pytorch.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        world_model_pytorch,
        "_AutoModel",
        types.SimpleNamespace(from_pretrained=lambda model_name: _FakePretrainedModel()),
    )
    monkeypatch.setattr(world_model_pytorch, "hf_hub_download", lambda **kwargs: "/tmp/video_preprocessor_config.json")
    monkeypatch.setattr(
        world_model_pytorch,
        "_VJEPA2VideoProcessor",
        types.SimpleNamespace(from_pretrained=lambda path: object()),
    )

    adapter = VJepa2Adapter(
        model_name="facebook/vjepa2-vitl-fpc64-256",
        device="cuda",
    )

    assert adapter.temporal_bins_for_num_frames(1) == 1
    assert adapter.temporal_bins_for_num_frames(3) == 1
    assert adapter.token_count_for_num_frames(3) == 4
    torch.testing.assert_close(
        adapter.frame_mask_to_temporal_mask(torch.tensor([[True, False, True]], dtype=torch.bool)),
        torch.tensor([[False]], dtype=torch.bool),
    )
    torch.testing.assert_close(
        adapter.expand_frame_mask_to_token_mask(torch.tensor([[True, False, True]], dtype=torch.bool)),
        torch.tensor([[False, False, False, False]], dtype=torch.bool),
    )


def test_vjepa2_adapter_loads_remote_video_processor_from_explicit_config(monkeypatch):
    captured = {}

    monkeypatch.setattr(world_model_pytorch.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        world_model_pytorch,
        "_AutoModel",
        types.SimpleNamespace(from_pretrained=lambda model_name: _FakePretrainedModel()),
    )

    def fake_hf_hub_download(*, repo_id, filename):
        captured["repo_id"] = repo_id
        captured["filename"] = filename
        return "/tmp/video_preprocessor_config.json"

    monkeypatch.setattr(world_model_pytorch, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(
        world_model_pytorch,
        "_VJEPA2VideoProcessor",
        types.SimpleNamespace(from_pretrained=lambda path: {"processor_path": path}),
    )

    VJepa2Adapter(
        model_name="facebook/vjepa2-vitl-fpc64-256",
        device="cuda",
    )

    assert captured == {
        "repo_id": "facebook/vjepa2-vitl-fpc64-256",
        "filename": VJepa2Adapter.DEFAULT_PROCESSOR_CONFIG_NAME,
    }


def test_vjepa2_adapter_loads_local_video_processor_config(monkeypatch, tmp_path):
    processor_config = tmp_path / VJepa2Adapter.DEFAULT_PROCESSOR_CONFIG_NAME
    processor_config.write_text("{}", encoding="utf-8")

    captured = {}
    monkeypatch.setattr(world_model_pytorch.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        world_model_pytorch,
        "_AutoModel",
        types.SimpleNamespace(from_pretrained=lambda model_name: _FakePretrainedModel()),
    )
    monkeypatch.setattr(
        world_model_pytorch,
        "hf_hub_download",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("hf_hub_download should not be called")),
    )
    monkeypatch.setattr(
        world_model_pytorch,
        "_VJEPA2VideoProcessor",
        types.SimpleNamespace(from_pretrained=lambda path: captured.setdefault("processor_path", path)),
    )

    VJepa2Adapter(
        model_name=str(tmp_path),
        device="cuda",
    )

    assert captured["processor_path"] == str(processor_config)


def test_vjepa2_adapter_quiets_httpx_and_hf_hub_loggers(monkeypatch):
    monkeypatch.setattr(world_model_pytorch.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        world_model_pytorch,
        "_AutoModel",
        types.SimpleNamespace(from_pretrained=lambda model_name: _FakePretrainedModel()),
    )
    monkeypatch.setattr(world_model_pytorch, "hf_hub_download", lambda **kwargs: "/tmp/video_preprocessor_config.json")
    monkeypatch.setattr(
        world_model_pytorch,
        "_VJEPA2VideoProcessor",
        types.SimpleNamespace(from_pretrained=lambda path: object()),
    )

    httpx_logger = logging.getLogger("httpx")
    hub_logger = logging.getLogger("huggingface_hub")
    old_httpx_level = httpx_logger.level
    old_hub_level = hub_logger.level
    try:
        httpx_logger.setLevel(logging.INFO)
        hub_logger.setLevel(logging.INFO)

        VJepa2Adapter(
            model_name="facebook/vjepa2-vitl-fpc64-256",
            device="cuda",
        )

        assert httpx_logger.level == logging.WARNING
        assert hub_logger.level == logging.WARNING
    finally:
        httpx_logger.setLevel(old_httpx_level)
        hub_logger.setLevel(old_hub_level)


def test_vjepa2_adapter_has_no_target_projector_or_ema_path():
    assert not hasattr(VJepa2Adapter, "apply_target_projector")
    assert not hasattr(VJepa2Adapter, "update_target_projector")
    assert not hasattr(VJepa2Adapter, "target_projector")


def test_world_model_future_slot_builder_selects_prefix_future_frames():
    adapter = _FakeAdapter()
    builder = world_model_pytorch.WorldModelFutureSlotBuilder(adapter=adapter, embed_dim=4, slot_max_len=8)
    with torch.no_grad():
        for idx in range(builder.slot_max_len):
            builder.slot_embed[0, idx, :].fill_(100.0 + idx)
            builder.horizon_embed.weight[idx, :].fill_(10.0 * idx)

    class _Observation:
        def __init__(self):
            self.images = {
                "base_0_rgb_history": torch.zeros((2, 4, 8, 8, 3), dtype=torch.float32),
                "base_0_rgb_future": torch.zeros((2, 3, 8, 8, 3), dtype=torch.float32),
            }
            self.image_masks = {
                "base_0_rgb_history_selection": torch.tensor(
                    [[False, False, True, True], [False, False, True, True]],
                    dtype=torch.bool,
                ),
                "base_0_rgb_future_selection": torch.tensor(
                    [[True, True, False], [True, True, False]],
                    dtype=torch.bool,
                ),
            }

    wm_batch = builder.build_training_inputs(_Observation())

    assert adapter.recorded_shapes == [
        (2, 2, 8, 8, 3),
        (2, 2, 8, 8, 3),
    ]
    assert wm_batch.wm_target.shape == (2, 2, 4)
    assert wm_batch.wm_target_mask.shape == (2, 2)
    assert wm_batch.future_token_loss_normalizer == 2
    expected_slots = torch.tensor(
        [
            [
                [100.0, 100.0, 100.0, 100.0],
                [111.0, 111.0, 111.0, 111.0],
            ],
            [
                [100.0, 100.0, 100.0, 100.0],
                [111.0, 111.0, 111.0, 111.0],
            ],
        ],
        dtype=torch.float32,
    )
    expected_target_mask = torch.tensor([[True, True], [True, True]], dtype=torch.bool)
    torch.testing.assert_close(wm_batch.wm_inputs[:, -2:, :], expected_slots)
    torch.testing.assert_close(wm_batch.wm_target_mask, expected_target_mask)


def test_world_model_future_slot_builder_keeps_raw_future_targets_with_input_projector():
    adapter = _ProjectedAdapter(raw_dim=4, projected_dim=6)
    builder = world_model_pytorch.WorldModelFutureSlotBuilder(adapter=adapter, embed_dim=6, slot_max_len=8)

    class _Observation:
        def __init__(self):
            self.images = {
                "base_0_rgb_history": torch.zeros((2, 4, 8, 8, 3), dtype=torch.float32),
                "base_0_rgb_future": torch.zeros((2, 4, 8, 8, 3), dtype=torch.float32),
            }
            self.image_masks = {
                "base_0_rgb_history": torch.ones((2, 4), dtype=torch.bool),
                "base_0_rgb_future": torch.ones((2, 4), dtype=torch.bool),
            }

    wm_batch = builder.build_training_inputs(_Observation())
    predictor = world_model_pytorch.WorldModelPredictorHead(input_dim=6, output_dim=4)
    wm_pred = predictor(wm_batch.wm_inputs[:, wm_batch.l_hist :, :])

    assert wm_batch.wm_inputs.shape[-1] == 6
    assert wm_batch.wm_target.shape[-1] == 4
    assert wm_pred.shape[-1] == 4


def test_world_model_future_slot_builder_detaches_targets():
    adapter = _GradAdapter()
    builder = world_model_pytorch.WorldModelFutureSlotBuilder(
        adapter=adapter,
        embed_dim=4,
        slot_max_len=8,
    )

    class _Observation:
        def __init__(self):
            self.images = {
                "base_0_rgb_history": torch.zeros((2, 4, 8, 8, 3), dtype=torch.float32),
                "base_0_rgb_future": torch.zeros((2, 4, 8, 8, 3), dtype=torch.float32),
            }
            self.image_masks = {
                "base_0_rgb_history": torch.ones((2, 4), dtype=torch.bool),
                "base_0_rgb_future": torch.ones((2, 4), dtype=torch.bool),
            }

    wm_batch = builder.build_training_inputs(_Observation())

    assert wm_batch.wm_inputs.requires_grad
    assert not wm_batch.wm_target.requires_grad


def test_world_model_future_slot_builder_builds_inference_inputs_without_future_gt_encoding():
    adapter = _FakeAdapter()
    builder = world_model_pytorch.WorldModelFutureSlotBuilder(adapter=adapter, embed_dim=4, slot_max_len=8)
    with torch.no_grad():
        for idx in range(builder.slot_max_len):
            builder.slot_embed[0, idx, :].fill_(100.0 + idx)
            builder.horizon_embed.weight[idx, :].fill_(10.0 * idx)

    class _Observation:
        def __init__(self):
            self.images = {
                "base_0_rgb_history": torch.zeros((2, 4, 8, 8, 3), dtype=torch.float32),
                # Future images may be present in offline evaluation batches, but inference must not encode them.
                "base_0_rgb_future": torch.full((2, 3, 8, 8, 3), 7.0, dtype=torch.float32),
            }
            self.image_masks = {
                "base_0_rgb_history_selection": torch.tensor(
                    [[False, False, True, True], [False, False, True, True]],
                    dtype=torch.bool,
                ),
                "base_0_rgb_future_selection": torch.tensor(
                    [[True, True, False], [True, True, False]],
                    dtype=torch.bool,
                ),
            }

    wm_embeddings = builder.build_inference_inputs(_Observation())

    assert adapter.recorded_shapes == [
        (2, 2, 8, 8, 3),
    ]
    assert wm_embeddings.embeddings.shape == (2, 4, 4)
    assert wm_embeddings.pad_mask.shape == (2, 4)
    assert wm_embeddings.att_mask.shape == (2, 4)
    assert wm_embeddings.read_mask.shape == (2, 4)
    expected_slots = torch.tensor(
        [
            [[100.0, 100.0, 100.0, 100.0], [111.0, 111.0, 111.0, 111.0]],
            [[100.0, 100.0, 100.0, 100.0], [111.0, 111.0, 111.0, 111.0]],
        ],
        dtype=torch.float32,
    )
    expected_att_mask = torch.tensor(
        [[True, False, True, False], [True, False, True, False]],
        dtype=torch.bool,
    )
    torch.testing.assert_close(wm_embeddings.embeddings[:, -2:, :], expected_slots)
    torch.testing.assert_close(wm_embeddings.att_mask, expected_att_mask)


def test_world_model_future_slot_builder_masks_per_sample_temporal_padding():
    adapter = _FakeAdapter()
    builder = world_model_pytorch.WorldModelFutureSlotBuilder(adapter=adapter, embed_dim=4, slot_max_len=8)

    class _Observation:
        def __init__(self):
            self.images = {
                "base_0_rgb_history": torch.zeros((2, 4, 8, 8, 3), dtype=torch.float32),
                "base_0_rgb_future": torch.zeros((2, 3, 8, 8, 3), dtype=torch.float32),
            }
            self.image_masks = {
                "base_0_rgb_history": torch.tensor(
                    [[False, False, True, True], [False, True, True, True]],
                    dtype=torch.bool,
                ),
                "base_0_rgb_future": torch.tensor(
                    [[True, False, False], [True, True, False]],
                    dtype=torch.bool,
                ),
            }

    wm_batch = builder.build_training_inputs(_Observation())

    assert adapter.recorded_shapes == [
        (2, 4, 8, 8, 3),
        (2, 3, 8, 8, 3),
    ]
    assert wm_batch.wm_target.shape == (2, 2, 4)
    torch.testing.assert_close(
        wm_batch.wm_pad_mask,
        torch.tensor(
            [
                [True, True, True, True, True, True],
                [True, True, True, True, True, True],
            ],
            dtype=torch.bool,
        ),
    )
    torch.testing.assert_close(
        wm_batch.wm_read_mask,
        torch.tensor(
            [
                [False, False, True, True, False, False],
                [False, False, True, True, True, True],
            ],
            dtype=torch.bool,
        ),
    )
    torch.testing.assert_close(
        wm_batch.wm_att_mask,
        torch.tensor(
            [
                [True, False, False, False, True, False],
                [True, False, False, False, True, False],
            ],
            dtype=torch.bool,
        ),
    )
    torch.testing.assert_close(
        wm_batch.wm_target_mask,
        torch.tensor(
            [[False, False], [True, True]],
            dtype=torch.bool,
        ),
    )
    assert wm_batch.future_token_loss_normalizer == 2


def test_world_model_future_slot_builder_accepts_short_history_with_zero_future_slots():
    adapter = _FakeAdapter()
    builder = world_model_pytorch.WorldModelFutureSlotBuilder(adapter=adapter, embed_dim=4, slot_max_len=8)

    class _Observation:
        def __init__(self):
            self.images = {
                "base_0_rgb_history": torch.zeros((1, 2, 8, 8, 3), dtype=torch.float32),
                "base_0_rgb_future": torch.zeros((1, 0, 8, 8, 3), dtype=torch.float32),
            }
            self.image_masks = {
                "base_0_rgb_history": torch.ones((1, 2), dtype=torch.bool),
                "base_0_rgb_future": torch.ones((1, 0), dtype=torch.bool),
            }

    wm_embeddings = builder.build_inference_inputs(_Observation())

    assert adapter.recorded_shapes == [
        (1, 2, 8, 8, 3),
    ]
    assert wm_embeddings.embeddings.shape == (1, 2, 4)
    torch.testing.assert_close(wm_embeddings.pad_mask, torch.ones((1, 2), dtype=torch.bool))
    torch.testing.assert_close(wm_embeddings.read_mask, torch.ones((1, 2), dtype=torch.bool))
