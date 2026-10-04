"""World Model (V-JEPA2) integration for PI0 policy."""

import logging
import pathlib
from typing import NamedTuple

from huggingface_hub import hf_hub_download
import torch
from torch import Tensor
from torch import nn
import torch._dynamo

from openpi.models import vjepa2 as _vjepa2

try:
    from transformers import AutoModel as _AutoModel
    from transformers.models.vjepa2 import VJEPA2VideoProcessor as _VJEPA2VideoProcessor
except ImportError:
    raise ImportError("Please install transformers>=4.30.0")


class WorldModelEmbeddings(NamedTuple):
    """Container for world model embeddings and related masks."""

    embeddings: Tensor  # (batch_size, num_tokens, embed_dim)
    pad_mask: Tensor  # (batch_size, num_tokens) - bool
    att_mask: Tensor  # (batch_size, num_tokens) - bool
    read_mask: Tensor | None = None  # (batch_size, num_tokens) - bool, controls suffix access to cached tokens
    history_token_len: int = 0


class WorldModelFutureSlotBatch(NamedTuple):
    """Training batch for future-slot world modeling."""

    wm_inputs: Tensor  # [B, L_hist + L_future, D] = [hist, slots]
    wm_pad_mask: Tensor  # [B, L_hist + L_future] bool
    wm_read_mask: Tensor  # [B, L_hist + L_future] bool
    wm_att_mask: Tensor  # [B, L_hist + L_future] bool (prefix-lm style mask_ar)
    wm_target: Tensor  # [B, L_future, D] absolute future embeddings
    wm_target_mask: Tensor  # [B, L_future] bool, token-level loss mask
    l_hist: int
    future_token_loss_normalizer: int  # original future token count before training-time sampling


class VJepa2Adapter(nn.Module):
    """Adapter for V-JEPA2 world model integration.

    By default the downstream world model operates directly on pretrained
    V-JEPA2 tokens without a trainable projection, so the encoder's hidden
    size must match the downstream world-model expert width.

    When ``enable_input_projector=True`` we instead build a single trainable
    projector that maps V-JEPA2's native hidden size to
    ``expected_embedding_dim`` on the *input* side only. History embeddings fed
    into the world-model expert live in the expert width, while future
    supervision stays in frozen raw V-JEPA space.

    Args:
        model_name: HF repo id or local path passed to ``AutoModel.from_pretrained``.
        device: CUDA device the encoder lives on.
        expected_embedding_dim: Width that the downstream world-model expert
            consumes. Without ``enable_input_projector`` this must equal the
            loaded V-JEPA2 hidden size; with ``enable_input_projector`` it is
            the projector output width.
        enable_input_projector: When ``True``, insert an input-side projector
            instead of requiring a strict hidden-size match.
        input_projector_hidden_dim: Optional MLP hidden width. When ``None``
            the projector is a single ``LayerNorm + Linear`` stack; otherwise
            it is ``LayerNorm -> Linear -> GELU -> Linear``.
        expected_vjepa_hidden_size: Optional expected hidden size resolved from
            the canonical V-JEPA2 metadata table. When provided, the loaded
            checkpoint must match it.
    """

    # Filename inside an HF V-JEPA2 repo holding the video processor config.
    DEFAULT_PROCESSOR_CONFIG_NAME = "video_preprocessor_config.json"
    # Data-pipeline keys (kept here as they are part of the dataset contract).
    WORLD_MODEL_IMAGE_KEY = "base_0_rgb_history"
    WORLD_MODEL_HISTORY_SELECTION_MASK_KEY = "base_0_rgb_history_selection"
    WORLD_MODEL_FUTURE_SELECTION_MASK_KEY = "base_0_rgb_future_selection"

    def __init__(
        self,
        model_name: str,
        device: str | torch.device = "cpu",
        expected_embedding_dim: int | None = None,
        enable_input_projector: bool = False,
        input_projector_hidden_dim: int | None = None,
        expected_vjepa_hidden_size: int | None = None,
    ):
        super().__init__()
        if not model_name:
            raise ValueError("VJepa2Adapter requires an explicit model_name.")
        self.model_name = model_name
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise RuntimeError(f"VJepa2Adapter requires CUDA device, got {self.device}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, but VJepa2Adapter requires CUDA")

        # Root logging is INFO during training; keep third-party HTTP chatter out of the logs.
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

        # Load V-JEPA2 model and processor
        try:
            self._vjepa2_model = _AutoModel.from_pretrained(self.model_name).to(self.device)
            model_path = pathlib.Path(self.model_name).expanduser()
            if model_path.exists():
                processor_config_path = model_path / self.DEFAULT_PROCESSOR_CONFIG_NAME
                if not processor_config_path.is_file():
                    raise FileNotFoundError(
                        f"Missing {self.DEFAULT_PROCESSOR_CONFIG_NAME!r} in local V-JEPA2 path: {model_path}"
                    )
                processor_config_path = str(processor_config_path)
            else:
                processor_config_path = hf_hub_download(
                    repo_id=self.model_name,
                    filename=self.DEFAULT_PROCESSOR_CONFIG_NAME,
                )

            self._vjepa2_processor = _VJEPA2VideoProcessor.from_pretrained(processor_config_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load V-JEPA2 model '{self.model_name}'. "
                f"Please ensure the model is available on HuggingFace. Error: {e}"
            ) from e

        self.verify_pretrained_loaded()

        # Hidden size is read from the loaded checkpoint instead of being hardcoded,
        # so swapping in a larger V-JEPA2 variant just works.
        self.embedding_dim = int(self._vjepa2_model.config.hidden_size)
        if expected_vjepa_hidden_size is not None and self.embedding_dim != int(expected_vjepa_hidden_size):
            raise ValueError(
                "Loaded V-JEPA2 hidden size does not match canonical metadata: "
                f"expected {int(expected_vjepa_hidden_size)}, got {self.embedding_dim}."
            )
        self.enable_input_projector = bool(enable_input_projector)

        if self.enable_input_projector:
            if expected_embedding_dim is None:
                raise ValueError(
                    "enable_input_projector=True requires an explicit expected_embedding_dim "
                    "(the downstream world-model expert width)."
                )
            self.output_dim = int(expected_embedding_dim)
            self.input_projector = self._build_projector(
                in_dim=self.embedding_dim,
                out_dim=self.output_dim,
                hidden_dim=input_projector_hidden_dim,
            ).to(self.device)
            logging.info(
                "VJepa2Adapter: enabled input projector %d -> %d (mlp_hidden=%s)",
                self.embedding_dim,
                self.output_dim,
                input_projector_hidden_dim,
            )
        else:
            if expected_embedding_dim is not None and expected_embedding_dim != self.embedding_dim:
                raise ValueError(
                    f"V-JEPA2 hidden size ({self.embedding_dim}) does not match the expected "
                    f"width ({expected_embedding_dim}). Either enable an input projector "
                    "(`vjepa2_enable_input_projector=True`) or pick a matching "
                    "`world_model_expert_variant` / `vjepa2_variant` pair."
                )
            self.output_dim = self.embedding_dim
            self.input_projector = None

        logging.info(
            "Initialized VJepa2Adapter: %s (hidden_size=%d, output_dim=%d, input_projector=%s)",
            self.model_name,
            self.embedding_dim,
            self.output_dim,
            self.enable_input_projector,
        )

    @staticmethod
    def _build_projector(in_dim: int, out_dim: int, hidden_dim: int | None) -> nn.Module:
        if hidden_dim is None or int(hidden_dim) <= 0:
            return nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, out_dim),
            )
        hidden_dim = int(hidden_dim)
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def apply_input_projector(self, tokens: Tensor) -> Tensor:
        """Project raw V-JEPA2 tokens to the world-model expert width.

        Returns ``tokens`` unchanged when no projector is configured.
        """
        if self.input_projector is None:
            return tokens
        return self.input_projector(tokens.to(dtype=next(self.input_projector.parameters()).dtype))

    @property
    def encoder_module(self) -> nn.Module:
        """Frozen encoder module backing the adapter.

        Exposed so that downstream training code can freeze or unfreeze the encoder.
        """
        return self._vjepa2_model

    def verify_pretrained_loaded(self) -> None:
        loaded_from = getattr(self._vjepa2_model.config, "_name_or_path", None)
        if loaded_from is None:
            raise RuntimeError("V-JEPA2 model has no _name_or_path; pretrained load may have failed.")
        n_params = sum(p.numel() for p in self._vjepa2_model.parameters())
        if n_params == 0:
            raise RuntimeError("V-JEPA2 model has zero parameters.")
        logging.info(f"V-JEPA2 pretrained loaded from: {loaded_from}, params={n_params:,}")

    def freeze(self) -> None:
        """Freeze V-JEPA2 model parameters."""
        for param in self._vjepa2_model.parameters():
            param.requires_grad = False
        logging.info("V-JEPA2 model frozen")

    def unfreeze(self) -> None:
        """Unfreeze V-JEPA2 model parameters."""
        for param in self._vjepa2_model.parameters():
            param.requires_grad = True
        logging.info("V-JEPA2 model unfrozen")

    @staticmethod
    def _prepare_videos_for_processor(
        video_frames: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> list[torch.Tensor]:
        """Convert batched video tensors into the per-video 4D format expected by transformers.

        HuggingFace video processors accept either a single 4D video tensor or a list of 4D
        videos. Passing a raw 5D batched tensor causes `make_batched_videos` to wrap it into
        a single 6D item, which then fails channel-dimension inference.
        """
        if isinstance(video_frames, (list, tuple)):
            if len(video_frames) == 0:
                raise ValueError("video_frames must not be empty")

            videos = [torch.as_tensor(video) for video in video_frames]
        else:
            tensor = torch.as_tensor(video_frames)
            if tensor.ndim == 5:
                if tensor.shape[0] == 0:
                    raise ValueError(
                        f"video_frames batch dimension must be > 0, got shape {tuple(tensor.shape)}"
                    )
                videos = list(torch.unbind(tensor, dim=0))
            elif tensor.ndim == 4:
                videos = [tensor]
            else:
                raise ValueError(
                    "video_frames must be a 4D video tensor, a 5D batched video tensor, "
                    f"or a non-empty list/tuple of 4D videos, got shape {tuple(tensor.shape)}"
                )

        for idx, video in enumerate(videos):
            if video.ndim != 4:
                raise ValueError(
                    f"Expected video {idx} to be 4D after batching, got shape {tuple(video.shape)}"
                )

        return videos

    @torch._dynamo.disable
    def _process_video_frames(self, video_frames: torch.Tensor) -> tuple[dict, torch.device]:
        """Process video frames with transformers processor.

        This method is excluded from torch.compile to avoid CUDA Graph tensor rewriting issues.
        """
        processor_videos = self._prepare_videos_for_processor(video_frames)
        pixel_values = self._vjepa2_processor(processor_videos, return_tensors="pt")
        device = next(self._vjepa2_model.parameters()).device

        # Move to device and clone to break connection with processor internals
        processed_inputs = {}
        for k, v in pixel_values.items():
            if isinstance(v, torch.Tensor):
                processed_inputs[k] = v.to(device).clone().detach()
            else:
                processed_inputs[k] = v

        return processed_inputs, device

    @property
    def tubelet_size(self) -> int:
        return int(self._vjepa2_model.config.tubelet_size)

    @property
    def spatial_tokens_per_temporal_bin(self) -> int:
        crop_size = int(self._vjepa2_model.config.crop_size)
        patch_size = int(self._vjepa2_model.config.patch_size)
        return (crop_size // patch_size) * (crop_size // patch_size)

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

    def frame_mask_to_temporal_mask(self, frame_mask: Tensor, *, num_frames: int | None = None) -> Tensor:
        frame_mask = torch.as_tensor(frame_mask, dtype=torch.bool)
        if frame_mask.ndim == 1:
            frame_mask = frame_mask[None, :]
        if frame_mask.ndim != 2:
            raise ValueError(f"Expected frame mask to be 1D or 2D, got shape {tuple(frame_mask.shape)}")

        total_frames = int(frame_mask.shape[1]) if num_frames is None else int(num_frames)
        if frame_mask.shape[1] != total_frames:
            raise ValueError(
                f"Expected frame mask with {total_frames} entries, got shape {tuple(frame_mask.shape)}."
            )

        temporal_bins = self.temporal_bins_for_num_frames(total_frames)
        if temporal_bins == 0:
            return torch.zeros((frame_mask.shape[0], 0), dtype=torch.bool, device=frame_mask.device)

        if total_frames < self.tubelet_size:
            return frame_mask[:, :1]

        consumed_frames = temporal_bins * self.tubelet_size
        return frame_mask[:, :consumed_frames].reshape(frame_mask.shape[0], temporal_bins, self.tubelet_size).all(dim=2)

    def expand_frame_mask_to_token_mask(
        self,
        frame_mask: Tensor,
        *,
        num_frames: int | None = None,
        token_count: int | None = None,
    ) -> Tensor:
        temporal_mask = self.frame_mask_to_temporal_mask(frame_mask, num_frames=num_frames)
        token_mask = temporal_mask.repeat_interleave(self.spatial_tokens_per_temporal_bin, dim=1)
        if token_count is not None and token_mask.shape[1] != token_count:
            raise ValueError(
                f"Expected expanded token mask with {token_count} entries, got {token_mask.shape[1]}."
            )
        return token_mask

    def forward(
        self,
        video_frames: torch.Tensor,
        skip_predictor: bool = True,
    ) -> WorldModelEmbeddings:
        """Process video frames through V-JEPA2 and return embeddings.

        Args:
            video_frames: Batched video frames ``(B, T, H, W, C)``. Accepts either
                float-valued tensors normalized to ``[-1, 1]`` (the convention used by
                the rest of the policy pipeline) or pre-converted ``uint8`` tensors in
                ``[0, 255]`` (legacy callers). Floats are converted to ``uint8`` here so
                callers do not need to know about the V-JEPA HuggingFace processor's
                preferred input format.
            skip_predictor: Whether to skip V-JEPA2 predictor head (use encoder only)

        Returns:
            WorldModelEmbeddings: Contains embeddings, pad_mask, and att_mask

        Raises:
            ValueError: If video_frames is None or invalid shape
        """
        if video_frames is None:
            raise ValueError("video_frames cannot be None")
        if video_frames.ndim != 5:
            raise ValueError(f"Expected batched 5D video tensor, got shape {tuple(video_frames.shape)}")

        # The downstream HuggingFace VideoProcessor expects uint8 RGB in [0, 255].
        # The rest of the policy pipeline carries images as float in [-1, 1], so we
        # convert here instead of forcing every caller to do it.
        if video_frames.dtype != torch.uint8:
            video_frames = ((video_frames + 1.0) / 2.0 * 255.0).clamp(0.0, 255.0).to(torch.uint8)

        # Process video through V-JEPA2 processor
        pixel_values, device = self._process_video_frames(video_frames)

        # Get embeddings from V-JEPA2
        with torch.no_grad():
            outputs = self._vjepa2_model(**pixel_values, skip_predictor=skip_predictor)
        video_embeddings = outputs.last_hidden_state

        # Create attention masks
        batch_size, num_tokens = video_embeddings.shape[:2]
        pad_mask = torch.ones(
            batch_size,
            num_tokens,
            dtype=torch.bool,
            device=device,
        )
        att_mask = torch.zeros(
            batch_size,
            num_tokens,
            dtype=torch.bool,
            device=device,
        )

        return WorldModelEmbeddings(
            embeddings=video_embeddings,
            pad_mask=pad_mask,
            att_mask=att_mask,
            read_mask=pad_mask,
            history_token_len=0,
        )


class WorldModelFutureSlotBuilder(nn.Module):
    """Builds world-model training inputs: [history, future_slots]."""

    def __init__(
        self,
        adapter: VJepa2Adapter,
        embed_dim: int,
        slot_max_len: int = 512,
    ):
        super().__init__()
        self.adapter = adapter
        self.slot_max_len = slot_max_len
        self.slot_embed = nn.Parameter(torch.randn(1, slot_max_len, embed_dim) * 0.02)
        self.horizon_embed = nn.Embedding(slot_max_len, embed_dim)

    def _build_slots(self, batch_size: int, slot_len: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        if slot_len > self.slot_max_len:
            raise ValueError(f"future slot_len ({slot_len}) exceeds slot_max_len ({self.slot_max_len}).")
        base = self.slot_embed[:, :slot_len, :].expand(batch_size, -1, -1)
        pos = torch.arange(slot_len, device=device, dtype=torch.long)[None, :].expand(batch_size, -1)

        return (base + self.horizon_embed(pos)).to(dtype=dtype)

    def _get_temporal_selection_mask(
        self,
        observation,
        images: Tensor,
        *,
        selection_key: str,
        image_key: str,
    ) -> Tensor | None:
        selection_mask = getattr(observation, "image_masks", {}).get(selection_key)
        if selection_mask is None:
            return None

        selection_mask = torch.as_tensor(selection_mask, dtype=torch.bool, device=images.device)
        if selection_mask.ndim == 2:
            if selection_mask.shape[0] != images.shape[0]:
                raise ValueError(
                    f"Expected temporal selection mask batch dimension {images.shape[0]} for key '{image_key}', "
                    f"got {selection_mask.shape[0]}."
                )
            if not torch.equal(selection_mask, selection_mask[:1].expand_as(selection_mask)):
                raise ValueError(
                    f"Temporal selection masks must be identical across the batch for key '{image_key}'."
                )
            selection_mask = selection_mask[0]
        elif selection_mask.ndim != 1:
            raise ValueError(
                f"Expected temporal selection mask to be 1D or 2D for key '{image_key}', got shape {tuple(selection_mask.shape)}."
            )

        if selection_mask.shape[0] != images.shape[1]:
            raise ValueError(
                f"Expected temporal selection mask with {images.shape[1]} entries for key '{image_key}', "
                f"got {selection_mask.shape[0]}."
            )
        if not torch.any(selection_mask):
            raise ValueError(f"Temporal selection mask for key '{image_key}' selects no frames.")

        return selection_mask

    def _normalize_temporal_selection_mask(
        self,
        selection_mask,
        *,
        device: torch.device,
        image_key: str,
        batch_size: int | None = None,
        total_count: int | None = None,
    ) -> Tensor:
        selection_mask = torch.as_tensor(selection_mask, dtype=torch.bool, device=device)
        if selection_mask.ndim == 2:
            if batch_size is not None and selection_mask.shape[0] != batch_size:
                raise ValueError(
                    f"Expected temporal selection mask batch dimension {batch_size} for key '{image_key}', "
                    f"got {selection_mask.shape[0]}."
                )
            if not torch.equal(selection_mask, selection_mask[:1].expand_as(selection_mask)):
                raise ValueError(
                    f"Temporal selection masks must be identical across the batch for key '{image_key}'."
                )
            selection_mask = selection_mask[0]
        elif selection_mask.ndim != 1:
            raise ValueError(
                f"Expected temporal selection mask to be 1D or 2D for key '{image_key}', got shape {tuple(selection_mask.shape)}."
            )

        if total_count is not None and selection_mask.shape[0] != total_count:
            raise ValueError(
                f"Expected temporal selection mask with {total_count} entries for key '{image_key}', "
                f"got {selection_mask.shape[0]}."
            )
        if not torch.any(selection_mask):
            raise ValueError(f"Temporal selection mask for key '{image_key}' selects no frames.")

        return selection_mask

    def _normalize_temporal_valid_mask(
        self,
        valid_mask,
        *,
        device: torch.device,
        image_key: str,
        batch_size: int,
        total_count: int,
    ) -> Tensor:
        valid_mask = torch.as_tensor(valid_mask, dtype=torch.bool, device=device)
        if valid_mask.ndim == 1:
            if valid_mask.shape[0] != total_count:
                raise ValueError(
                    f"Expected temporal mask with {total_count} entries for key '{image_key}', "
                    f"got {valid_mask.shape[0]}."
                )
            return valid_mask[None, :].expand(batch_size, -1)
        if valid_mask.ndim != 2:
            raise ValueError(
                f"Expected temporal mask to be 1D or 2D for key '{image_key}', got shape {tuple(valid_mask.shape)}."
            )
        if valid_mask.shape != (batch_size, total_count):
            raise ValueError(
                f"Expected temporal mask with shape ({batch_size}, {total_count}) for key '{image_key}', "
                f"got {tuple(valid_mask.shape)}."
            )
        return valid_mask

    def _select_temporal_frames(
        self,
        images: Tensor,
        *,
        selection_mask: Tensor | None,
    ) -> Tensor:
        if selection_mask is None:
            return images
        return images[:, selection_mask, ...]

    def _resolve_inference_future_slot_len(
        self,
        observation,
        history_images: Tensor,
        future_num_frames: int | None,
    ) -> int:
        if future_num_frames is not None:
            resolved = int(future_num_frames)
            if resolved < 0:
                raise ValueError(f"future_num_frames must be >= 0, got {resolved}")
            return resolved

        future_selection = getattr(observation, "image_masks", {}).get(VJepa2Adapter.WORLD_MODEL_FUTURE_SELECTION_MASK_KEY)
        if future_selection is not None:
            selection_mask = self._normalize_temporal_selection_mask(
                future_selection,
                device=history_images.device,
                image_key="base_0_rgb_future",
                batch_size=history_images.shape[0],
            )
            return int(selection_mask.sum().item())

        future_images = observation.images.get("base_0_rgb_future")
        if future_images is not None and getattr(future_images, "ndim", 0) >= 2:
            return int(future_images.shape[1])

        future_mask = getattr(observation, "image_masks", {}).get("base_0_rgb_future")
        if future_mask is not None:
            future_mask = torch.as_tensor(future_mask, dtype=torch.bool, device=history_images.device)
            if future_mask.ndim == 2:
                if future_mask.shape[0] != history_images.shape[0]:
                    raise ValueError(
                        f"Expected temporal future mask batch dimension {history_images.shape[0]}, got {future_mask.shape[0]}."
                    )
                return int(future_mask.shape[1])
            if future_mask.ndim == 1:
                return int(future_mask.shape[0])
            raise ValueError(f"Expected base_0_rgb_future mask to be 1D or 2D, got {tuple(future_mask.shape)}.")

        raise ValueError(
            "Cannot infer inference future slot count. Provide future_num_frames, "
            "base_0_rgb_future_selection, or base_0_rgb_future metadata."
        )

    def build_training_inputs(self, observation) -> WorldModelFutureSlotBatch:
        history_key = VJepa2Adapter.WORLD_MODEL_IMAGE_KEY
        future_key = history_key.replace("_history", "_future")

        images_history = observation.images.get(history_key)
        images_future = observation.images.get(future_key)

        if images_history is None:
            raise ValueError(f"Missing required world-model history key: '{history_key}'")
        if images_future is None:
            raise ValueError(f"Missing required world-model future key: '{future_key}'")

        # Adapters consume normalized ``[-1, 1]`` floats directly and apply
        # whatever encoder-specific preprocessing they require internally.
        future_frame_count = int(images_future.shape[1])

        history_selection_mask = self._get_temporal_selection_mask(
            observation,
            images_history,
            selection_key=VJepa2Adapter.WORLD_MODEL_HISTORY_SELECTION_MASK_KEY,
            image_key=history_key,
        )
        history_valid_mask = self._normalize_temporal_valid_mask(
            getattr(observation, "image_masks", {}).get(history_key, torch.ones(images_history.shape[:2])),
            device=images_history.device,
            image_key=history_key,
            batch_size=images_history.shape[0],
            total_count=images_history.shape[1],
        )
        future_selection_mask = self._get_temporal_selection_mask(
            observation,
            images_future,
            selection_key=VJepa2Adapter.WORLD_MODEL_FUTURE_SELECTION_MASK_KEY,
            image_key=future_key,
        )
        future_valid_mask = self._normalize_temporal_valid_mask(
            getattr(observation, "image_masks", {}).get(future_key, torch.ones(images_future.shape[:2])),
            device=images_future.device,
            image_key=future_key,
            batch_size=images_future.shape[0],
            total_count=images_future.shape[1],
        )
        if history_selection_mask is not None:
            history_valid_mask = history_valid_mask[:, history_selection_mask]
        images_history = self._select_temporal_frames(
            images_history,
            selection_mask=history_selection_mask,
        )
        if future_selection_mask is not None:
            future_valid_mask = future_valid_mask[:, future_selection_mask]
        images_future = self._select_temporal_frames(
            images_future,
            selection_mask=future_selection_mask,
        )

        hist = self.adapter(images_history)
        fut = self.adapter(images_future)

        hist_input_embeddings = self.adapter.apply_input_projector(hist.embeddings)
        fut_target_embeddings = fut.embeddings

        bsz, l_hist, _ = hist_input_embeddings.shape
        l_future = fut_target_embeddings.shape[1]

        slots = self._build_slots(
            batch_size=bsz,
            slot_len=l_future,
            device=hist_input_embeddings.device,
            dtype=hist_input_embeddings.dtype,
        )

        wm_inputs = torch.cat([hist_input_embeddings, slots], dim=1)
        hist_read_mask = self.adapter.expand_frame_mask_to_token_mask(
            history_valid_mask,
            num_frames=images_history.shape[1],
            token_count=l_hist,
        ).to(device=hist.pad_mask.device)
        fut_read_mask = self.adapter.expand_frame_mask_to_token_mask(
            future_valid_mask,
            num_frames=images_future.shape[1],
            token_count=l_future,
        ).to(device=fut.pad_mask.device)
        wm_pad_mask = torch.cat([hist.pad_mask, fut.pad_mask], dim=1)
        wm_read_mask = torch.cat([hist_read_mask, fut_read_mask], dim=1)

        # Prefix-LM style mask_ar:
        # hist: 0...0, slots: 1,0,0,... (same block mechanism as the action suffix)
        hist_att = torch.zeros_like(hist.pad_mask, dtype=torch.bool)
        slot_att = torch.zeros_like(fut.pad_mask, dtype=torch.bool)
        if l_hist > 0:
            hist_att[:, 0] = True
        if l_future > 0:
            slot_att[:, 0] = True

        wm_att_mask = torch.cat([hist_att, slot_att], dim=1)
        # Detach the absolute future target so the loss cannot collapse the encoder.
        wm_target = fut_target_embeddings.detach()

        return WorldModelFutureSlotBatch(
            wm_inputs=wm_inputs,
            wm_pad_mask=wm_pad_mask,
            wm_read_mask=wm_read_mask,
            wm_att_mask=wm_att_mask,
            wm_target=wm_target,
            wm_target_mask=fut_read_mask,
            l_hist=l_hist,
            future_token_loss_normalizer=self.adapter.token_count_for_num_frames(future_frame_count),
        )

    def build_prefix_memory(self, observation) -> WorldModelEmbeddings:
        history_key = VJepa2Adapter.WORLD_MODEL_IMAGE_KEY
        images_history = observation.images.get(history_key)
        if images_history is None:
            raise ValueError(f"Missing required world-model history key: '{history_key}'")
        history_selection_mask = self._get_temporal_selection_mask(
            observation,
            images_history,
            selection_key=VJepa2Adapter.WORLD_MODEL_HISTORY_SELECTION_MASK_KEY,
            image_key=history_key,
        )
        history_valid_mask = self._normalize_temporal_valid_mask(
            getattr(observation, "image_masks", {}).get(history_key, torch.ones(images_history.shape[:2])),
            device=images_history.device,
            image_key=history_key,
            batch_size=images_history.shape[0],
            total_count=images_history.shape[1],
        )
        if history_selection_mask is not None:
            history_valid_mask = history_valid_mask[:, history_selection_mask]
        images_history = self._select_temporal_frames(
            images_history,
            selection_mask=history_selection_mask,
        )
        hist = self.adapter(images_history)
        hist_input_embeddings = self.adapter.apply_input_projector(hist.embeddings)
        return WorldModelEmbeddings(
            embeddings=hist_input_embeddings,
            pad_mask=hist.pad_mask,
            att_mask=hist.att_mask,
            read_mask=self.adapter.expand_frame_mask_to_token_mask(
                history_valid_mask,
                num_frames=images_history.shape[1],
                token_count=hist_input_embeddings.shape[1],
            ).to(device=hist.pad_mask.device),
            history_token_len=hist_input_embeddings.shape[1],
        )

    def build_inference_inputs(
        self,
        observation,
        *,
        future_num_frames: int | None = None,
    ) -> WorldModelEmbeddings:
        history_key = VJepa2Adapter.WORLD_MODEL_IMAGE_KEY
        images_history = observation.images.get(history_key)
        if images_history is None:
            raise ValueError(f"Missing required world-model history key: '{history_key}'")

        history_selection_mask = self._get_temporal_selection_mask(
            observation,
            images_history,
            selection_key=VJepa2Adapter.WORLD_MODEL_HISTORY_SELECTION_MASK_KEY,
            image_key=history_key,
        )
        history_valid_mask = self._normalize_temporal_valid_mask(
            getattr(observation, "image_masks", {}).get(history_key, torch.ones(images_history.shape[:2])),
            device=images_history.device,
            image_key=history_key,
            batch_size=images_history.shape[0],
            total_count=images_history.shape[1],
        )
        if history_selection_mask is not None:
            history_valid_mask = history_valid_mask[:, history_selection_mask]
        images_history = self._select_temporal_frames(
            images_history,
            selection_mask=history_selection_mask,
        )

        hist = self.adapter(images_history)
        hist_input_embeddings = self.adapter.apply_input_projector(hist.embeddings)
        bsz, l_hist, _ = hist_input_embeddings.shape
        l_future = self._resolve_inference_future_slot_len(
            observation,
            images_history,
            future_num_frames,
        )
        slot_token_len = self.adapter.token_count_for_num_frames(l_future)

        slots = self._build_slots(
            batch_size=bsz,
            slot_len=slot_token_len,
            device=hist_input_embeddings.device,
            dtype=hist_input_embeddings.dtype,
        )

        wm_inputs = torch.cat([hist_input_embeddings, slots], dim=1)
        slot_pad_mask = torch.ones((bsz, slot_token_len), dtype=torch.bool, device=hist.pad_mask.device)
        wm_pad_mask = torch.cat([hist.pad_mask, slot_pad_mask], dim=1)
        hist_read_mask = self.adapter.expand_frame_mask_to_token_mask(
            history_valid_mask,
            num_frames=images_history.shape[1],
            token_count=l_hist,
        ).to(device=hist.pad_mask.device)
        wm_read_mask = torch.cat([hist_read_mask, slot_pad_mask], dim=1)

        hist_att = torch.zeros_like(hist.pad_mask, dtype=torch.bool)
        slot_att = torch.zeros_like(slot_pad_mask, dtype=torch.bool)
        if l_hist > 0:
            hist_att[:, 0] = True
        if l_future > 0:
            slot_att[:, 0] = True
        wm_att_mask = torch.cat([hist_att, slot_att], dim=1)

        return WorldModelEmbeddings(
            embeddings=wm_inputs,
            pad_mask=wm_pad_mask,
            att_mask=wm_att_mask,
            read_mask=wm_read_mask,
            history_token_len=l_hist,
        )


class WorldModelPredictorHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        # Defensive dtype alignment: the world-model expert's final output is
        # fp32 (its final norm stays fp32), while ``self.norm`` follows the
        # model precision (bf16 during training). ``nn.LayerNorm`` requires
        # input and weight to share a dtype, so align to the norm's dtype
        # before computing.
        if x.dtype != self.norm.weight.dtype:
            x = x.to(dtype=self.norm.weight.dtype)
        return self.proj(self.norm(x))


class WorldModelConfig:
    """Configuration for the frozen V-JEPA2 encoder used by the world model."""

    def __init__(
        self,
        *,
        expected_embedding_dim: int,
        device: str | torch.device = "cpu",
        variant: _vjepa2.Variant = "vitl-256",
        model_name_override: str | None = None,
        enable_input_projector: bool = False,
        input_projector_hidden_dim: int | None = None,
    ):
        self.expected_embedding_dim = int(expected_embedding_dim)
        self.device = device
        self.variant = variant
        self.model_name_override = model_name_override
        self.enable_input_projector = bool(enable_input_projector)
        self.input_projector_hidden_dim = input_projector_hidden_dim

    def build_adapter(self) -> VJepa2Adapter:
        model_name = _vjepa2.resolve_repo_id(self.variant, self.model_name_override)
        expected_vjepa_hidden_size = None
        if self.model_name_override is None:
            expected_vjepa_hidden_size = _vjepa2.resolve_hidden_size(self.variant)
        return VJepa2Adapter(
            model_name=model_name,
            device=self.device,
            expected_embedding_dim=self.expected_embedding_dim,
            enable_input_projector=self.enable_input_projector,
            input_projector_hidden_dim=self.input_projector_hidden_dim,
            expected_vjepa_hidden_size=expected_vjepa_hidden_size,
        )
