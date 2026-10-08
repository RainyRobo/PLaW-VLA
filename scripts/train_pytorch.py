# Derived from openpi (Copyright 2024 Physical Intelligence, Inc.; Apache-2.0).
# Modified for PLaW-VLA by the PLaW-VLA authors, 2026.
"""Train PLaW-VLA with PyTorch on one GPU or with distributed data parallelism.

Training and data settings come from ``openpi.training.config``. The stage
wrappers prepare the required assets; direct invocations expect them to exist.

Run the small synthetic-data configuration on one GPU::

    uv run scripts/train_pytorch.py debug_pi05 --exp-name example

Resume that run with overwriting disabled::

    uv run scripts/train_pytorch.py debug_pi05 --exp-name example --resume --no-overwrite

For multiple GPUs on one node, launch one process per visible GPU::

    uv run torchrun --standalone --nnodes=1 --nproc-per-node=<num_gpus> \
        scripts/train_pytorch.py <config_name> --exp-name <run_name>

For multiple nodes, use the same master address and port on every node, with a
distinct ``--node-rank`` for each::

    uv run torchrun --nnodes=<num_nodes> --nproc-per-node=<gpus_per_node> \
        --node-rank=<node_rank> --master-addr=<master_ip> --master-port=<port> \
        scripts/train_pytorch.py <config_name> --exp-name <run_name>
"""

import copy
import dataclasses
import gc
import logging
import os
import pathlib
import platform
import random
import shutil
import threading
import time
from typing import Any

# The PyTorch trainer only uses JAX for tree/data utilities.  Keeping JAX on the
# CPU prevents every DataLoader worker from probing CUDA, ROCm, and TPU backends
# (and, more importantly, from competing with PyTorch for GPU memory).
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm
import wandb

import openpi.models.model as _model
import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data


def init_logging(logging_level=logging.INFO):
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging_level)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def _use_offline_wandb_without_credentials() -> None:
    """Keep an unconfigured run on disk instead of failing at wandb.init."""
    if os.environ.get("WANDB_MODE") or os.environ.get("WANDB_API_KEY"):
        return
    netrc = pathlib.Path.home() / ".netrc"
    try:
        has_wandb_login = "api.wandb.ai" in netrc.read_text()
    except OSError:
        has_wandb_login = False
    if has_wandb_login:
        return
    logging.warning(
        "No Weights & Biases credentials found. Logging this run offline under wandb/. "
        "Run `wandb login` before the next run to sync it."
    )
    os.environ["WANDB_MODE"] = "offline"


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    """Initialize wandb logging."""
    if not enabled:
        wandb.init(mode="disabled")
        return
    _use_offline_wandb_without_credentials()

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")

        # Set up debugging environment variables for DDP issues
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device


def cleanup_ddp():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def _log_model_init_progress(stop_event: threading.Event, interval_seconds: float = 15.0) -> None:
    """Emit a heartbeat while model construction or HF downloads are in progress."""
    started_at = time.monotonic()
    hf_home = pathlib.Path(os.environ.get("HF_HOME", pathlib.Path.home() / ".cache/huggingface"))
    hub_cache = pathlib.Path(os.environ.get("HF_HUB_CACHE", hf_home / "hub"))

    while not stop_event.wait(interval_seconds):
        incomplete_files = list(hub_cache.glob("**/*.incomplete")) if hub_cache.exists() else []
        downloaded_bytes = sum(path.stat().st_size for path in incomplete_files if path.is_file())
        logging.info(
            "Still initializing model: elapsed=%.0fs, active HF downloads=%d, downloaded=%.1f MiB",
            time.monotonic() - started_at,
            len(incomplete_files),
            downloaded_bytes / 2**20,
        )


def set_seed(seed: int, rank: int):
    torch.manual_seed(seed + rank)
    np.random.seed((seed + rank) % 2**32)
    random.seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


def build_datasets(config: _config.TrainConfig, *, checkpoint_assets: pathlib.Path | None = None):
    if checkpoint_assets is not None:
        # A multi-dataset factory forwards these assets to every child factory.
        # Override even explicitly configured live asset roots on resume.
        factory = dataclasses.replace(
            config.data,
            assets=dataclasses.replace(config.data.assets, assets_dir=str(checkpoint_assets)),
        )
        config = dataclasses.replace(config, data=factory)
        logging.info("Resuming with original normalization assets from %s", checkpoint_assets)
    # Use the unified data loader with PyTorch framework
    return _data.create_data_loader(config, framework="pytorch", shuffle=True)


def _build_log_payload(
    infos: list[dict[str, float]],
    *,
    global_step: int,
    elapsed: float,
    log_interval: int,
) -> dict[str, float]:
    avg_loss = sum(info["loss"] for info in infos) / len(infos)
    avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)

    log_payload: dict[str, float] = {
        "loss": avg_loss,
        "learning_rate": avg_lr,
        "step": global_step,
        "time_per_step": elapsed / log_interval,
    }

    grad_norm_vals = [info["grad_norm"] for info in infos if "grad_norm" in info and info["grad_norm"] is not None]
    if grad_norm_vals:
        log_payload["grad_norm"] = sum(grad_norm_vals) / len(grad_norm_vals)

    base_keys = {"loss", "learning_rate", "grad_norm"}
    extra_keys = set().union(*(info.keys() for info in infos)) - base_keys

    for k in sorted(extra_keys):
        vals = [info[k] for info in infos if k in info and isinstance(info[k], (int, float))]
        if not vals:
            continue
        log_payload[k] = sum(vals) / len(vals)

    return log_payload


def get_model_state_dict(model):
    """Get state dict from model, handling DDP wrapper."""
    return (
        model.module.state_dict()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.state_dict()
    )


def get_model_parameters(model):
    """Get parameters from model, handling DDP wrapper."""
    return (
        model.module.parameters()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.parameters()
    )


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None,
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("This training checkpoint requires CUDA to restore its random state.")
        torch.cuda.set_rng_state(state["cuda"])


def _resume_config_signature(config: _config.TrainConfig) -> dict[str, Any]:
    # The target step and logging/checkpoint frequency may change on resume.
    # Architecture, objective and update rules must still describe the same run.
    return {
        "model": dataclasses.asdict(config.model),
        "training_stage": config.training_stage,
        "precision": config.pytorch_training_precision,
        "optimizer": dataclasses.asdict(config.optimizer),
        "lr_schedule": dataclasses.asdict(config.lr_schedule),
    }


def save_checkpoint(model, optimizer, global_step, config, is_main, data_loader):
    """Save a checkpoint with model state, optimizer state, and metadata."""
    # Only save if it's time to save or if it's the final step. Every rank must
    # evaluate the same condition and wait together; otherwise the next step's
    # dropout broadcast starts while rank 0 is still writing the checkpoint.
    should_save = (global_step % config.save_interval == 0 and global_step > 0) or (
        global_step == config.num_train_steps
    )
    if not should_save:
        return
    distributed = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if distributed else 1
    rank_state = {
        "rank": dist.get_rank() if distributed else 0,
        "rng": _capture_rng_state(),
        "data_loader": data_loader.state_dict(),
    }
    rank_states = [None] * world_size if is_main else None
    if distributed:
        dist.gather_object(rank_state, rank_states, dst=0)
    else:
        rank_states = [rank_state]
    if is_main and should_save:
        # Create temporary directory for atomic checkpoint saving
        final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
        tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"

        # Remove any existing temp directory and create new one
        if tmp_ckpt_dir.exists():
            shutil.rmtree(tmp_ckpt_dir)
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model state using safetensors (handle shared tensors)
        model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")

        # Save optimizer state using PyTorch format
        torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")

        # Save training metadata (avoid saving full config to prevent JAX/Flax compatibility issues)
        metadata = {
            "global_step": global_step,
            "config": dataclasses.asdict(config),
            "timestamp": time.time(),
        }
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")
        torch.save(
            {
                "version": 1,
                "global_step": global_step,
                "world_size": world_size,
                "config": _resume_config_signature(config),
                "ranks": rank_states,
            },
            tmp_ckpt_dir / "training_state.pt",
        )

        # save norm stats
        _checkpoints.save_data_configs_assets(
            tmp_ckpt_dir / "assets",
            data_loader.data_configs(),
            data_loader.checkpoint_asset_metadata(),
        )

        # Atomically move temp directory to final location
        if final_ckpt_dir.exists():
            shutil.rmtree(final_ckpt_dir)
        tmp_ckpt_dir.rename(final_ckpt_dir)

        logging.info(f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")

        # Log checkpoint to wandb
        if config.wandb_enabled:
            wandb.log({"checkpoint_step": global_step}, step=global_step)

    if distributed:
        dist.barrier()
    # Checkpoint I/O and logging must not advance the next training step's RNG.
    _restore_rng_state(rank_state["rng"])


def load_checkpoint(model, optimizer, checkpoint_dir, device, data_loader, config):
    """Restore one selected step's weights, optimizer, data position and RNG."""
    ckpt_dir = pathlib.Path(checkpoint_dir)
    if not ckpt_dir.is_dir() or not ckpt_dir.name.isdigit():
        raise ValueError(f"Expected a checkpoint step directory, got {ckpt_dir}.")
    latest_step = int(ckpt_dir.name)
    training_state_path = ckpt_dir / "training_state.pt"
    if not training_state_path.is_file():
        raise ValueError(
            f"Checkpoint {ckpt_dir} has no training_state.pt and cannot resume its data or RNG state. "
            "Use its weights to initialize a new experiment instead."
        )
    training_state = torch.load(training_state_path, map_location="cpu", weights_only=False)
    distributed = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if distributed else 1
    rank = dist.get_rank() if distributed else 0
    if training_state.get("version") != 1 or training_state.get("world_size") != world_size:
        raise ValueError("Resume requires a supported training state and the original number of DDP processes.")
    if training_state.get("config") != _resume_config_signature(config):
        raise ValueError("Resume requires the original model, training stage, precision, optimizer and LR schedule.")
    rank_states = training_state.get("ranks", [])
    if len(rank_states) != world_size or rank_states[rank].get("rank") != rank:
        raise ValueError(f"Training checkpoint has no valid runtime state for rank {rank}.")
    rank_state = rank_states[rank]

    # Clear memory before loading checkpoints
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "before_loading_checkpoint")

    try:
        # Load model state with error handling
        logging.info("Loading model state...")
        safetensors_path = ckpt_dir / "model.safetensors"

        if safetensors_path.exists():
            model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            safetensors.torch.load_model(model_to_load, safetensors_path, device=str(device))
            logging.info("Loaded model state from safetensors format")
        else:
            raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_model")

        # Load optimizer state with error handling
        logging.info("Loading optimizer state...")
        optimizer_path = ckpt_dir / "optimizer.pt"

        if optimizer_path.exists():
            optimizer_state_dict = torch.load(optimizer_path, map_location=device, weights_only=False)
            logging.info("Loaded optimizer state from pt format")
        else:
            raise FileNotFoundError(f"No optimizer checkpoint found at {ckpt_dir}")

        optimizer.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_optimizer")

        # Load metadata
        logging.info("Loading metadata...")
        metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
        global_step = metadata.get("global_step", latest_step)
        if global_step != latest_step or training_state.get("global_step") != global_step:
            raise ValueError(f"Checkpoint {ckpt_dir} has inconsistent completed-step metadata.")
        del metadata
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_metadata")

        # Seeking can start workers and decode discarded samples. Restore the
        # checkpoint's main-process RNG only after all of that work is complete.
        data_loader.load_state_dict(rank_state["data_loader"])
        _restore_rng_state(rank_state["rng"])

        logging.info(f"Successfully loaded all checkpoint components from step {latest_step}")
        return global_step

    except RuntimeError as e:
        if "out of memory" in str(e):
            # Clear memory and provide detailed error message
            torch.cuda.empty_cache()
            gc.collect()
            logging.error(f"Out of memory error while loading checkpoint: {e!s}")
            log_memory_usage(device, latest_step, "after_oom_error")
            raise RuntimeError(
                "Out of memory while loading checkpoint. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            ) from e
        raise


def get_latest_checkpoint_step(checkpoint_dir):
    """Get the latest checkpoint step number from a checkpoint directory."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(checkpoint_steps) if checkpoint_steps else None


def log_memory_usage(device, step, phase="unknown"):
    """Log detailed memory usage information."""
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_free = torch.cuda.mem_get_info(device)[0] / 1e9

    # Get more detailed memory info
    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

    # Get DDP info if available
    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}"
    )


def _sample_batch_from_existing_loader(
    loader: _data.DataLoader[tuple[_model.Observation, _model.Actions]],
    *,
    seed: int,
) -> tuple[_model.Observation, _model.Actions]:
    """Materialize one preview batch from an already-initialized loader."""
    torch_loader = loader._data_loader.torch_loader  # type: ignore[attr-defined]
    sample = torch_loader.dataset[0]
    batch = _data._collate_fn([sample])
    batch = _data._maybe_apply_world_model_sampling(batch, loader.data_config(), np.random.default_rng(seed))
    batch = jax.tree.map(torch.as_tensor, batch)
    return _model.Observation.from_dict(batch), batch.get("actions")


def _wandb_image(frame: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert a channels-first or channels-last policy image to uint8 RGB."""
    frame = frame.detach().cpu().numpy() if isinstance(frame, torch.Tensor) else np.asarray(frame)
    if frame.ndim != 3:
        raise ValueError(f"Expected a 3D RGB frame, got {frame.shape}.")
    if frame.shape[-1] != 3:
        if frame.shape[0] != 3:
            raise ValueError(f"Expected three RGB channels, got {frame.shape}.")
        frame = np.moveaxis(frame, 0, -1)
    if np.issubdtype(frame.dtype, np.floating):
        # Policy images are normalized to [-1, 1], including bright frames
        # whose pixels happen to be entirely nonnegative.
        frame = (frame + 1.0) * 127.5
    return np.clip(frame, 0, 255).astype(np.uint8)


def log_sample_images_to_wandb(sample_batch: tuple[_model.Observation, _model.Actions]):
    """Log sample batch images to wandb for visualization.

    Logs representative visual inputs to verify:
    - Data pipeline correctness (temporal alignment, normalization)
    - Multi-view consistency across cameras
    - World model frame sampling (history/future context)
    """
    observation, actions = sample_batch
    sample_batch = observation.to_dict()

    # Separate image categories for clean UI organization
    single_frame_images = {}  # Current observations: {camera_name: image}
    temporal_images = {}  # World model sequences: {camera_name: concat_frames}

    sample_idx = 0  # Log first sample only

    for key, img in sample_batch["image"].items():
        if img.ndim == 4:
            frame = _wandb_image(img[sample_idx])
            single_frame_images[key] = frame

        elif img.ndim == 5:
            num_frames = img.shape[1]
            frames = [_wandb_image(img[sample_idx, t]) for t in range(num_frames)]

            # Horizontal concatenation for temporal visualization
            concat_img = np.concatenate(frames, axis=1)
            temporal_images[key] = (concat_img, num_frames)

    # Log organized by visual input type
    log_dict = {}
    # Log each view under its own key. W&B requires all images in one list to
    # have the same dimensions; camera views and temporal sequences may not.
    for key, frame in single_frame_images.items():
        log_dict[f"data/observations/{key}"] = wandb.Image(frame, caption=key)
    for key, (concat_img, num_frames) in temporal_images.items():
        log_dict[f"data/world_model_context/{key}"] = wandb.Image(
            concat_img, caption=f"{key} (T={num_frames})"
        )

    # Log action statistics for verification
    if actions is not None:
        action_tensor = actions[sample_idx]  # (action_horizon, action_dim)
        log_dict["data/action_mean"] = action_tensor.mean().item()
        log_dict["data/action_std"] = action_tensor.std().item()
        log_dict["data/action_range"] = (action_tensor.max() - action_tensor.min()).item()

    if log_dict:
        wandb.log(log_dict, step=0)
        logging.info(
            f"Logged sample data: {len(single_frame_images)} observation views, "
            f"{len(temporal_images)} temporal sequences"
        )

    # Clean up
    del sample_batch, observation, actions, single_frame_images, temporal_images
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _checkpoint_tensor_into_target(key: str, saved: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Allow only the future-query table to grow during a training handoff."""
    if saved.shape == target.shape:
        return saved
    if (
        key == "world_future_builder.slot_embed"
        and saved.ndim == target.ndim == 3
        and saved.shape[0] == target.shape[0]
        and saved.shape[2] == target.shape[2]
        and 0 < saved.shape[1] < target.shape[1]
    ):
        fitted = target.detach().clone()
        fitted[:, : saved.shape[1], :] = saved.to(device=fitted.device, dtype=fitted.dtype)
        logging.info("Expanded %s from %s to %s", key, tuple(saved.shape), tuple(target.shape))
        return fitted
    raise ValueError(
        f"Incompatible checkpoint tensor {key!r}: saved shape {tuple(saved.shape)}, "
        f"model shape {tuple(target.shape)}. Use a matching model configuration."
    )


def _load_pytorch_weights(model: torch.nn.Module, model_path: str, *, training_stage: str) -> None:
    """Validate stage initialization before loading any model weights.

    Stage I may initialize its foundation from a base π₀.₅ checkpoint without
    world-model tensors. Later handoffs require every model tensor, including
    shared parameters omitted by SafeTensors, and reject structural mismatches.
    """
    saved_state = safetensors.torch.load_file(model_path)
    target_state = model.state_dict()
    unexpected = sorted(saved_state.keys() - target_state.keys())
    if unexpected:
        raise ValueError(f"Unexpected checkpoint tensors: {unexpected[:10]}")
    if not saved_state:
        raise ValueError(f"Checkpoint {model_path} contains no model tensors")

    world_prefixes = (
        "world_model_adapter.",
        "world_future_builder.",
        "world_pred_head.",
        "paligemma_with_expert.gemma_world_model_expert.",
    )
    base_initialization = training_stage == "wm_alignment" and not any(
        key.startswith(world_prefixes) for key in saved_state
    )

    # save_model writes only one name for tied parameters. Match exact tensor
    # views so alias names do not masquerade as missing pretrained weights.
    def tensor_view(tensor: torch.Tensor) -> tuple:
        return (tensor.data_ptr(), tensor.storage_offset(), tuple(tensor.shape), tuple(tensor.stride()))

    loaded_views = {tensor_view(target_state[key]) for key in saved_state}
    missing = [
        key
        for key, target in target_state.items()
        if key not in saved_state
        and tensor_view(target) not in loaded_views
        and not (base_initialization and key.startswith(world_prefixes))
    ]
    if missing:
        raise ValueError(f"Missing checkpoint tensors for {training_stage}: {sorted(missing)[:10]}")
    loadable = {
        key: _checkpoint_tensor_into_target(key, saved, target_state[key])
        for key, saved in saved_state.items()
    }
    model.load_state_dict(loadable, strict=False)
    if base_initialization:
        logging.info("Initialized foundation weights; new world-model tensors keep their initial values")


def train_loop(config: _config.TrainConfig):
    use_ddp, _local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    rank = dist.get_rank() if use_ddp else 0
    set_seed(config.seed, rank)

    if not config.resume and not config.overwrite:
        # Rank 0 checks before creating the directory, then shares the result so
        # every DDP process exits together instead of waiting at the next barrier.
        existing_directory = [config.checkpoint_dir.exists() if is_main else False]
        if use_ddp:
            dist.broadcast_object_list(existing_directory, src=0)
        if existing_directory[0]:
            raise FileExistsError(
                f"Experiment directory {config.checkpoint_dir} already exists. "
                "Use a new --exp-name, --resume, or --overwrite."
            )

    # Initialize checkpoint directory and wandb
    resuming = False
    resume_checkpoint_dir = None
    if config.resume:
        # Find checkpoint directory based on experiment name
        exp_checkpoint_dir = config.checkpoint_dir
        if exp_checkpoint_dir.exists():
            # Use validation to find the latest working checkpoint
            latest_step = get_latest_checkpoint_step(exp_checkpoint_dir)
            if latest_step is not None:
                resume_checkpoint_dir = exp_checkpoint_dir / str(latest_step)
                if not (resume_checkpoint_dir / "training_state.pt").is_file():
                    raise ValueError(
                        f"Checkpoint {exp_checkpoint_dir / str(latest_step)} has no resumable training state. "
                        "Use its weights to initialize a new experiment instead of --resume."
                    )
                resuming = True
                logging.info(
                    f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} at step {latest_step}"
                )
            else:
                raise FileNotFoundError(f"No valid checkpoints found in {exp_checkpoint_dir} for resume")
        else:
            raise FileNotFoundError(f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume")
    elif is_main and config.overwrite and config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
        logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")

    # Create checkpoint directory with experiment name. In DDP, only rank 0 mutates
    # the filesystem and the rest wait until the directory is ready.
    if not resuming:
        exp_checkpoint_dir = config.checkpoint_dir
        if is_main:
            exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created experiment checkpoint directory: {exp_checkpoint_dir}")
    elif is_main:
        # For resume, checkpoint_dir is already set to the experiment directory.
        logging.info(f"Using existing experiment checkpoint directory: {config.checkpoint_dir}")

    if use_ddp:
        dist.barrier()

    # Initialize wandb (only on main process)
    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # Build data loader using the unified data loader
    # Calculate effective batch size per GPU for DDP
    # For N GPUs, each GPU should get batch_size/N samples, so total across all GPUs is batch_size
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    if world_size < 1 or config.batch_size % world_size != 0:
        raise ValueError(
            f"batch_size ({config.batch_size}) must be divisible by the number of processes ({world_size}). "
            "Set NUM_GPUS to a divisor of the config batch size, or pass --batch-size."
        )
    effective_batch_size = config.batch_size // world_size
    logging.info(
        f"Using batch size per GPU: {effective_batch_size} (total batch size across {world_size} GPUs: {config.batch_size})"
    )

    # Pass the original batch size to data loader - it will handle DDP splitting internally
    loader = build_datasets(
        config,
        checkpoint_assets=resume_checkpoint_dir / "assets" if resume_checkpoint_dir is not None else None,
    )

    # Log sample images to wandb (only for main process on new runs)
    if is_main and config.wandb_enabled and not resuming:
        log_sample_images_to_wandb(_sample_batch_from_existing_loader(loader, seed=config.seed))

    # Build model
    if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
        # Convert dataclass to Pi0Config if needed
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            world_model_expert_variant=getattr(config.model, "world_model_expert_variant", "gemma_300m"),
            enable_world_model=getattr(config.model, "enable_world_model", False),
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        model_cfg = copy.copy(config.model)
        # Update dtype to match pytorch_training_precision
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    if device.type != "cuda":
        raise RuntimeError(f"Expected CUDA device, got {device}")
    object.__setattr__(model_cfg, "device", device)
    object.__setattr__(model_cfg, "training_stage", config.training_stage)

    model_init_started = time.monotonic()
    model_init_stop = threading.Event()
    model_init_thread = None
    if is_main:
        logging.info("Starting model initialization on %s; large Hugging Face downloads may take several minutes", device)
        model_init_thread = threading.Thread(
            target=_log_model_init_progress,
            args=(model_init_stop,),
            name="model-init-progress",
            daemon=True,
        )
        model_init_thread.start()
    try:
        model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)
    finally:
        model_init_stop.set()
        if model_init_thread is not None:
            model_init_thread.join(timeout=1.0)
    if is_main:
        logging.info("Model initialization completed in %.1fs", time.monotonic() - model_init_started)

    if hasattr(model, "gradient_checkpointing_enable"):
        enable_gradient_checkpointing = bool(getattr(config, "enable_gradient_checkpointing", True))
        if enable_gradient_checkpointing:
            gc_modules = getattr(config, "gradient_checkpointing_modules", None)
            model.gradient_checkpointing_enable(modules=gc_modules)
            logging.info(
                "Enabled gradient checkpointing for modules: %s",
                gc_modules if gc_modules else "all (default)",
            )
        elif hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
            logging.info("Disabled gradient checkpointing for throughput optimization")
        else:
            logging.info("Gradient checkpointing toggle requested, but disable path is unavailable")
    else:
        enable_gradient_checkpointing = False
        logging.info("Gradient checkpointing is not supported for this model")

    # Log initial memory usage after model creation
    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

    # Enable memory optimizations for large-scale training
    if world_size >= 8:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set memory allocation configuration
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        logging.info("Enabled memory optimizations for 8+ GPU training")

    if use_ddp:
        ddp_static_graph = getattr(config, "ddp_static_graph", None)
        if ddp_static_graph is None:
            # static_graph requires every trainable parameter to receive a
            # gradient on every step. Post-training leaves the last VLM layer's
            # output projection unused (it never feeds another layer), and
            # world-model dropout changes the set of unused parameters between
            # steps. Stage 1 opts in explicitly after freezing those modules.
            wm_dropout = float(getattr(config.model, "wm_loss_dropout_alpha", 0.0))
            wm_enabled = bool(getattr(config.model, "enable_world_model", False))
            post_training = config.training_stage == "post_training"
            ddp_static_graph = (
                world_size >= 8 and not post_training and not (wm_enabled and wm_dropout > 0.0)
            )
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=bool(getattr(config, "ddp_find_unused_parameters", True)),
            gradient_as_bucket_view=True,  # Enable for memory efficiency
            static_graph=bool(ddp_static_graph),
        )

    # Load weights from weight_loader if specified (for fine-tuning)
    # Stage handoff validates the foundation and world-model weights before loading.
    if config.pytorch_weight_path is not None and not resuming:
        logging.info(f"Loading PI05 weights from: {config.pytorch_weight_path}")

        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
        target_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        _load_pytorch_weights(target_model, model_path, training_stage=config.training_stage)
        logging.info(f"Loaded PyTorch weights from {config.pytorch_weight_path}")

    # Optimizer + learning rate schedule from config
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    # Create optimizer with config parameters
    fused_adamw = bool(getattr(config.optimizer, "fused", False))
    if fused_adamw and "fused" not in torch.optim.AdamW.__init__.__code__.co_varnames:
        logging.warning("Requested fused AdamW, but this PyTorch build does not support it; falling back.")
        fused_adamw = False
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
        fused=fused_adamw,
    )

    # Load checkpoint if resuming
    global_step = 0
    if resuming:
        global_step = load_checkpoint(model, optim, resume_checkpoint_dir, device, loader, config)
        logging.info(f"Resumed training from step {global_step}")

    def lr_schedule(step: int):
        if step < warmup_steps:
            # Match JAX behavior: start from peak_lr / (warmup_steps + 1)
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        # cosine decay
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    model.train()
    start_time = time.time()
    infos = []  # Collect stats over log interval
    if is_main:
        logging.info(
            f"Running on: {platform.node()} | world_size={torch.distributed.get_world_size() if use_ddp else 1}"
        )
        logging.info(
            f"Training config: batch_size={config.batch_size}, effective_batch_size={effective_batch_size}, num_train_steps={config.num_train_steps}"
        )
        logging.info(f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}")
        logging.info(
            "DDP optimizer flags: find_unused_parameters=%s static_graph=%s fused_adamw=%s",
            bool(getattr(config, "ddp_find_unused_parameters", True)),
            bool(ddp_static_graph) if use_ddp else False,
            fused_adamw,
        )
        logging.info(
            f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}"
        )
        logging.info(
            f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}"
        )
        logging.info("EMA is not supported for PyTorch training")
        logging.info(f"Training precision: {model_cfg.dtype}")

    # Training loop - iterate until we reach num_train_steps
    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main)
        if is_main
        else None
    )

    while global_step < config.num_train_steps:
        for observation, actions in loader:
            observation = jax.tree.map(
                lambda x: x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x,
                observation,
            )  # noqa: PLW2901
            if actions is not None:
                actions = actions.to(device=device, dtype=torch.float32, non_blocking=True)  # noqa: PLW2901

            # Update LR
            for pg in optim.param_groups:
                pg["lr"] = lr_schedule(global_step)

            # Forward pass
            metrics = {}
            model_out = model(observation, actions)

            # New format: (loss_tensor, metrics_dict)
            if isinstance(model_out, tuple) and len(model_out) == 2 and isinstance(model_out[1], dict):
                losses, metrics = model_out
            else:
                losses = model_out

            if isinstance(losses, (list, tuple)):
                losses = torch.stack(
                    [
                        x if isinstance(x, torch.Tensor) else torch.tensor(x, device=device, dtype=torch.float32)
                        for x in losses
                    ]
                )
            elif not isinstance(losses, torch.Tensor):
                losses = torch.tensor(losses, device=device, dtype=torch.float32)

            loss = losses.mean()

            # Backward pass
            loss.backward()

            # Log memory usage after backward pass
            if global_step < 5 and is_main and torch.cuda.is_available():
                log_memory_usage(device, global_step, "after_backward")

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optimizer.clip_gradient_norm)

            # Optimizer step
            optim.step()
            optim.zero_grad(set_to_none=True)

            # Collect stats
            if is_main:
                info = {
                    "loss": loss.item(),
                    "learning_rate": optim.param_groups[0]["lr"],
                    "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                }

                for k, v in metrics.items():
                    if isinstance(v, torch.Tensor):
                        info[k] = float(v.detach().mean().item())
                    else:
                        info[k] = float(v)

                infos.append(info)

            if is_main and (global_step % config.log_interval == 0):
                elapsed = time.time() - start_time

                log_payload = _build_log_payload(
                    infos,
                    global_step=global_step,
                    elapsed=elapsed,
                    log_interval=config.log_interval,
                )
                avg_loss = log_payload["loss"]
                avg_lr = log_payload["learning_rate"]
                avg_grad_norm = log_payload.get("grad_norm")
                logging.info(
                    f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} grad_norm={avg_grad_norm:.2f} time={elapsed:.1f}s"
                    if avg_grad_norm is not None
                    else f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} time={elapsed:.1f}s"
                )

                # Log to wandb
                if config.wandb_enabled and len(infos) > 0:
                    wandb.log(log_payload, step=global_step)

                start_time = time.time()
                infos = []  # Reset stats collection

            global_step += 1
            # Save checkpoint using the new mechanism
            save_checkpoint(model, optim, global_step, config, is_main, loader)

            # Update progress bar
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "lr": f"{optim.param_groups[0]['lr']:.2e}", "step": global_step}
                )
            if global_step >= config.num_train_steps:
                break

    # Close progress bar
    if pbar is not None:
        pbar.close()

    # Finish wandb run
    if is_main and config.wandb_enabled:
        wandb.finish()

    cleanup_ddp()


def main():
    init_logging(logging_level=logging.INFO)
    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()
