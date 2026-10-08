#!/usr/bin/env python3
# Copyright 2026 PLaW-VLA authors. SPDX-License-Identifier: Apache-2.0
"""Download the base checkpoint, V-JEPA2 encoder, tokenizer, and LIBERO dataset.

Complete local assets are reused. ``--stage 3`` and
``--stage all`` prepare LIBERO fine-tuning statistics. Pretraining data
must be acquired and converted separately; see docs/pretraining.md.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import pathlib
import subprocess
import sys

import plawvla.shared.download as download
import plawvla.training.base_checkpoints as base_checkpoints

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
NORM_STATS_CONFIGS = {
    "3": "stage3_finetuning_libero",
}

logger = logging.getLogger("download_assets")


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, found {value!r}.")
    return value


def _dataset_file(path: pathlib.Path, pattern: str, **indices: object) -> pathlib.Path:
    if not isinstance(pattern, str):
        raise ValueError("Dataset shard paths must be strings.")
    relative = pathlib.Path(pattern.format(**indices))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Dataset file points outside its directory: {relative}.")
    return path / relative


def _validate_dataset(path: pathlib.Path) -> None:
    """Check LIBERO metadata and shard footers without loading frame or video data."""
    import pyarrow.parquet as pq

    info = json.loads((path / "meta/info.json").read_text())
    if not isinstance(info, dict) or info.get("codebase_version") != "v3.0":
        raise ValueError("Expected a LeRobot v3.0 dataset.")
    total_episodes = _positive_int(info["total_episodes"], "total_episodes")
    total_frames = _positive_int(info["total_frames"], "total_frames")
    total_tasks = _positive_int(info["total_tasks"], "total_tasks")
    if pq.ParquetFile(path / "meta/tasks.parquet").metadata.num_rows != total_tasks:
        raise ValueError("Task metadata does not match total_tasks.")

    features = info["features"]
    if not isinstance(features, dict) or any(not isinstance(feature, dict) for feature in features.values()):
        raise ValueError("Dataset features must describe each feature's dtype.")
    raw_stats_path = path / "meta/stats.json"
    if not raw_stats_path.is_file():
        raw_stats_path = path / "meta/norm_stats.json"
    raw_stats = json.loads(raw_stats_path.read_text())
    if not isinstance(raw_stats, dict):
        raise ValueError("Raw dataset statistics must contain a JSON object.")
    for key in ("observation.state", "action"):
        shape = features[key]["shape"]
        if not isinstance(shape, list) or len(shape) != 1:
            raise ValueError(f"Expected a vector feature for {key}.")
        dimension = _positive_int(shape[0], f"{key} dimension")
        feature_stats = raw_stats[key]
        if not isinstance(feature_stats, dict):
            raise ValueError(f"Raw statistics must describe {key}.")
        q01, q99 = feature_stats["q01"], feature_stats["q99"]
        if any(
            not isinstance(quantile, list)
            or len(quantile) != dimension
            or any(
                not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)
                for value in quantile
            )
            for quantile in (q01, q99)
        ):
            raise ValueError(f"Raw statistics for {key} must contain finite {dimension}D q01/q99 arrays.")
        if any(low > high for low, high in zip(q01, q99, strict=True)):
            raise ValueError(f"Raw statistics for {key} have inverted quantiles.")
    video_keys = [key for key, feature in features.items() if feature.get("dtype") == "video"]
    columns = ["episode_index", "length", "data/chunk_index", "data/file_index"]
    for key in video_keys:
        columns.extend((f"videos/{key}/chunk_index", f"videos/{key}/file_index"))
    data_frames: dict[pathlib.Path, int] = {}
    videos: set[pathlib.Path] = set()
    episodes: set[int] = set()
    for episode_file in sorted((path / "meta/episodes").rglob("*.parquet")):
        for episode in pq.read_table(episode_file, columns=columns).to_pylist():
            episode_index = episode["episode_index"]
            if not isinstance(episode_index, int) or isinstance(episode_index, bool) or episode_index < 0:
                raise ValueError(f"Invalid episode index: {episode_index!r}.")
            if episode_index in episodes:
                raise ValueError(f"Duplicate episode index: {episode_index}.")
            episodes.add(episode_index)
            length = _positive_int(episode["length"], f"Episode {episode_index} length")
            for prefix in ("data", *(f"videos/{key}" for key in video_keys)):
                chunk_index = episode[f"{prefix}/chunk_index"]
                file_index = episode[f"{prefix}/file_index"]
                if any(
                    not isinstance(index, int) or isinstance(index, bool) or index < 0
                    for index in (chunk_index, file_index)
                ):
                    raise ValueError(f"Invalid shard indices in episode {episode_index}: {prefix}.")
                indices = {"chunk_index": chunk_index, "file_index": file_index}
                if prefix == "data":
                    data_file = _dataset_file(path, info["data_path"], **indices)
                    data_frames[data_file] = data_frames.get(data_file, 0) + length
                else:
                    videos.add(
                        _dataset_file(path, info["video_path"], video_key=prefix.removeprefix("videos/"), **indices)
                    )
    if episodes != set(range(total_episodes)) or sum(data_frames.values()) != total_frames:
        raise ValueError("Episode metadata does not cover the declared episodes and frames.")
    for data_file, expected_frames in data_frames.items():
        if pq.ParquetFile(data_file).metadata.num_rows != expected_frames:
            raise ValueError(f"Incomplete frame shard: {data_file}.")
    for video in videos:
        if not video.is_file() or video.stat().st_size == 0:
            raise ValueError(f"Missing or empty video shard: {video}.")


def _dataset_ready(path: pathlib.Path) -> bool:
    try:
        _validate_dataset(path)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        logger.debug("Dataset is incomplete at %s: %s", path, exc)
        return False
    return True


def _base_checkpoint_ready(path: pathlib.Path) -> bool:
    """Check SafeTensors headers and core foundation shapes without reading tensor data."""
    from safetensors import SafetensorError
    from safetensors import safe_open

    expected_config = {
        "pi05": True,
        "action_dim": 32,
        "paligemma_variant": "gemma_2b",
        "action_expert_variant": "gemma_300m",
    }
    shapes = {
        "action_in_proj.weight": (1024, 32),
        "action_out_proj.weight": (32, 1024),
        "time_mlp_in.weight": (1024, 1024),
        "time_mlp_out.weight": (1024, 1024),
        "paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj.weight": (2048, 1024),
        "paligemma_with_expert.gemma_expert.model.layers.17.self_attn.q_proj.weight": (2048, 1024),
        "paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.q_proj.weight": (2048, 2048),
        "paligemma_with_expert.paligemma.model.language_model.layers.17.self_attn.q_proj.weight": (2048, 2048),
        "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight": (
            1152,
            3,
            14,
            14,
        ),
        "paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.26.self_attn.q_proj.weight": (
            1152,
            1152,
        ),
    }
    try:
        config = json.loads((path / "config.json").read_text())
        if not isinstance(config, dict) or any(
            type(config.get(key)) is not type(value) or config[key] != value for key, value in expected_config.items()
        ):
            return False
        dtype = {"bfloat16": "BF16", "float32": "F32", "float16": "F16"}[config["precision"]]
        with safe_open(path / "model.safetensors", framework="np") as weights:
            keys = set(weights.keys())
            embeddings = (
                "paligemma_with_expert.paligemma.lm_head.weight",
                "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight",
            )
            embedding_key = next((key for key in embeddings if key in keys), None)
            if embedding_key is None:
                return False
            shapes[embedding_key] = (257152, 2048)
            return all(
                key in keys
                and tuple(weights.get_slice(key).get_shape()) == shape
                and weights.get_slice(key).get_dtype() == dtype
                for key, shape in shapes.items()
            )
    except (KeyError, OSError, TypeError, ValueError, SafetensorError) as exc:
        logger.debug("Base checkpoint is incomplete at %s: %s", path, exc)
        return False


def ensure_dataset() -> pathlib.Path:
    local_dir = REPO_ROOT / base_checkpoints.DEFAULT_DATASET_DIR
    if _dataset_ready(local_dir):
        logger.info("Dataset already present at %s", local_dir)
        return local_dir

    repo_id = base_checkpoints.DEFAULT_DATASET_REPO
    logger.info("Downloading %s into %s", repo_id, local_dir)
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(local_dir))
    try:
        _validate_dataset(local_dir)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise ValueError(f"Downloaded {repo_id}, but the dataset is incomplete at {local_dir}: {exc}") from exc
    return local_dir


def ensure_base_checkpoint(name: str = "pi05_base") -> pathlib.Path:
    spec = base_checkpoints.get_base_checkpoint(name)
    if not spec.pi05:
        raise ValueError(
            f"Base checkpoint {spec.name!r} is not a π0.5 checkpoint. "
            "Stage I initializes PaliGemma and the action expert from a π0.5 JAX checkpoint "
            "with the same layout as pi05_base."
        )
    output_dir = REPO_ROOT / "checkpoints" / spec.pytorch_dirname
    weights = output_dir / "model.safetensors"
    if _base_checkpoint_ready(output_dir):
        logger.info("Base checkpoint %s already present at %s", spec.name, output_dir)
        return output_dir

    logger.info("Downloading %s and converting it to %s", spec.jax_uri, output_dir)
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "setup" / "convert_jax_model_to_pytorch.py"),
            "--checkpoint-dir",
            spec.jax_uri,
            "--config-name",
            "stage1_world_model_pretraining",
            "--output-path",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    if not _base_checkpoint_ready(output_dir):
        raise ValueError(f"Conversion finished without a complete π0.5 base checkpoint at {weights}.")
    return output_dir


def ensure_stage3_initialization(source: str = "auto") -> tuple[pathlib.Path, base_checkpoints.Stage3Initialization]:
    """Prepare the registry-selected direct Stage III initialization."""

    spec = base_checkpoints.resolve_stage3_initialization(source)
    if spec.name == "pi05":
        if spec.base_checkpoint is None:
            raise ValueError("The pi05 Stage III initialization must name a base checkpoint.")
        return ensure_base_checkpoint(spec.base_checkpoint), spec

    if spec.pytorch_dirname is None:
        raise ValueError(f"Stage III initialization {spec.name!r} does not define a local checkpoint directory.")
    if (spec.uri is None) == (spec.hf_repo_id is None):
        raise ValueError(
            f"Released Stage III initialization {spec.name!r} must define exactly one of uri or hf_repo_id."
        )

    output_dir = REPO_ROOT / "checkpoints" / spec.pytorch_dirname
    if (output_dir / "model.safetensors").is_file():
        logger.info("Stage III initialization %s already present at %s", spec.name, output_dir)
        return output_dir, spec

    output_dir.mkdir(parents=True, exist_ok=True)
    if spec.hf_repo_id is not None:
        logger.info("Downloading official Stage III initialization %s into %s", spec.hf_repo_id, output_dir)
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=spec.hf_repo_id, repo_type="model", local_dir=str(output_dir))
    else:
        assert spec.uri is not None
        logger.info("Downloading official Stage III initialization %s into %s", spec.uri, output_dir)
        downloaded = download.maybe_download(spec.uri)
        if downloaded.is_dir():
            import shutil

            shutil.copytree(downloaded, output_dir, dirs_exist_ok=True)
        else:
            raise ValueError(
                f"Stage III initialization URI must resolve to a checkpoint directory, got {downloaded}."
            )

    if not (output_dir / "model.safetensors").is_file():
        raise ValueError(
            f"Downloaded Stage III initialization {spec.name!r}, but model.safetensors is missing at {output_dir}."
        )
    return output_dir, spec


def ensure_vjepa2() -> None:
    logger.info("Fetching V-JEPA2 weights %s", base_checkpoints.VJEPA2_REPO)
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=base_checkpoints.VJEPA2_REPO)


def ensure_paligemma_tokenizer() -> None:
    logger.info("Fetching PaliGemma tokenizer %s", base_checkpoints.PALIGEMMA_TOKENIZER)
    download.maybe_download(base_checkpoints.PALIGEMMA_TOKENIZER, gs={"token": "anon"})


def _norm_stats_path(config_name: str) -> pathlib.Path:
    import plawvla.training.config as config

    train_config = config.get_config(config_name)
    return train_config.assets_dirs / "libero_v3_eef" / "norm_stats.json"


def _norm_stats_ready(config_name: str, stats_path: pathlib.Path) -> bool:
    if not stats_path.is_file():
        return False

    import numpy as np

    from plawvla.policies import libero_policy
    from plawvla.shared import normalize
    from plawvla.training import config

    train_config = config.get_config(config_name)
    data_factory = dataclasses.replace(train_config.data, load_norm_stats=False)
    data_config = data_factory.create(train_config.assets_dirs, train_config.model)
    libero_inputs = next(
        transform
        for transform in data_config.data_transforms.inputs
        if isinstance(transform, libero_policy.LiberoInputs)
    )
    dimensions = (
        (libero_policy.CANONICAL_STATE_DIM, libero_policy.CANONICAL_ACTION_DIM)
        if libero_inputs.canonicalize_ee_pose_gripper
        else (libero_policy.RAW_STATE_DIM, libero_policy.RAW_ACTION_DIM)
    )
    try:
        stats = normalize.load(stats_path.parent)
        normalize.validate_contract(stats_path.parent, data_config.normalization_contract)
        for key, dimension in zip(("state", "actions"), dimensions, strict=True):
            item = stats[key]
            values = [item.mean, item.std]
            if (item.q01 is None) != (item.q99 is None):
                return False
            if item.q01 is not None and item.q99 is not None:
                values.extend((item.q01, item.q99))
                if np.any(item.q01 > item.q99):
                    return False
            elif data_config.use_quantile_norm:
                return False
            if any(np.asarray(value).shape != (dimension,) or not np.isfinite(value).all() for value in values):
                return False
            if np.any(item.std < 0):
                return False
    except (KeyError, OSError, TypeError, ValueError) as exc:
        logger.debug("Norm stats are incomplete at %s: %s", stats_path, exc)
        return False
    return True


def ensure_norm_stats(config_name: str) -> None:
    stats_path = _norm_stats_path(config_name)
    if _norm_stats_ready(config_name, stats_path):
        logger.info("Norm stats for %s already present at %s", config_name, stats_path)
        return

    logger.info("Computing norm stats for %s", config_name)
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "train" / "compute_norm_stats.py"), "--config-name", config_name],
        cwd=REPO_ROOT,
        check=True,
    )
    if not _norm_stats_ready(config_name, stats_path):
        raise ValueError(f"Complete norm stats and their matching contract were not written to {stats_path}.")


def download_assets(stage: str, checkpoint_name: str, *, skip_base_checkpoint: bool = False) -> None:
    if stage in {"3", "all"}:
        ensure_dataset()
    checkpoint_dir = (
        ensure_base_checkpoint(checkpoint_name) if stage in {"1", "all"} and not skip_base_checkpoint else None
    )
    ensure_vjepa2()
    ensure_paligemma_tokenizer()
    if stage in {"3", "all"}:
        ensure_norm_stats(NORM_STATS_CONFIGS["3"])
    if checkpoint_dir is not None:
        logger.info("Foundation initialization checkpoint: %s", checkpoint_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("1", "2", "3", "all"), default="1")
    parser.add_argument("--checkpoint", default="pi05_base", help="Name in plawvla.training.base_checkpoints.")
    parser.add_argument(
        "--skip-base-checkpoint", action="store_true", help="Skip downloading and converting the base initialization checkpoint."
    )
    parser.add_argument(
        "--stage3-base-source",
        choices=("auto", "plawvla", "pi05"),
        help=(
            "Prepare the direct Stage III initialization selected by the release registry. "
            "This is separate from --checkpoint, which configures Stage I π₀.₅ initialization."
        ),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.stage3_base_source is not None:
        checkpoint_dir, spec = ensure_stage3_initialization(args.stage3_base_source)
        print(
            json.dumps(
                {
                    "requested_source": args.stage3_base_source,
                    "resolved_source": spec.name,
                    "weight_load_mode": spec.weight_load_mode,
                    "checkpoint_dir": str(checkpoint_dir),
                    "description": spec.description,
                }
            )
        )
        return
    download_assets(args.stage, args.checkpoint, skip_base_checkpoint=args.skip_base_checkpoint)


if __name__ == "__main__":
    main()
