#!/usr/bin/env python3
"""Download the base checkpoint, V-JEPA2 encoder, tokenizer, and LIBERO dataset.

Each item is skipped when it is already on disk. ``--stage 2``, ``--stage 3``,
and ``--stage all`` also compute normalization statistics for those stages.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import subprocess
import sys

import openpi.shared.download as download
import openpi.training.base_checkpoints as base_checkpoints

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
NORM_STATS_CONFIGS = {
    "2": "stage2_pretraining",
    "3": "stage3_finetuning_libero",
}

logger = logging.getLogger("download_assets")


def _dataset_ready(path: pathlib.Path) -> bool:
    return (path / "meta" / "info.json").is_file() and (path / "data").is_dir()


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
    if not _dataset_ready(local_dir):
        raise FileNotFoundError(
            f"Downloaded {repo_id}, but {local_dir} is not a LeRobot dataset "
            "(expected meta/info.json and data/)."
        )
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
    if weights.is_file():
        logger.info("Base checkpoint %s already present at %s", spec.name, output_dir)
        return output_dir

    logger.info("Downloading %s and converting it to %s", spec.jax_uri, output_dir)
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "examples" / "convert_jax_model_to_pytorch.py"),
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
    if not weights.is_file():
        raise FileNotFoundError(f"Conversion finished without writing {weights}")
    return output_dir


def ensure_vjepa2() -> None:
    logger.info("Fetching V-JEPA2 weights %s", base_checkpoints.VJEPA2_REPO)
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=base_checkpoints.VJEPA2_REPO)


def ensure_paligemma_tokenizer() -> None:
    logger.info("Fetching PaliGemma tokenizer %s", base_checkpoints.PALIGEMMA_TOKENIZER)
    download.maybe_download(base_checkpoints.PALIGEMMA_TOKENIZER, gs={"token": "anon"})


def _norm_stats_path(config_name: str) -> pathlib.Path:
    import openpi.training.config as config

    train_config = config.get_config(config_name)
    return train_config.assets_dirs / "libero_v3_eef" / "norm_stats.json"


def ensure_norm_stats(config_name: str) -> None:
    stats_path = _norm_stats_path(config_name)
    if stats_path.is_file():
        logger.info("Norm stats for %s already present at %s", config_name, stats_path)
        return

    logger.info("Computing norm stats for %s", config_name)
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "compute_norm_stats.py"), "--config-name", config_name],
        cwd=REPO_ROOT,
        check=True,
    )
    if not stats_path.is_file():
        raise FileNotFoundError(f"Norm stats were not written to {stats_path}")


def download_assets(stage: str, checkpoint_name: str) -> None:
    ensure_dataset()
    checkpoint_dir = ensure_base_checkpoint(checkpoint_name)
    ensure_vjepa2()
    ensure_paligemma_tokenizer()
    if stage in {"2", "all"}:
        ensure_norm_stats(NORM_STATS_CONFIGS["2"])
    if stage in {"3", "all"}:
        ensure_norm_stats(NORM_STATS_CONFIGS["3"])
    logger.info("Stage I init checkpoint: %s", checkpoint_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("1", "2", "3", "all"), default="1")
    parser.add_argument("--checkpoint", default="pi05_base", help="Name in openpi.training.base_checkpoints.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    download_assets(args.stage, args.checkpoint)


if __name__ == "__main__":
    main()
