from __future__ import annotations

import asyncio
from collections.abc import Sequence
import concurrent.futures as futures
import dataclasses
import json
import logging
import pathlib
from typing import Any, Protocol

from etils import epath
import jax
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils


def _iter_asset_ids(asset_id: str | Sequence[str] | None) -> list[str]:
    if asset_id is None:
        return []
    if isinstance(asset_id, str):
        return [asset_id]
    return [str(item) for item in asset_id]


def _normalize_checkpoint_asset_id(asset_id: str) -> str:
    return _config.normalize_asset_id(asset_id)


def _checkpoint_asset_ids(asset_id: str | Sequence[str] | None) -> list[str]:
    checkpoint_ids: list[str] = []
    for raw_asset_id in _iter_asset_ids(asset_id):
        checkpoint_asset_id = _normalize_checkpoint_asset_id(raw_asset_id)
        if checkpoint_asset_id not in checkpoint_ids:
            checkpoint_ids.append(checkpoint_asset_id)
    return checkpoint_ids


def save_norm_stats_assets(
    assets_dir: epath.Path | pathlib.Path | str,
    norm_stats: dict[str, _normalize.NormStats] | None,
    asset_id: str | Sequence[str] | None,
    normalization_contract: str | None = None,
) -> None:
    if norm_stats is None or asset_id is None:
        return
    for checkpoint_asset_id in _checkpoint_asset_ids(asset_id):
        _normalize.save(epath.Path(assets_dir) / checkpoint_asset_id, norm_stats)
        _normalize.save_contract(epath.Path(assets_dir) / checkpoint_asset_id, normalization_contract)


def _is_sequence_of_strs(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str)


def _save_data_config_norm_stats(
    assets_dir: epath.Path | pathlib.Path | str,
    data_config: _config.DataConfig,
) -> None:
    """Write norm_stats for one DataConfig, fanning out per-task stats when present."""
    per_repo = data_config.per_repo_norm_stats or {}
    repo_id = data_config.repo_id
    asset_id = data_config.asset_id

    if (
        per_repo
        and _is_sequence_of_strs(repo_id)
        and _is_sequence_of_strs(asset_id)
        and len(repo_id) == len(asset_id)
    ):
        for single_repo_id, single_asset_id in zip(repo_id, asset_id, strict=True):
            stats = per_repo.get(single_repo_id, data_config.norm_stats)
            save_norm_stats_assets(
                assets_dir,
                stats,
                single_asset_id,
                data_config.normalization_contract,
            )
        return

    save_norm_stats_assets(
        assets_dir,
        data_config.norm_stats,
        asset_id,
        data_config.normalization_contract,
    )


def _json_compatible(value: Any) -> Any:
    if isinstance(value, pathlib.PurePath):
        return value.as_posix()
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_json_compatible(item) for item in value]
    return value


def _write_assets_manifest(
    assets_dir: epath.Path | pathlib.Path | str,
    data_configs: Sequence[_config.DataConfig],
    checkpoint_asset_metadata: Sequence[dict[str, Any]] | None = None,
) -> None:
    manifest_datasets = []
    metadata_by_index = {
        entry.get("index", index): entry for index, entry in enumerate(checkpoint_asset_metadata or ())
    }

    for index, data_config in enumerate(data_configs):
        metadata = metadata_by_index.get(index, {})
        per_repo = data_config.per_repo_norm_stats or {}
        manifest_datasets.append(
            {
                "index": index,
                "dataset_type": metadata.get("dataset_type"),
                "repo_id": _json_compatible(metadata.get("repo_id", data_config.repo_id)),
                "asset_id": _json_compatible(metadata.get("asset_id", data_config.asset_id)),
                "checkpoint_asset_ids": _checkpoint_asset_ids(data_config.asset_id),
                "weight": metadata.get("weight"),
                "has_norm_stats": data_config.norm_stats is not None,
                # Surface per-task stats coverage so it is easy to spot from the
                # checkpoint whether multi-task norm_stats survived the round-trip.
                "has_per_repo_norm_stats": bool(per_repo),
                "per_repo_norm_stats_count": len(per_repo),
            }
        )

    manifest_path = epath.Path(assets_dir) / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"datasets": manifest_datasets}, indent=2))


def save_data_configs_assets(
    assets_dir: epath.Path | pathlib.Path | str,
    data_configs: Sequence[_config.DataConfig],
    checkpoint_asset_metadata: Sequence[dict[str, Any]] | None = None,
) -> None:
    for data_config in data_configs:
        _save_data_config_norm_stats(assets_dir, data_config)

    if len(data_configs) > 1:
        _write_assets_manifest(assets_dir, data_configs, checkpoint_asset_metadata)


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
):
    def save_assets(directory: epath.Path):
        save_data_configs_assets(
            directory,
            data_loader.data_configs(),
            data_loader.checkpoint_asset_metadata(),
        )

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    items = {
        "assets": save_assets,
        "train_state": train_state,
        "params": {"params": params},
    }
    checkpoint_manager.save(step, items)


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    del data_loader

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        restored = checkpoint_manager.restore(
            step,
            items={
                "train_state": train_state,
                "params": {"params": params},
            },
        )
    return _merge_params(restored["train_state"], restored["params"])


def load_norm_stats(
    assets_dir: epath.Path | str,
    asset_id: str | Sequence[str],
    expected_contract: str | None = None,
) -> dict[str, _normalize.NormStats] | None:
    for checkpoint_asset_id in _checkpoint_asset_ids(asset_id):
        norm_stats_dir = epath.Path(assets_dir) / checkpoint_asset_id
        try:
            norm_stats = _normalize.load(norm_stats_dir)
            _normalize.validate_contract(norm_stats_dir, expected_contract)
            logging.info(f"Loaded norm stats from {norm_stats_dir}")
            return norm_stats
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"Norm stats not found in checkpoint assets {assets_dir} for asset_id={asset_id!r}")


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])
