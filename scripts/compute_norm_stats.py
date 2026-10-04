"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import dataclasses
import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


class KeepKeys(transforms.DataTransformFn):
    def __init__(self, keys: tuple[str, ...]):
        self._keys = keys

    def __call__(self, x: dict) -> dict:
        return {k: x[k] for k in self._keys if k in x}


def _norm_stats_model_config(model_config: _model.BaseModelConfig) -> _model.BaseModelConfig:
    if dataclasses.is_dataclass(model_config) and hasattr(model_config, "enable_world_model"):
        return dataclasses.replace(model_config, enable_world_model=False)
    return model_config


def _dataset_label(data_config: _config.DataConfig) -> str:
    if data_config.repo_id:
        return data_config.repo_id
    if isinstance(data_config.asset_id, str) and data_config.asset_id:
        return data_config.asset_id
    return "<unknown-dataset>"


def _loader_label(data_config: _config.DataConfig) -> str:
    del data_config
    return "torch"


def _format_count(value: int) -> str:
    return f"{value:,}"


def _batch_size_from_batch(batch: dict, keys: list[str]) -> int:
    for key in keys:
        value = batch.get(key)
        if value is None:
            continue
        array = np.asarray(value)
        if array.ndim == 0:
            return 1
        return int(array.shape[0])
    return 0


def _resolve_norm_stats_inputs(
    config_name: str,
    *,
    num_workers: int | None,
    batch_size: int | None,
) -> tuple[_config.TrainConfig, _model.BaseModelConfig, int, int, list[_config.DataConfig]]:
    config = _config.get_config(config_name)
    stats_model_config = _norm_stats_model_config(config.model)
    effective_num_workers = config.num_workers if num_workers is None else num_workers
    effective_batch_size = config.batch_size if batch_size is None else batch_size

    data_factory = dataclasses.replace(config.data, load_norm_stats=False)
    root_data_configs = [data_factory.create(config.assets_dirs, stats_model_config)]

    expanded_data_configs: list[_config.DataConfig] = []
    for data_config in root_data_configs:
        expanded_data_configs.extend(_data_loader._expand_data_config_children(data_config))

    return config, stats_model_config, effective_num_workers, effective_batch_size, expanded_data_configs


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(
        data_config,
        action_horizon,
        model_config,
        include_videos=False,
    )
    dataset = _data_loader.FaultTolerantDataset(
        dataset,
        repo_id=str(data_config.repo_id),
        catch_all_errors=True,
    )
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            KeepKeys(("state", "actions")),
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    # Keep the scan on CPU. The loader's default JAX mesh shards each batch
    # across every visible device and rejects sizes that do not divide that count.
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
        framework="pytorch",
    )
    return data_loader, num_batches


def compute_dataset_norm_stats(
    *,
    config: _config.TrainConfig,
    data_config: _config.DataConfig,
    stats_model_config: _model.BaseModelConfig,
    dataset_index: int,
    total_datasets: int,
    max_frames: int | None,
    num_workers: int,
    batch_size: int,
    show_progress: bool,
) -> None:
    dataset_name = _dataset_label(data_config)
    loader_name = _loader_label(data_config)
    tqdm.tqdm.write(f"[{dataset_index}/{total_datasets}] Preparing {dataset_name} with {loader_name} loader")

    data_loader, num_batches = create_torch_dataloader(
        data_config,
        stats_model_config.action_horizon,
        batch_size,
        stats_model_config,
        num_workers,
        max_frames,
    )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}
    expected_frames = num_batches * batch_size
    tqdm.tqdm.write(
        f"[{dataset_index}/{total_datasets}] Scanning {_format_count(num_batches)} batch(es)"
        f" (~{_format_count(expected_frames)} frame(s)) for {dataset_name}"
    )

    processed_frames = 0

    with tqdm.tqdm(
        data_loader,
        total=num_batches,
        desc=f"[{dataset_index}/{total_datasets}] {dataset_name}",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
        disable=not show_progress,
    ) as batch_progress:
        for batch in batch_progress:
            for key in keys:
                stats[key].update(np.asarray(batch[key]))

            processed_frames += _batch_size_from_batch(batch, keys)
            if show_progress:
                batch_progress.set_postfix_str(f"frames={_format_count(processed_frames)}")

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    asset_id = data_config.asset_id
    if asset_id is None or not isinstance(asset_id, str):
        raise ValueError(f"Expected a resolved string asset_id for {data_config.repo_id}, got {asset_id!r}.")

    output_path = config.assets_dirs / _config.normalize_asset_id(asset_id)
    tqdm.tqdm.write(f"[{dataset_index}/{total_datasets}] Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)
    normalize.save_contract(output_path, data_config.normalization_contract)
    tqdm.tqdm.write(
        f"[{dataset_index}/{total_datasets}] Finished {dataset_name}: "
        f"{_format_count(processed_frames)} frame(s) processed"
    )


def main(
    config_name: str,
    max_frames: int | None = None,
    num_workers: int | None = None,
    batch_size: int | None = None,
):
    config, stats_model_config, effective_num_workers, effective_batch_size, expanded_data_configs = (
        _resolve_norm_stats_inputs(
            config_name,
            num_workers=num_workers,
            batch_size=batch_size,
        )
    )

    total_datasets = len(expanded_data_configs)
    tqdm.tqdm.write(
        f"Resolved {total_datasets} dataset(s) from config '{config_name}'"
        + (f" with max_frames={_format_count(max_frames)}" if max_frames is not None else "")
        + f"; batch_size={effective_batch_size}; torch_num_workers={effective_num_workers}"
    )

    dataset_progress = tqdm.tqdm(
        range(1, total_datasets + 1),
        total=total_datasets,
        desc="Datasets",
        unit="dataset",
        dynamic_ncols=True,
    )

    for dataset_position, current_dataset_index in enumerate(dataset_progress, start=1):
        data_config = expanded_data_configs[current_dataset_index - 1]
        dataset_name = _dataset_label(data_config)
        dataset_progress.set_description(f"Datasets [{dataset_position}/{total_datasets}]")
        dataset_progress.set_postfix_str(f"{dataset_name}")
        compute_dataset_norm_stats(
            config=config,
            data_config=data_config,
            stats_model_config=stats_model_config,
            dataset_index=current_dataset_index,
            total_datasets=total_datasets,
            max_frames=max_frames,
            num_workers=effective_num_workers,
            batch_size=effective_batch_size,
            show_progress=True,
        )


if __name__ == "__main__":
    tyro.cli(main)
