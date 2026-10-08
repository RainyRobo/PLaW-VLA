import pytest

from plawvla.models import model as _model
from plawvla.training import config as _config
from plawvla.training import data_loader as _data_loader

# This manual integration test intentionally expands grouped configs through
# the loader's internal helper so every local leaf dataset is exercised.
# ruff: noqa: SLF001


def _iter_data_configs(train_config: _config.TrainConfig) -> list[_config.DataConfig]:
    if isinstance(train_config.data, _config.MultiDatasetPretrainDataConfig):
        root_configs = [
            data_config
            for data_config, _weight in train_config.data.create_all(train_config.assets_dirs, train_config.model)
        ]
    else:
        root_configs = [train_config.data.create(train_config.assets_dirs, train_config.model)]

    data_configs = []
    for data_config in root_configs:
        data_configs.extend(_data_loader._expand_data_config_children(data_config))
    return data_configs


def _load_one_sample(
    data_config: _config.DataConfig,
    *,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
) -> object:
    if data_config.repo_id is None:
        pytest.fail("Selected config does not define a dataset source. Use a training config with repo_id.")

    dataset = _data_loader.create_torch_dataset(
        data_config,
        action_horizon=action_horizon,
        model_config=model_config,
    )
    return dataset[0]


@pytest.mark.manual
def test_can_load_dataset_for_named_config(dataset_config_name: str | None) -> None:
    if not dataset_config_name:
        pytest.skip("Pass --dataset-config-name=<config> to run this manual dataset-load test.")

    train_config = _config.get_config(dataset_config_name)
    data_configs = _iter_data_configs(train_config)

    assert data_configs, f"No data configs were created for {dataset_config_name!r}."

    samples = []
    for data_config in data_configs:
        sample = _load_one_sample(
            data_config,
            action_horizon=train_config.model.action_horizon,
            model_config=train_config.model,
        )
        assert sample is not None, f"Loaded an empty sample for repo_id={data_config.repo_id!r}."
        samples.append(sample)

    assert len(samples) == len(data_configs)
