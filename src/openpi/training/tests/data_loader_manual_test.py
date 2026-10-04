import dataclasses
import os

import pytest

from openpi.models import model as _model
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def _iter_data_configs(train_config: _config.TrainConfig) -> list[_config.DataConfig]:
    return [train_config.data.create(train_config.assets_dirs, train_config.model)]


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


def _maybe_limit_repo_ids_for_smoke_test(data_config: _config.DataConfig) -> _config.DataConfig:
    max_repo_ids = os.getenv("PLAW_VLA_MANUAL_TEST_MAX_REPO_IDS_PER_CONFIG")
    if not max_repo_ids:
        return data_config

    repo_id = data_config.repo_id
    if not isinstance(repo_id, list):
        return data_config

    limit = int(max_repo_ids)
    if limit <= 0 or len(repo_id) <= limit:
        return data_config

    return dataclasses.replace(
        data_config,
        repo_id=repo_id[:limit],
        asset_id=repo_id[:limit] if isinstance(data_config.asset_id, list) else data_config.asset_id,
    )


@pytest.mark.manual
def test_can_load_dataset_for_named_config(dataset_config_name: str | None) -> None:
    if not dataset_config_name:
        pytest.skip("Pass --dataset-config-name=<config> to run this manual dataset-load test.")

    train_config = _config.get_config(dataset_config_name)
    data_configs = _iter_data_configs(train_config)

    assert data_configs, f"No data configs were created for {dataset_config_name!r}."

    samples = []
    for data_config in data_configs:
        test_data_config = _maybe_limit_repo_ids_for_smoke_test(data_config)
        sample = _load_one_sample(
            test_data_config,
            action_horizon=train_config.model.action_horizon,
            model_config=train_config.model,
        )
        assert sample is not None, f"Loaded an empty sample for repo_id={test_data_config.repo_id!r}."
        samples.append(sample)

    assert len(samples) == len(data_configs)
