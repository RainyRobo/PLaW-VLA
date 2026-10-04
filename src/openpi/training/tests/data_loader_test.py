import dataclasses
from pathlib import Path
import warnings

import jax
from lerobot.datasets import video_utils as lerobot_video_utils
import numpy as np
import pytest
import torch

from openpi.models import pi0_config
from openpi.policies import libero_policy
import openpi.transforms as _transforms
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def _mesh_compatible_batch_size(requested: int) -> int:
    shard_count = max(1, jax.local_device_count())
    return max(shard_count, ((requested + shard_count - 1) // shard_count) * shard_count)


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    local_batch_size = _mesh_compatible_batch_size(4)
    dataset = _data_loader.FakeDataset(config, local_batch_size * 4)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == local_batch_size for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    local_batch_size = _mesh_compatible_batch_size(4)
    dataset = _data_loader.FakeDataset(config, local_batch_size)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=local_batch_size)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


class _IndexDataset(torch.utils.data.Dataset):
    def __init__(self, size: int):
        self._size = size

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        return {"index": np.int64(index)}


def _batch_indices(batch) -> list[int]:
    values = batch["index"] if isinstance(batch, dict) else batch
    return [int(value) for value in np.asarray(values).reshape(-1)]


def test_distributed_sampler_changes_order_on_each_dataset_pass():
    dataset = _IndexDataset(16)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=2,
        rank=0,
        shuffle=True,
        drop_last=True,
        seed=0,
    )
    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        sampler=sampler,
        num_workers=0,
        framework="pytorch",
        seed=0,
    )
    assert len(loader) == 2

    data_iter = iter(loader)
    first_pass = _batch_indices(next(data_iter)) + _batch_indices(next(data_iter))
    second_pass = _batch_indices(next(data_iter)) + _batch_indices(next(data_iter))

    assert sampler.epoch == 1
    assert len(first_pass) == len(second_pass) == 8
    assert len(set(first_pass)) == 8
    assert first_pass != second_pass


def test_data_loader_set_epoch_selects_the_resume_pass():
    dataset = _IndexDataset(16)
    resumed = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=2,
        rank=0,
        shuffle=True,
        drop_last=True,
        seed=0,
    )
    baseline = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=2,
        rank=0,
        shuffle=True,
        drop_last=True,
        seed=0,
    )
    torch_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        sampler=resumed,
        num_workers=0,
        framework="pytorch",
        seed=0,
    )
    data_config = _config.DataConfig(repo_id="index", asset_id="index")
    loader = _data_loader.DataLoaderImpl(data_config, torch_loader)
    loader.set_epoch(3)
    baseline.set_epoch(3)

    resumed_indices = [int(index) for index in resumed]
    assert resumed_indices == [int(index) for index in baseline]
    assert len(loader) == 2


def test_hf_column_sequence_compat_keeps_string_key_access_lazy():
    class _ExplodingColumn:
        def __iter__(self):
            raise AssertionError("column iteration should stay lazy")

        def __getitem__(self, key):
            return key

    class _Dataset:
        def __getitem__(self, key):
            if key == "image":
                return _ExplodingColumn()
            return {"key": key}

        def __len__(self):
            return 1

    wrapped = _data_loader._HFColumnSequenceCompat(_Dataset())  # noqa: SLF001

    column = wrapped["image"]

    assert isinstance(column, _ExplodingColumn)
    assert column[[0, 2]] == [0, 2]


class _FakeChoiceRng:
    def __init__(self, outputs: list[int]):
        self._outputs = outputs

    def choice(self, candidates, p=None, size=None, replace=True):
        del p
        del replace
        if not self._outputs:
            raise AssertionError("Ran out of fake RNG outputs")
        if size is None:
            value = self._outputs.pop(0)
            if value not in set(np.asarray(candidates).tolist()):
                raise AssertionError(f"Requested value {value} not present in candidates {candidates}")
            return value

        values = [self._outputs.pop(0) for _ in range(size)]
        if np.isscalar(candidates):
            candidate_set = set(range(int(candidates)))
        else:
            candidate_set = set(np.asarray(candidates).tolist())
        if any(value not in candidate_set for value in values):
            raise AssertionError(f"Requested values {values} not present in candidates {candidates}")
        return np.asarray(values)


def test_resolve_sampled_frame_count_supports_weighted_choice():
    sampled_count = _data_loader._resolve_sampled_frame_count(
        1,
        4,
        _FakeChoiceRng([1]),
        name="history_num_frames",
        sample_power=1.0,
    )

    assert sampled_count == 1


def test_maybe_apply_world_model_sampling_creates_prefix_and_suffix_selection_masks():
    batch = {
        "image": {
            "base_0_rgb_history": np.zeros((2, 4, 8, 8, 3), dtype=np.uint8),
            "base_0_rgb_future": np.zeros((2, 3, 8, 8, 3), dtype=np.uint8),
        },
        "image_mask": {
            "base_0_rgb_history": np.ones((2, 4), dtype=bool),
            "base_0_rgb_future": np.ones((2, 3), dtype=bool),
        },
    }
    config = _config.DataConfig(
        world_model=_config.WorldModelDataConfig(
            train_min_history_num_frames=1,
            train_min_future_num_frames=1,
            train_history_sample_power=1.0,
            train_future_sample_power=1.0,
        ),
        world_model_transforms=_transforms.Group(inputs=[lambda data: data]),
    )

    result = _data_loader._maybe_apply_world_model_sampling(batch, config, _FakeChoiceRng([2, 1]))

    assert result["image"]["base_0_rgb_history"].shape == (2, 4, 8, 8, 3)
    assert result["image"]["base_0_rgb_future"].shape == (2, 3, 8, 8, 3)
    np.testing.assert_array_equal(
        result["image_mask"]["base_0_rgb_history_selection"],
        np.array([[False, False, True, True], [False, False, True, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        result["image_mask"]["base_0_rgb_future_selection"],
        np.array([[True, False, False], [True, False, False]], dtype=bool),
    )


def test_data_loader_impl_re_tensorizes_world_model_selection_masks(monkeypatch):
    class _StubLoader:
        def __iter__(self):
            yield {
                "state": torch.zeros((2, 8), dtype=torch.float32),
                "tokenized_prompt": torch.zeros((2, 4), dtype=torch.int64),
                "tokenized_prompt_mask": torch.ones((2, 4), dtype=torch.bool),
                "token_ar_mask": torch.zeros((2, 4), dtype=torch.int64),
                "token_loss_mask": torch.zeros((2, 4), dtype=torch.bool),
                "image": {
                    "base_0_rgb": torch.zeros((2, 224, 224, 3), dtype=torch.float32),
                    "left_wrist_0_rgb": torch.zeros((2, 224, 224, 3), dtype=torch.float32),
                    "right_wrist_0_rgb": torch.zeros((2, 224, 224, 3), dtype=torch.float32),
                    "base_0_rgb_history": torch.zeros((2, 4, 8, 8, 3), dtype=torch.float32),
                    "base_0_rgb_future": torch.zeros((2, 3, 8, 8, 3), dtype=torch.float32),
                },
                "image_mask": {
                    "base_0_rgb": torch.ones((2,), dtype=torch.bool),
                    "left_wrist_0_rgb": torch.ones((2,), dtype=torch.bool),
                    "right_wrist_0_rgb": torch.ones((2,), dtype=torch.bool),
                    "base_0_rgb_history": torch.ones((2, 4), dtype=torch.bool),
                    "base_0_rgb_future": torch.ones((2, 3), dtype=torch.bool),
                },
                "actions": torch.zeros((2, 3, 8), dtype=torch.float32),
            }

    config = _config.DataConfig(
        world_model=_config.WorldModelDataConfig(
            train_min_history_num_frames=1,
            train_min_future_num_frames=1,
            train_history_sample_power=1.0,
            train_future_sample_power=1.0,
        ),
        world_model_transforms=_transforms.Group(inputs=[lambda data: data]),
    )

    monkeypatch.setattr(
        _data_loader,
        "_resolve_sampled_frame_count",
        lambda min_count, max_count, rng, *, name, sample_power=0.0: max_count - 1,
    )

    loader = _data_loader.DataLoaderImpl(config, _StubLoader(), seed=0)
    observation, actions = next(iter(loader))

    assert isinstance(observation.image_masks["base_0_rgb_history_selection"], torch.Tensor)
    assert isinstance(observation.image_masks["base_0_rgb_future_selection"], torch.Tensor)
    assert observation.image_masks["base_0_rgb_history_selection"].dtype == torch.bool
    assert observation.image_masks["base_0_rgb_future_selection"].dtype == torch.bool
    assert isinstance(actions, torch.Tensor)

def test_torch_data_loader_parallel(monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    local_batch_size = _mesh_compatible_batch_size(4)
    dataset = _data_loader.FakeDataset(config, local_batch_size * 3)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=local_batch_size, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == local_batch_size for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")
    config = dataclasses.replace(config, batch_size=_mesh_compatible_batch_size(config.batch_size))

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_default_lerobot_video_backend_prefers_env_override(monkeypatch):
    monkeypatch.setenv("PLAW_VLA_LEROBOT_VIDEO_BACKEND", "pyav")

    assert _data_loader._default_lerobot_video_backend() == "pyav"


def test_default_lerobot_video_backend_prefers_pyav_when_unset(monkeypatch):
    monkeypatch.delenv("PLAW_VLA_LEROBOT_VIDEO_BACKEND", raising=False)
    monkeypatch.setattr(_data_loader, "_torchcodec_decoder_available", lambda: True)

    assert _data_loader._default_lerobot_video_backend() == "pyav"


def test_default_lerobot_video_backend_falls_back_to_pyav(monkeypatch):
    monkeypatch.delenv("PLAW_VLA_LEROBOT_VIDEO_BACKEND", raising=False)
    monkeypatch.setattr(_data_loader, "_torchcodec_decoder_available", lambda: False)

    assert _data_loader._default_lerobot_video_backend() == "pyav"


def test_decode_video_frames_with_torchcodec_fallback_retries_with_pyav(monkeypatch):
    calls: list[str] = []

    def fake_decode(video_path, timestamps, tolerance_s, backend):
        del video_path, timestamps, tolerance_s
        calls.append(backend)
        if backend == "torchcodec":
            raise RuntimeError("Invalid frame index=10 for streamIndex=0 numFrames=9")
        return "decoded-with-pyav"

    monkeypatch.setattr(_data_loader, "_ORIG_LEROBOT_DECODE_VIDEO_FRAMES", fake_decode)

    result = _data_loader._decode_video_frames_with_torchcodec_fallback(
        "video.mp4",
        [0.1, 0.2],
        0.05,
        "torchcodec",
    )

    assert result == "decoded-with-pyav"
    assert calls == ["torchcodec", "pyav"]


def test_install_torchvision_video_warning_filter(monkeypatch):
    recorded = {}

    def fake_filterwarnings(*args, **kwargs):
        recorded["args"] = args
        recorded["kwargs"] = kwargs

    monkeypatch.setattr(warnings, "filterwarnings", fake_filterwarnings)
    monkeypatch.setattr(_data_loader, "_LEROBOT_VIDEO_WARNING_FILTER_ACTIVE", False)

    _data_loader._install_torchvision_video_warning_filter()

    assert recorded["args"] == ("ignore",)
    assert recorded["kwargs"]["category"] is UserWarning
    assert "torchvision are deprecated" in recorded["kwargs"]["message"]
    assert recorded["kwargs"]["module"] == r"torchvision\.io\._video_deprecation_warning"


def test_fault_tolerant_dataset_skips_retryable_bad_sample_and_logs(monkeypatch, tmp_path):
    log_path = tmp_path / "bad_samples.log"
    monkeypatch.setenv("PLAW_VLA_BAD_SAMPLE_LOG_PATH", str(log_path))
    monkeypatch.setattr(_data_loader, "_BAD_SAMPLE_LOGGED_SIGNATURES", set())
    monkeypatch.setattr(_data_loader, "_BAD_SAMPLE_INDEX_BLACKLIST", {})
    monkeypatch.setattr(_data_loader, "_BAD_VIDEO_PATH_BLACKLIST", set())
    monkeypatch.setattr(_data_loader, "_BAD_SAMPLE_LOG_SYNC_STATE", {})
    monkeypatch.setattr(_data_loader.os, "getpid", lambda: 0)

    class _Dataset:
        def __getitem__(self, index):
            if index == 0:
                raise lerobot_video_utils.FrameTimestampError("video decode failed\nvideo: /tmp/bad.mp4")
            return {"index": index}

        def __len__(self):
            return 4

    dataset = _data_loader.FaultTolerantDataset(_Dataset(), repo_id="repo", max_retries=3)

    result = dataset[0]

    assert result == {"index": 1}
    log_text = log_path.read_text(encoding="utf-8")
    assert "repo_id=repo" in log_text
    assert "failed_index=0" in log_text
    assert "FrameTimestampError" in log_text


def test_fault_tolerant_dataset_propagates_non_retryable_errors(monkeypatch):
    monkeypatch.setattr(_data_loader, "_BAD_SAMPLE_LOGGED_SIGNATURES", set())
    monkeypatch.setattr(_data_loader, "_BAD_SAMPLE_INDEX_BLACKLIST", {})
    monkeypatch.setattr(_data_loader, "_BAD_VIDEO_PATH_BLACKLIST", set())
    monkeypatch.setattr(_data_loader, "_BAD_SAMPLE_LOG_SYNC_STATE", {})

    class _Dataset:
        def __getitem__(self, index):
            del index
            raise KeyError("not a video failure")

        def __len__(self):
            return 4

    dataset = _data_loader.FaultTolerantDataset(_Dataset(), repo_id="repo", max_retries=1)

    with pytest.raises(KeyError, match="not a video failure"):
        _ = dataset[0]


def test_fault_tolerant_dataset_can_skip_any_read_error_when_enabled(monkeypatch, tmp_path):
    log_path = tmp_path / "bad_samples.log"
    monkeypatch.setenv("PLAW_VLA_BAD_SAMPLE_LOG_PATH", str(log_path))
    monkeypatch.setattr(_data_loader, "_BAD_SAMPLE_LOGGED_SIGNATURES", set())
    monkeypatch.setattr(_data_loader, "_BAD_SAMPLE_INDEX_BLACKLIST", {})
    monkeypatch.setattr(_data_loader, "_BAD_VIDEO_PATH_BLACKLIST", set())
    monkeypatch.setattr(_data_loader, "_BAD_SAMPLE_LOG_SYNC_STATE", {})
    monkeypatch.setattr(_data_loader.os, "getpid", lambda: 0)

    class _Dataset:
        def __getitem__(self, index):
            if index == 0:
                raise KeyError("broken parquet row")
            return {"index": index}

        def __len__(self):
            return 4

    dataset = _data_loader.FaultTolerantDataset(
        _Dataset(),
        repo_id="repo",
        max_retries=3,
        catch_all_errors=True,
    )

    result = dataset[0]

    assert result == {"index": 1}
    log_text = log_path.read_text(encoding="utf-8")
    assert "repo_id=repo" in log_text
    assert "KeyError" in log_text
    assert "broken parquet row" in log_text


def test_fault_tolerant_dataset_blacklists_bad_video_paths(monkeypatch, tmp_path):
    log_path = tmp_path / "bad_samples.log"
    monkeypatch.setenv("PLAW_VLA_BAD_SAMPLE_LOG_PATH", str(log_path))
    monkeypatch.setattr(_data_loader, "_BAD_SAMPLE_LOGGED_SIGNATURES", set())
    monkeypatch.setattr(_data_loader, "_BAD_SAMPLE_INDEX_BLACKLIST", {})
    monkeypatch.setattr(_data_loader, "_BAD_VIDEO_PATH_BLACKLIST", set())
    monkeypatch.setattr(_data_loader, "_BAD_SAMPLE_LOG_SYNC_STATE", {})
    monkeypatch.setattr(_data_loader.os, "getpid", lambda: 0)

    class _Meta:
        video_keys = ("observation.images.cam_high",)

        def get_video_file_path(self, ep_idx, vid_key):
            del vid_key
            return "bad.mp4" if ep_idx in {0, 2} else "good.mp4"

    class _Dataset:
        root = tmp_path
        meta = _Meta()
        hf_dataset = [
            {"episode_index": np.int64(0)},
            {"episode_index": np.int64(1)},
            {"episode_index": np.int64(2)},
        ]

        def __init__(self):
            self.calls = []

        def _ensure_hf_dataset_loaded(self):
            return None

        def __getitem__(self, index):
            self.calls.append(index)
            if index in {0, 2}:
                raise RuntimeError("Invalid data found when processing input")
            return {"index": index}

        def __len__(self):
            return 3

    raw_dataset = _Dataset()
    dataset = _data_loader.FaultTolerantDataset(
        raw_dataset,
        repo_id="repo",
        max_retries=4,
        catch_all_errors=True,
    )

    assert dataset[0] == {"index": 1}
    assert dataset[2] == {"index": 1}
    assert raw_dataset.calls == [0, 1, 1]


def test_sync_bad_sample_blacklist_from_log(monkeypatch, tmp_path):
    log_path = tmp_path / "bad_samples.log"
    log_path.write_text(
        "2026-03-31T00:00:00Z\tpid=1\trepo_id=repo\trequested_index=4\tfailed_index=7\tattempt=0\t"
        "error_type=RuntimeError\tvideo_paths=('/tmp/bad.mp4', '/tmp/other.mp4')\tmessage='broken'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PLAW_VLA_BAD_SAMPLE_LOG_PATH", str(log_path))
    monkeypatch.setattr(_data_loader, "_BAD_SAMPLE_INDEX_BLACKLIST", {})
    monkeypatch.setattr(_data_loader, "_BAD_VIDEO_PATH_BLACKLIST", set())
    monkeypatch.setattr(_data_loader, "_BAD_SAMPLE_LOG_SYNC_STATE", {})

    loaded = _data_loader._sync_bad_sample_blacklist_from_log()

    assert loaded == 1
    assert 7 in _data_loader._BAD_SAMPLE_INDEX_BLACKLIST["repo"]
    assert "/tmp/bad.mp4" in _data_loader._BAD_VIDEO_PATH_BLACKLIST


def test_fault_tolerant_dataset_imports_external_bad_shard_log(monkeypatch, tmp_path):
    log_path = tmp_path / "bad_samples.log"
    bad_shard = str((tmp_path / "bad.mp4").resolve())
    log_path.write_text(
        "2026-03-31T00:00:00Z\tpid=99\trepo_id=repo\trequested_index=0\tfailed_index=0\tattempt=0\t"
        f"error_type=RuntimeError\tvideo_paths=('{bad_shard}',)\tmessage='broken'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PLAW_VLA_BAD_SAMPLE_LOG_PATH", str(log_path))
    monkeypatch.setenv("PLAW_VLA_BAD_SAMPLE_LOG_SYNC_INTERVAL_S", "0")
    monkeypatch.setattr(_data_loader, "_BAD_SAMPLE_LOGGED_SIGNATURES", set())
    monkeypatch.setattr(_data_loader, "_BAD_SAMPLE_INDEX_BLACKLIST", {})
    monkeypatch.setattr(_data_loader, "_BAD_VIDEO_PATH_BLACKLIST", set())
    monkeypatch.setattr(_data_loader, "_BAD_SAMPLE_LOG_SYNC_STATE", {})
    monkeypatch.setattr(_data_loader.os, "getpid", lambda: 0)

    class _Meta:
        video_keys = ("observation.images.cam_high",)

        def get_video_file_path(self, ep_idx, vid_key):
            del vid_key
            return "bad.mp4" if ep_idx in {0, 2} else "good.mp4"

    class _Dataset:
        root = tmp_path
        meta = _Meta()
        hf_dataset = [
            {"episode_index": np.int64(0)},
            {"episode_index": np.int64(1)},
            {"episode_index": np.int64(2)},
        ]

        def __init__(self):
            self.calls = []

        def _ensure_hf_dataset_loaded(self):
            return None

        def __getitem__(self, index):
            self.calls.append(index)
            if index in {0, 2}:
                raise RuntimeError("Invalid data found when processing input")
            return {"index": index}

        def __len__(self):
            return 3

    raw_dataset = _Dataset()
    dataset = _data_loader.FaultTolerantDataset(
        raw_dataset,
        repo_id="repo",
        max_retries=4,
        catch_all_errors=True,
    )

    assert dataset[2] == {"index": 1}
    assert raw_dataset.calls == [1]


def test_data_loader_impl_exposes_all_data_configs_for_checkpoint_assets():
    model_config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(model_config, 8)
    local_batch_size = _mesh_compatible_batch_size(4)
    dataset = _data_loader.FakeDataset(model_config, local_batch_size * 2)
    torch_loader = _data_loader.TorchDataLoader(dataset, local_batch_size=local_batch_size, num_batches=1, num_workers=0)
    data_config_1 = _config.DataConfig(repo_id="dataset_a", asset_id="dataset_a")
    data_config_2 = _config.DataConfig(repo_id="dataset_b", asset_id="dataset_b")

    loader = _data_loader.DataLoaderImpl(
        data_config_1,
        torch_loader,
        data_configs=[data_config_1, data_config_2],
        checkpoint_asset_metadata=[
            {"dataset_type": "libero", "weight": 0.1},
            {"dataset_type": "libero", "weight": 1.0},
        ],
    )

    assert loader.data_config() == data_config_1
    assert list(loader.data_configs()) == [data_config_1, data_config_2]
    assert list(loader.checkpoint_asset_metadata()) == [
        {"dataset_type": "libero", "weight": 0.1},
        {"dataset_type": "libero", "weight": 1.0},
    ]


@pytest.mark.manual
def test_with_real_dataset():
    config = _config.get_config("stage3_finetuning_libero")
    repo_id = config.data.repo_id
    repo_paths = repo_id if isinstance(repo_id, list) else [repo_id]
    if not all(isinstance(path, str) and Path(path).exists() for path in repo_paths):
        pytest.skip("Local real-data test dataset is not available in this environment.")
    config = dataclasses.replace(config, batch_size=_mesh_compatible_batch_size(4))

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    expected_data_config = config.data.create(config.assets_dirs, config.model)
    assert loader.data_config().repo_id == expected_data_config.repo_id
    assert loader.data_config().asset_id == expected_data_config.asset_id

    try:
        batches = list(loader)
    except (RuntimeError, lerobot_video_utils.FrameTimestampError) as exc:
        pytest.skip(f"Local LIBERO fixture is not decodable in this environment: {exc}")

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_create_torch_dataset_eagerly_constructs_single_lerobot_dataset(monkeypatch):
    class _Meta:
        fps = 10
        tasks = {0: "task"}
        episodes = {0: {"length": 3}, 1: {"length": 2}}

    calls = {"count": 0}

    class _Dataset:
        def __getitem__(self, index):
            return {"index": index}

        def __len__(self):
            return 5

    class _FakeLeRobotDataset:
        def __init__(self, *args, **kwargs):
            calls["count"] += 1
            del args, kwargs
            self._dataset = _Dataset()

        def load_hf_dataset(self):
            return self._dataset

        def __getitem__(self, index):
            return self._dataset[index]

        def __len__(self):
            return len(self._dataset)

    monkeypatch.setattr(_data_loader.lerobot_dataset_metadata, "LeRobotDatasetMetadata", lambda repo_id: _Meta())
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDataset", _FakeLeRobotDataset)

    dataset = _data_loader.create_torch_dataset(
        _config.DataConfig(repo_id="repo"),
        action_horizon=4,
        model_config=pi0_config.Pi0Config(),
    )

    assert calls["count"] == 1
    assert len(dataset) == 5
    assert dataset[1] == {"index": 1}


def test_load_nested_dataset_with_default_num_proc_uses_hf_loader_for_data_dir(monkeypatch, tmp_path):
    captured = {"download_and_prepare": None, "as_dataset": None}
    sentinel = object()
    parquet_dir = tmp_path / "data"
    chunk_dir = parquet_dir / "chunk-000"
    chunk_dir.mkdir(parents=True)
    (chunk_dir / "file-000.parquet").touch()

    class _FakeBuilder:
        def __init__(self, **kwargs):
            captured["builder_kwargs"] = kwargs

        def download_and_prepare(self, num_proc):
            captured["download_and_prepare"] = num_proc

        def as_dataset(self, split):
            captured["as_dataset"] = split
            return sentinel

    monkeypatch.setattr(_data_loader, "HFParquetBuilder", _FakeBuilder)

    result = _data_loader._load_nested_dataset_with_default_num_proc(parquet_dir)

    assert result is sentinel
    assert captured["builder_kwargs"]["dataset_name"] == "parquet"
    assert captured["builder_kwargs"]["data_files"] == {"train": [str(chunk_dir / "file-000.parquet")]}
    assert captured["download_and_prepare"] == 32
    assert captured["as_dataset"] == "train"


def test_load_nested_dataset_with_default_num_proc_skips_metadata_dirs(monkeypatch, tmp_path):
    captured = {"called": False}
    sentinel = object()
    parquet_dir = tmp_path / "meta" / "episodes"
    parquet_dir.mkdir(parents=True)

    def fake_orig_loader(pq_dir, features=None, episodes=None):
        captured["called"] = True
        captured["pq_dir"] = pq_dir
        captured["features"] = features
        captured["episodes"] = episodes
        return sentinel

    monkeypatch.setattr(_data_loader, "_ORIG_LEROBOT_LOAD_NESTED_DATASET", fake_orig_loader)

    result = _data_loader._load_nested_dataset_with_default_num_proc(parquet_dir)

    assert result is sentinel
    assert captured["called"] is True
    assert captured["pq_dir"] == parquet_dir


def test_check_cached_episodes_without_video_file_scan_preserves_episode_coverage_check():
    class _FakeHFDataset:
        def __len__(self):
            return 4

        def unique(self, key):
            assert key == "episode_index"
            return [0, 1]

    class _FakeMeta:
        total_episodes = 2

    dataset = type("FakeDataset", (), {})()
    dataset.hf_dataset = _FakeHFDataset()
    dataset.meta = _FakeMeta()
    dataset.episodes = None

    assert _data_loader._check_cached_episodes_without_video_file_scan(dataset) is True


def test_check_cached_episodes_without_video_file_scan_rejects_missing_episode():
    class _FakeHFDataset:
        def __len__(self):
            return 3

        def unique(self, key):
            assert key == "episode_index"
            return [0]

    class _FakeMeta:
        total_episodes = 2

    dataset = type("FakeDataset", (), {})()
    dataset.hf_dataset = _FakeHFDataset()
    dataset.meta = _FakeMeta()
    dataset.episodes = None

    assert _data_loader._check_cached_episodes_without_video_file_scan(dataset) is False


class _IndexDataset:
    def __init__(self, length: int):
        self._length = length

    def __getitem__(self, index):
        return index

    def __len__(self) -> int:
        return self._length


def test_weighted_concat_dataset_requires_one_weight_per_sample():
    with pytest.raises(ValueError, match="Expected 3 sample weights"):
        _data_loader.WeightedConcatDataset([_IndexDataset(3)], [1.0, 2.0])


def test_weighted_concat_dataset_preserves_explicit_sample_weights():
    dataset = _data_loader.WeightedConcatDataset([_IndexDataset(2), _IndexDataset(3)], [0.5, 0.5, 0.1, 0.1, 0.1])
    assert dataset.sample_weights == [0.5, 0.5, 0.1, 0.1, 0.1]
    assert len(dataset) == 5


def test_weighted_group_sampler_draws_parent_groups_by_weight():
    generator = torch.Generator()
    generator.manual_seed(0)
    sampler = _data_loader.WeightedGroupSampler(
        [[2, 3], [5]],
        [0.4, 0.6],
        num_samples=20_000,
        generator=generator,
        chunk_size=1024,
    )

    group_counts = [0, 0]
    for index in sampler:
        if index < 5:
            group_counts[0] += 1
        else:
            group_counts[1] += 1

    total = sum(group_counts)
    assert group_counts[0] / total == pytest.approx(0.4, abs=0.03)
    assert group_counts[1] / total == pytest.approx(0.6, abs=0.03)


def test_weighted_group_sampler_supports_total_frames_above_2_to_24():
    generator = torch.Generator()
    generator.manual_seed(0)
    sampler = _data_loader.WeightedGroupSampler(
        [[2**24 + 10], [7]],
        [0.5, 0.5],
        num_samples=16,
        generator=generator,
        chunk_size=4,
    )

    indices = list(sampler)
    assert len(indices) == 16
    assert all(0 <= index < (2**24 + 17) for index in indices)


def test_weighted_group_sampler_rejects_invalid_group_metadata():
    with pytest.raises(ValueError, match="same length"):
        _data_loader.WeightedGroupSampler([[2]], [0.5, 0.5], num_samples=4)
    with pytest.raises(ValueError, match="at least one dataset group"):
        _data_loader.WeightedGroupSampler([], [], num_samples=4)

def test_load_child_datasets_preserves_input_order(monkeypatch):
    child_configs = [
        _config.DataConfig(repo_id="repo_c"),
        _config.DataConfig(repo_id="repo_a"),
        _config.DataConfig(repo_id="repo_b"),
    ]

    class _Loaded:
        def __init__(self, repo_id: str):
            self.repo_id = repo_id

        def __len__(self):
            return 1

    def fake_create_torch_dataset(data_config, action_horizon, model_config):
        del action_horizon, model_config
        return _Loaded(str(data_config.repo_id))

    def fake_transform_dataset(dataset, data_config, *, skip_norm_stats=False):
        del data_config, skip_norm_stats
        return dataset

    monkeypatch.setattr(_data_loader, "create_torch_dataset", fake_create_torch_dataset)
    monkeypatch.setattr(_data_loader, "transform_dataset", fake_transform_dataset)

    loaded = _data_loader._load_child_datasets(
        child_configs,
        action_horizon=10,
        model_config=pi0_config.Pi0Config(),
        skip_norm_stats=True,
    )

    assert [entry.repo_id for entry in loaded] == ["repo_c", "repo_a", "repo_b"]


def test_create_multi_torch_data_loader_uses_parent_group_sampler(monkeypatch):
    class _Dataset:
        def __init__(self, length: int, label: str):
            self._length = length
            self._label = label

        def __getitem__(self, index):
            return {"value": index, "label": self._label}

        def __len__(self):
            return self._length

    datasets_by_repo = {
        "group_a_child_1": _Dataset(2, "a1"),
        "group_a_child_2": _Dataset(3, "a2"),
        "group_b_child_1": _Dataset(5, "b1"),
    }

    def fake_create_torch_dataset(data_config, action_horizon, model_config):
        del action_horizon, model_config
        return datasets_by_repo[str(data_config.repo_id)]

    def fake_transform_dataset(dataset, data_config, *, skip_norm_stats=False):
        del data_config, skip_norm_stats
        return dataset

    monkeypatch.setattr(_data_loader, "create_torch_dataset", fake_create_torch_dataset)
    monkeypatch.setattr(_data_loader, "transform_dataset", fake_transform_dataset)

    data_configs = [
        (_config.DataConfig(repo_id=["group_a_child_1", "group_a_child_2"]), 0.4),
        (_config.DataConfig(repo_id=["group_b_child_1"]), 0.6),
    ]

    loader = _data_loader.create_multi_torch_data_loader(
        data_configs,
        model_config=pi0_config.Pi0Config(),
        action_horizon=4,
        batch_size=2,
        num_batches=1,
        num_workers=0,
        framework="pytorch",
    )

    torch_loader = loader._data_loader.torch_loader
    assert isinstance(torch_loader.sampler, _data_loader.WeightedGroupSampler)
    assert len(torch_loader.dataset) == 10

class _StaticFactory:
    def __init__(self, data_config: _config.DataConfig):
        self._data_config = data_config

    def create(self, assets_dirs, model_config):
        return self._data_config


def test_create_data_loader_keeps_norm_stats_for_precomputed_actions(monkeypatch, tmp_path):
    captured = {}
    sentinel = object()

    def fake_create_torch_data_loader(data_config, model_config, action_horizon, batch_size, **kwargs):
        captured["skip_norm_stats"] = kwargs["skip_norm_stats"]
        return sentinel

    monkeypatch.setattr(_data_loader, "create_torch_data_loader", fake_create_torch_data_loader)

    config = _config.TrainConfig(
        name="precomputed_actions_with_norm_stats",
        exp_name="test",
        model=pi0_config.Pi0Config(pi05=True, action_dim=48, action_horizon=16),
        data=_StaticFactory(_config.DataConfig(repo_id="fake", action_sequence_keys=(), norm_stats={})),
        assets_base_dir=str(tmp_path),
        num_workers=0,
    )

    loader = _data_loader.create_data_loader(config, skip_norm_stats=False, framework="pytorch")

    assert loader is sentinel
    assert captured["skip_norm_stats"] is False


def test_create_data_loader_skips_norm_stats_when_precomputed_actions_have_no_stats(monkeypatch, tmp_path):
    captured = {}
    sentinel = object()

    def fake_create_torch_data_loader(data_config, model_config, action_horizon, batch_size, **kwargs):
        captured["skip_norm_stats"] = kwargs["skip_norm_stats"]
        return sentinel

    monkeypatch.setattr(_data_loader, "create_torch_data_loader", fake_create_torch_data_loader)

    config = _config.TrainConfig(
        name="precomputed_actions_without_norm_stats",
        exp_name="test",
        model=pi0_config.Pi0Config(pi05=True, action_dim=48, action_horizon=16),
        data=_StaticFactory(_config.DataConfig(repo_id="fake", action_sequence_keys=(), norm_stats=None)),
        assets_base_dir=str(tmp_path),
        num_workers=0,
    )

    loader = _data_loader.create_data_loader(config, skip_norm_stats=False, framework="pytorch")

    assert loader is sentinel
    assert captured["skip_norm_stats"] is True


class _ToyLiberoDataset:
    def __getitem__(self, index):
        del index
        return {
            "observation.images.image": np.zeros((32, 32, 3), dtype=np.uint8),
            "observation.images.wrist_image": np.ones((32, 32, 3), dtype=np.uint8),
            "observation.state": np.arange(8, dtype=np.float32),
            "action": np.arange(10 * 7, dtype=np.float32).reshape(10, 7),
            "prompt": "Push the block into the tray.",
        }

    def __len__(self) -> int:
        return 1


class _ToyBinaryGripperLiberoDataset:
    def __getitem__(self, index):
        del index
        actions = np.zeros((10, 7), dtype=np.float32)
        actions[:, 0] = np.linspace(0.0, 0.9, 10, dtype=np.float32)
        actions[:, 6] = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(10)], dtype=np.float32)
        return {
            "observation.images.image": np.zeros((32, 32, 3), dtype=np.uint8),
            "observation.images.wrist_image": np.ones((32, 32, 3), dtype=np.uint8),
            "observation.state": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, -0.02], dtype=np.float32),
            "action": actions,
            "prompt": "Toggle the gripper.",
        }

    def __len__(self) -> int:
        return 1


def test_libero_transform_dataset_supports_canonical_command_actions_from_singular_key(tmp_path):
    model_cfg = pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=10)
    data_cfg = _config.LeRobotLiberoDataConfig(
        repo_id="fake",
        canonicalize_ee_pose_gripper=True,
        treat_actions_as_commands=True,
    ).create(tmp_path, model_cfg)

    transformed = _data_loader.transform_dataset(_ToyLiberoDataset(), data_cfg, skip_norm_stats=True)
    item = transformed[0]

    raw_state = np.arange(8, dtype=np.float32)
    expected_state = libero_policy._canonicalize_libero_state(raw_state)
    raw_actions = np.arange(10 * 7, dtype=np.float32).reshape(10, 7)
    expected_actions = libero_policy._command_to_libero_absolute_actions(raw_actions, expected_state)

    np.testing.assert_allclose(item["state"][:8], expected_state)
    np.testing.assert_array_equal(item["state"][8:], np.zeros(24, dtype=np.float32))
    np.testing.assert_allclose(item["actions"][:, :8], expected_actions)
    np.testing.assert_array_equal(item["actions"][:, 8:], np.zeros((10, 24), dtype=np.float32))


def test_libero_transform_dataset_supports_binary_gripper_targets(tmp_path):
    model_cfg = pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=10)
    data_cfg = _config.LeRobotLiberoDataConfig(
        repo_id="fake",
        canonicalize_ee_pose_gripper=True,
        treat_actions_as_commands=True,
        dataset_action_gripper_format="binary_target",
    ).create(tmp_path, model_cfg)

    transformed = _data_loader.transform_dataset(_ToyBinaryGripperLiberoDataset(), data_cfg, skip_norm_stats=True)
    item = transformed[0]

    raw_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, -0.02], dtype=np.float32)
    expected_state = libero_policy._canonicalize_libero_state(raw_state)
    raw_actions = np.zeros((10, 7), dtype=np.float32)
    raw_actions[:, 0] = np.linspace(0.0, 0.9, 10, dtype=np.float32)
    raw_actions[:, 6] = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(10)], dtype=np.float32)
    expected_actions = libero_policy._command_to_libero_absolute_actions(
        raw_actions,
        expected_state,
        gripper_actions_are_binary_targets=True,
    )

    np.testing.assert_allclose(item["state"][:8], expected_state)
    np.testing.assert_array_equal(item["state"][8:], np.zeros(24, dtype=np.float32))
    np.testing.assert_allclose(item["actions"][:, :8], expected_actions)
    np.testing.assert_array_equal(item["actions"][:, 8:], np.zeros((10, 24), dtype=np.float32))

