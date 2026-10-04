import dataclasses
import importlib.util
import pathlib
import types

import numpy as np
import pytest

from openpi.models import pi0_config
from openpi.shared import normalize
from openpi.training import base_checkpoints
from openpi.training import config as _config
import openpi.transforms as _transforms

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]


def _load_script(module_name: str, relative_path: str):
    path = _REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalize_asset_id_strips_absolute_prefix():
    assert _config.normalize_asset_id("/data/my_robot") == "data/my_robot"
    assert _config.normalize_asset_id("my_robot") == "my_robot"


def test_absolute_local_repo_loads_norm_stats_under_config_assets(tmp_path):
    repo = tmp_path / "my_robot"
    (repo / "meta").mkdir(parents=True)
    (repo / "meta" / "info.json").write_text("{}", encoding="utf-8")
    (repo / "data").mkdir()

    stats = normalize.RunningStats()
    stats.update(np.arange(8, dtype=np.float32).reshape(2, 4))
    norm_stats = {"state": stats.get_statistics()}
    assets_dir = tmp_path / "assets"
    normalize.save(assets_dir / _config.normalize_asset_id(str(repo.resolve())), norm_stats)

    @dataclasses.dataclass(frozen=True)
    class _RepoFactory(_config.DataConfigFactory):
        def create(self, assets_dirs: pathlib.Path, model_config):
            return self.create_base_config(assets_dirs, model_config)

    factory = _RepoFactory(repo_id=str(repo), load_norm_stats=True)
    data_config = factory.create_base_config(assets_dir, pi0_config.Pi0Config(enable_world_model=False))

    assert pathlib.Path(str(data_config.repo_id)).is_absolute()
    assert data_config.norm_stats is not None
    assert np.allclose(data_config.norm_stats["state"].mean, norm_stats["state"].mean)


def test_converted_libero_data_matches_the_reference_converter():
    factory = _config.converted_libero_data("data/custom_libero", "custom_libero")

    assert factory.repo_id == "data/custom_libero"
    assert factory.assets.asset_id == "custom_libero"
    assert factory.dataset_action_gripper_format == "signed_command"
    assert factory.treat_actions_as_commands is True
    assert factory.canonicalize_ee_pose_gripper is True
    assert factory.use_canonical_ee_delta is True
    assert factory.base_image_key == "observation.images.image"
    assert factory.base_config is not None
    assert factory.base_config.world_model.image_keys == ("observation.images.image",)


def test_libero_world_model_uses_base_camera_when_image_keys_are_omitted(tmp_path):
    factory = _config.LeRobotLiberoDataConfig(
        repo_id="fake_libero",
        base_config=_config.DataConfig(world_model=_config.WorldModelDataConfig()),
    )
    data_config = factory.create(tmp_path, pi0_config.Pi0Config(enable_world_model=True))

    assert data_config.world_model.image_keys == ("observation.images.image",)
    assert data_config.action_sequence_keys == ("action",)


def test_libero_contract_is_read_from_the_normalized_asset_directory(tmp_path):
    repo = tmp_path / "custom_libero"
    (repo / "meta").mkdir(parents=True)
    (repo / "data").mkdir()
    (repo / "meta" / "info.json").write_text("{}", encoding="utf-8")
    (repo / "meta" / "stats.json").write_text(
        '{"observation.state": {"q01": [0, 0, 0, 0, 0, 0, 0, 0.001], "q99": [0, 0, 0, 0, 0, 0, 0, 0.04]},'
        ' "action": {"q01": [0, 0, 0, 0, 0, 0, -1], "q99": [0, 0, 0, 0, 0, 0, 1]}}',
        encoding="utf-8",
    )
    stats = normalize.RunningStats()
    stats.update(np.arange(8, dtype=np.float32).reshape(2, 4))
    assets_dir = tmp_path / "assets"
    stats_dir = assets_dir / _config.normalize_asset_id(str(repo.resolve()))
    normalize.save(stats_dir, {"state": stats.get_statistics()})
    normalize.save_contract(
        stats_dir,
        "libero_eef_v2:state=physical_width:action=signed_command:"
        "canonical_state=open_fraction:canonical_action=ee_delta",
    )

    data_config = _config.LeRobotLiberoDataConfig(
        repo_id=str(repo),
        canonicalize_ee_pose_gripper=True,
        treat_actions_as_commands=True,
        dataset_action_gripper_format="signed_command",
        use_canonical_ee_delta=True,
    ).create(assets_dir, pi0_config.Pi0Config(enable_world_model=False))

    assert data_config.norm_stats is not None


def test_local_dataset_is_opened_from_its_own_root(tmp_path, monkeypatch):
    import openpi.training.data_loader as data_loader

    repo = tmp_path / "robot"
    (repo / "meta").mkdir(parents=True)
    (repo / "meta" / "info.json").write_text("{}", encoding="utf-8")
    (repo / "data").mkdir()
    seen: dict[str, object] = {}

    class _FakeLeRobotDataset:
        def __init__(self, repo_id, root=None, **kwargs):
            del kwargs
            seen["repo_id"] = repo_id
            seen["root"] = root

        def __len__(self):
            return 1

        def __getitem__(self, index):
            del index
            return {}

    monkeypatch.setattr(data_loader, "_apply_lerobot_startup_patches", lambda: None)
    monkeypatch.setattr(data_loader, "_default_lerobot_video_backend", lambda: "pyav")
    monkeypatch.setattr(
        data_loader,
        "_load_startup_metadata",
        lambda repo_id: types.SimpleNamespace(fps=10.0, tasks=None),
    )
    monkeypatch.setattr(data_loader.lerobot_dataset, "LeRobotDataset", _FakeLeRobotDataset)

    data_loader.create_torch_dataset(
        _config.DataConfig(repo_id=str(repo), action_sequence_keys=("action",)),
        action_horizon=1,
        model_config=pi0_config.Pi0Config(enable_world_model=False, action_horizon=1),
    )

    assert seen["root"] == repo.resolve()


def test_reference_converter_matches_libero_data_config_defaults():
    converter = _load_script("convert_libero_data_to_lerobot", "examples/libero/convert_libero_data_to_lerobot.py")
    factory = _config.LeRobotLiberoDataConfig(repo_id="local_or_hub_id")

    assert factory.base_image_key in converter.LEROBOT_FEATURES
    assert factory.wrist_image_key in converter.LEROBOT_FEATURES
    assert converter.LEROBOT_FEATURES["observation.state"]["shape"] == (8,)
    assert converter.LEROBOT_FEATURES["action"]["shape"] == (7,)
    assert factory.dataset_action_gripper_format == "signed_command"


def test_unknown_base_checkpoint_names_the_registry():
    with pytest.raises(KeyError, match="pi05_base"):
        base_checkpoints.get_base_checkpoint("not_a_checkpoint")


def test_download_assets_converts_the_selected_pi05_checkpoint(tmp_path, monkeypatch):
    download_assets = _load_script("download_assets_script", "scripts/download_assets.py")
    monkeypatch.setattr(download_assets, "REPO_ROOT", tmp_path)
    monkeypatch.setitem(
        base_checkpoints.BASE_CHECKPOINTS,
        "pi05_alt",
        base_checkpoints.BaseCheckpoint(
            name="pi05_alt",
            jax_uri="gs://example/pi05_alt",
            pytorch_dirname="pi05_alt_pytorch",
            pi05=True,
        ),
    )
    calls: list[list[str]] = []

    def fake_run(cmd, cwd, check):
        del cwd, check
        calls.append(list(cmd))
        output = tmp_path / "checkpoints" / "pi05_alt_pytorch"
        output.mkdir(parents=True, exist_ok=True)
        (output / "model.safetensors").write_bytes(b"weights")

    monkeypatch.setattr(download_assets.subprocess, "run", fake_run)

    written = download_assets.ensure_base_checkpoint("pi05_alt")

    assert written == tmp_path / "checkpoints" / "pi05_alt_pytorch"
    assert calls[0][calls[0].index("--checkpoint-dir") + 1] == "gs://example/pi05_alt"
    assert calls[0][calls[0].index("--config-name") + 1] == "stage1_world_model_pretraining"
    assert calls[0][calls[0].index("--output-path") + 1] == str(written)
    download_assets.ensure_base_checkpoint("pi05_alt")
    assert len(calls) == 1


def test_download_assets_rejects_a_non_pi05_checkpoint(tmp_path, monkeypatch):
    download_assets = _load_script("download_assets_script_reject", "scripts/download_assets.py")
    monkeypatch.setattr(download_assets, "REPO_ROOT", tmp_path)
    monkeypatch.setitem(
        base_checkpoints.BASE_CHECKPOINTS,
        "pi0_base",
        base_checkpoints.BaseCheckpoint(
            name="pi0_base",
            jax_uri="gs://example/pi0_base",
            pytorch_dirname="pi0_base_pytorch",
            pi05=False,
        ),
    )

    with pytest.raises(ValueError, match="not a"):
        download_assets.ensure_base_checkpoint("pi0_base")


class _StateAction(_transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {
            "state": np.asarray(data["observation.state"], dtype=np.float32),
            "actions": np.asarray(data["action"], dtype=np.float32),
        }


def test_local_lerobot_dataset_norm_stats_land_in_config_assets(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset_dir = tmp_path / "custom_robot"
    dataset = LeRobotDataset.create(
        repo_id="custom_robot",
        root=dataset_dir,
        fps=10,
        robot_type="custom",
        use_videos=False,
        features={
            "observation.state": {"dtype": "float32", "shape": (4,), "names": ["state"]},
            "action": {"dtype": "float32", "shape": (4,), "names": ["action"]},
        },
    )
    for step in range(8):
        value = np.full(4, step, dtype=np.float32)
        dataset.add_frame({"observation.state": value, "action": value + 1, "task": "pick"})
    dataset.save_episode()
    dataset.finalize()

    compute_norm_stats = _load_script("compute_norm_stats_script", "scripts/compute_norm_stats.py")
    train_config = _config.TrainConfig(
        name="custom_robot",
        exp_name="custom_robot",
        assets_base_dir=str(tmp_path / "assets"),
        data=_config.FakeDataConfig(),
        batch_size=8,
        num_workers=0,
        wandb_enabled=False,
        model=pi0_config.Pi0Config(action_horizon=1, enable_world_model=False),
    )
    data_config = _config.DataConfig(
        repo_id=str(dataset_dir),
        asset_id="custom_robot",
        action_sequence_keys=("action",),
        data_transforms=_transforms.Group(inputs=[_StateAction()]),
    )

    compute_norm_stats.compute_dataset_norm_stats(
        config=train_config,
        data_config=data_config,
        stats_model_config=train_config.model,
        dataset_index=1,
        total_datasets=1,
        max_frames=None,
        num_workers=0,
        batch_size=8,
        show_progress=False,
    )

    stats_path = tmp_path / "assets" / "custom_robot" / "custom_robot" / "norm_stats.json"
    assert stats_path.is_file()
    loaded = normalize.load(stats_path.parent)
    assert loaded["state"].mean.shape == (4,)
    assert loaded["actions"].mean.shape == (4,)
