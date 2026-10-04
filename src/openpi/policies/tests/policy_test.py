import pathlib
import types

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


def test_create_trained_policy_defaults_wm_inference_future_frames_from_data_config(monkeypatch, tmp_path: pathlib.Path):
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.safetensors").write_text("", encoding="utf-8")

    fake_model = types.SimpleNamespace(
        enable_world_model=True,
        wm_inference_num_future_frames=None,
        paligemma_with_expert=types.SimpleNamespace(
            to_bfloat16_for_selected_params=lambda dtype: None,
        ),
    )

    class _FakeModelConfig:
        def load_pytorch(self, train_config, weight_path: str):
            return fake_model

    train_config = types.SimpleNamespace(
        model=_FakeModelConfig(),
        data=types.SimpleNamespace(
            create=lambda assets_dirs, model_cfg: _config.DataConfig(
                asset_id="dummy",
                world_model=_config.WorldModelDataConfig(history_num_frames=3, future_num_frames=5),
            )
        ),
        assets_dirs=tmp_path,
        policy_metadata=None,
    )

    monkeypatch.setattr(_policy_config.download, "maybe_download", lambda path: str(path))
    monkeypatch.setattr(
        _policy_config,
        "_policy",
        types.SimpleNamespace(Policy=lambda model, **kwargs: types.SimpleNamespace(model=model, kwargs=kwargs)),
    )

    policy = _policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        norm_stats={},
        pytorch_device="cpu",
    )

    assert policy.model.wm_inference_num_future_frames == 5


