import json

import pytest

from openpi import transforms as _transforms
from openpi.models import pi0_config
from openpi.policies import libero_policy
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def _make_libero_factory() -> _config.LeRobotLiberoDataConfig:
    return _config.LeRobotLiberoDataConfig(
        repo_id="fake_libero",
        base_config=_config.DataConfig(
            world_model=_config.WorldModelDataConfig(
                image_keys=("observation.images.image",),
                history_num_frames=3,
                future_num_frames=2,
                frame_stride=5,
            ),
        ),
    )

def test_world_model_flag_resolves_to_model_enabled(tmp_path):
    model_cfg = pi0_config.Pi0Config(enable_world_model=True)
    factory = _make_libero_factory()
    data_cfg = factory.create(tmp_path, model_cfg)

    assert len(data_cfg.world_model_transforms.inputs) == 1


def test_world_model_flag_resolves_to_model_disabled(tmp_path):
    model_cfg = pi0_config.Pi0Config(enable_world_model=False)
    factory = _make_libero_factory()
    data_cfg = factory.create(tmp_path, model_cfg)

    assert len(data_cfg.world_model_transforms.inputs) == 0


def test_delta_timestamps_and_transforms_follow_model_enabled(tmp_path):
    model_cfg = pi0_config.Pi0Config(enable_world_model=True, action_horizon=10)
    factory = _make_libero_factory()
    data_cfg = factory.create(tmp_path, model_cfg)

    assert len(data_cfg.world_model_transforms.inputs) == 1
    assert isinstance(data_cfg.world_model_transforms.inputs[0], _transforms.SplitTemporalFrames)

    delta = _data_loader._build_delta_timestamps(data_cfg, action_horizon=model_cfg.action_horizon, fps=10)
    assert "observation.images.image" in delta
    assert "action" in delta


def test_delta_timestamps_and_transforms_follow_model_disabled(tmp_path):
    model_cfg = pi0_config.Pi0Config(enable_world_model=False, action_horizon=10)
    factory = _make_libero_factory()
    data_cfg = factory.create(tmp_path, model_cfg)

    assert len(data_cfg.world_model_transforms.inputs) == 0

    delta = _data_loader._build_delta_timestamps(data_cfg, action_horizon=model_cfg.action_horizon, fps=10)
    assert "observation.images.image" not in delta
    assert "action" in delta

def test_world_model_count_based_frame_indices():
    wm_cfg = _config.WorldModelDataConfig(
        history_num_frames=4,
        future_num_frames=3,
        frame_stride=2,
        image_keys=("image",),
    )

    assert wm_cfg.resolve_history_num_frames() == 4
    assert wm_cfg.resolve_future_num_frames() == 3
    assert wm_cfg.resolve_frame_indices() == (-6, -4, -2, 0, 2, 4, 6)


def test_world_model_action_deltas_stay_tied_to_action_horizon(tmp_path):
    model_cfg = pi0_config.Pi0Config(enable_world_model=True, action_horizon=10)
    factory = _config.LeRobotLiberoDataConfig(
        repo_id="fake_libero",
        base_config=_config.DataConfig(
            world_model=_config.WorldModelDataConfig(
                image_keys=("observation.images.image",),
                history_num_frames=4,
                future_num_frames=3,
                frame_stride=2,
            ),
        ),
    )
    data_cfg = factory.create(tmp_path, model_cfg)

    delta = _data_loader._build_delta_timestamps(data_cfg, action_horizon=model_cfg.action_horizon, fps=10)

    assert len(delta["observation.images.image"]) == 7
    assert len(delta["action"]) == model_cfg.action_horizon

def test_custom_action_sequence_offsets_override_default_action_horizon_sampling():
    data_cfg = _config.DataConfig(
        action_sequence_keys=("actions",),
        action_sequence_offsets=(3, 6, 9, 12),
    )

    delta = _data_loader._build_delta_timestamps(data_cfg, action_horizon=4, fps=10)

    assert delta["actions"] == [0.3, 0.6, 0.9, 1.2]


def test_action_time_step_generates_uniform_future_action_times():
    data_cfg = _config.DataConfig(
        action_sequence_keys=("actions",),
        action_time_step_s=0.1,
    )

    delta = _data_loader._build_delta_timestamps(data_cfg, action_horizon=4, fps=50)

    assert delta["actions"] == [0.1, 0.2, 0.30000000000000004, 0.4]


def test_action_time_start_can_preserve_current_and_future_step_style():
    data_cfg = _config.DataConfig(
        action_sequence_keys=("actions",),
        action_time_step_s=0.05,
        action_time_start_s=0.0,
    )

    delta = _data_loader._build_delta_timestamps(data_cfg, action_horizon=4, fps=20)

    assert delta["actions"] == [0.0, 0.05, 0.1, 0.15000000000000002]


def test_explicit_time_offsets_override_fps_dependent_sampling():
    data_cfg = _config.DataConfig(
        action_sequence_keys=("actions",),
        action_sequence_time_offsets_s=(0.1, 0.2, 0.3, 0.4),
        world_model=_config.WorldModelDataConfig(
            image_keys=("image",),
            frame_stride=None,
            time_offsets_s=(-0.6, -0.3, 0.0, 0.3, 0.6),
        ),
        world_model_transforms=_transforms.Group(
            inputs=[_transforms.SplitTemporalFrames(frame_indices=(-2, -1, 0, 1, 2), image_keys=("image",))]
        ),
    )

    delta_20 = _data_loader._build_delta_timestamps(data_cfg, action_horizon=4, fps=20)
    delta_50 = _data_loader._build_delta_timestamps(data_cfg, action_horizon=4, fps=50)

    assert delta_20 == delta_50
    assert delta_20["image"] == [-0.6, -0.3, 0.0, 0.3, 0.6]
    assert delta_20["actions"] == [0.1, 0.2, 0.3, 0.4]


def test_action_time_step_conflicts_with_explicit_action_offsets():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _config.DataConfig(action_time_step_s=0.1, action_sequence_offsets=(1, 2))

    with pytest.raises(ValueError, match="mutually exclusive"):
        _config.DataConfig(action_time_step_s=0.1, action_sequence_time_offsets_s=(0.1, 0.2))

    with pytest.raises(ValueError, match="requires action_time_step_s"):
        _config.DataConfig(action_time_start_s=0.0)


def test_world_model_default_explicit_counts_match_existing_configs():
    wm_cfg = _config.WorldModelDataConfig()

    assert wm_cfg.resolve_frame_indices() == (-10, -5, 0, 5, 10)
    assert wm_cfg.resolve_layout_indices() == (-10, -5, 0, 5, 10)


def test_world_model_explicit_time_offsets_preserve_layout_and_seconds():
    wm_cfg = _config.WorldModelDataConfig(
        frame_stride=None,
        time_offsets_s=(-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    )

    assert wm_cfg.history_num_frames == 6
    assert wm_cfg.future_num_frames == 5
    assert wm_cfg.resolve_frame_indices() == (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5)
    assert wm_cfg.resolve_layout_indices() == (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5)
    assert wm_cfg.resolve_time_offsets(20) == (-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    assert wm_cfg.resolve_time_offsets(50) == (-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0)



def test_released_stage_configs_load():
    names = (
        "stage1_world_model_pretraining",
        "stage2_pretraining",
        "stage3_finetuning_libero",
    )
    for name in names:
        train_cfg = _config.get_config(name)
        assert isinstance(train_cfg.data, _config.LeRobotLiberoDataConfig)
        assert train_cfg.data.repo_id == "data/libero_v3_eef"
        assert train_cfg.data.base_config.world_model.time_offsets_s == _config._LIBERO_TIME_OFFSETS_S
        assert train_cfg.data.base_config.world_model.history_num_frames == 6
        assert train_cfg.data.base_config.world_model.future_num_frames == 6


def test_stage1_world_model_pretraining_uses_static_graph_after_freezing_unused_world_lm_head():
    train_cfg = _config.get_config("stage1_world_model_pretraining")

    assert train_cfg.ddp_find_unused_parameters is False
    assert train_cfg.ddp_static_graph is True
    assert train_cfg.data.pretrain_world_model is True
    assert train_cfg.data.load_norm_stats is False
    assert train_cfg.model.wm_loss_dropout_alpha == 0.0
    assert train_cfg.lr_schedule.decay_lr == train_cfg.lr_schedule.peak_lr


def test_stage2_pretraining_keeps_default_ddp_flags_for_dynamic_wm_dropout():
    train_cfg = _config.get_config("stage2_pretraining")

    assert train_cfg.ddp_find_unused_parameters is True
    assert train_cfg.ddp_static_graph is None
    assert train_cfg.data.use_canonical_ee_delta is True
    assert train_cfg.model.wm_loss_dropout_alpha == 0.3


def test_stage3_libero_matches_stage2_wm_loss_dropout():
    assert _config.get_config("stage3_finetuning_libero").model.wm_loss_dropout_alpha == 0.3


def test_stage3_libero_uses_explicit_time_based_world_model_and_action_schedule():
    train_cfg = _config.get_config("stage3_finetuning_libero")
    data_cfg = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)

    assert data_cfg.action_time_step_s == 0.1
    assert data_cfg.action_time_start_s == 0.0
    assert data_cfg.action_sequence_time_offsets_s is None
    assert data_cfg.world_model.frame_stride is None
    assert data_cfg.world_model.resolve_time_offsets(10) == (
        -1.0,
        -0.8,
        -0.6,
        -0.4,
        -0.2,
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
        1.2,
    )
    assert train_cfg.num_train_steps == 50_000
    assert train_cfg.lr_schedule.warmup_steps == 10_000
    assert train_cfg.lr_schedule.peak_lr == 5e-5
    assert train_cfg.lr_schedule.decay_steps == 1_000_000
    assert train_cfg.lr_schedule.decay_lr == 5e-5
    assert train_cfg.model.wm_slot_max_len == 768
    assert train_cfg.data.base_config.world_model.future_num_frames == 6


def test_stage3_libero_configs_reuse_canonical_ee_delta_pipeline(tmp_path):
    train_cfg = _config.get_config("stage3_finetuning_libero")
    data_cfg = train_cfg.data.create(tmp_path, train_cfg.model)

    assert isinstance(train_cfg.data, _config.LeRobotLiberoDataConfig)
    assert train_cfg.data.use_canonical_ee_delta is True
    assert isinstance(data_cfg.data_transforms.inputs[0], libero_policy.LiberoInputs)
    assert data_cfg.data_transforms.inputs[0].canonicalize_ee_pose_gripper is True
    assert data_cfg.data_transforms.inputs[0].treat_actions_as_commands is True
    assert data_cfg.data_transforms.inputs[0].dataset_state_gripper_format == "physical_width"
    assert data_cfg.data_transforms.inputs[0].dataset_action_gripper_format == "absolute_physical_width"

    assert isinstance(data_cfg.data_transforms.inputs[1], _transforms.DeltaActions)
    assert data_cfg.data_transforms.inputs[1].ee_pose is True
    assert tuple(data_cfg.data_transforms.inputs[1].mask) == (True,) * 8

    assert isinstance(data_cfg.data_transforms.outputs[0], _transforms.AbsoluteActions)
    assert data_cfg.data_transforms.outputs[0].ee_pose is True
    assert tuple(data_cfg.data_transforms.outputs[0].mask) == (True,) * 8

    assert isinstance(data_cfg.data_transforms.outputs[1], libero_policy.LiberoOutputs)
    assert data_cfg.normalization_contract.endswith("canonical_action=ee_delta")


def test_libero_gripper_contract_rejects_wrong_action_declaration(tmp_path):
    dataset = tmp_path / "libero"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta" / "norm_stats.json").write_text(
        json.dumps(
            {
                "observation.state": {"q01": [0.0] * 7 + [0.001], "q99": [0.0] * 7 + [0.04]},
                "action": {"q01": [0.0] * 7 + [-1.0], "q99": [0.0] * 7 + [1.0]},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="gripper contract mismatch"):
        _config._validate_local_libero_gripper_contract(
            str(dataset),
            state_format="physical_width",
            action_format="binary_target",
        )
