from openpi.serving import policy_input_spec
from openpi.training import config as _config


def test_stage3_libero_input_spec_stays_raw_for_eval():
    spec = policy_input_spec.build_from_train_config(_config.get_config("stage3_finetuning_libero"))

    assert spec.family == "libero"
    assert spec.image_keys == ("observation/image", "observation/wrist_image")
    assert spec.temporal_image_keys == ("observation/image",)
    assert spec.state_key == "observation/state"
    assert spec.prompt_key == "prompt"
    assert spec.history_step_offsets == (-10, -8, -6, -4, -2, 0)
    assert spec.future_step_offsets == (2, 4, 6, 8, 10, 12)
    assert spec.state_gripper_format == "two_finger_qpos"
    assert spec.action_gripper_format == "signed_command"
    assert spec.normalization_contract == (
        "libero_eef_v2:state=physical_width:action=absolute_physical_width:"
        "canonical_state=open_fraction:canonical_action=ee_delta"
    )


def test_build_server_metadata_exposes_round_trippable_input_spec():
    metadata = policy_input_spec.build_server_metadata(_config.get_config("stage3_finetuning_libero"))
    spec = policy_input_spec.parse_policy_input_spec(metadata)

    assert metadata["policy_config_name"] == "stage3_finetuning_libero"
    assert metadata["model_type"] == "pi05"
    assert metadata["action_dim"] == 32
    assert metadata["action_horizon"] == 10
    assert spec == policy_input_spec.PolicyInputSpec(
        family="libero",
        image_keys=("observation/image", "observation/wrist_image"),
        temporal_image_keys=("observation/image",),
        state_key="observation/state",
        prompt_key="prompt",
        history_step_offsets=(-10, -8, -6, -4, -2, 0),
        future_step_offsets=(2, 4, 6, 8, 10, 12),
        state_gripper_format="two_finger_qpos",
        action_gripper_format="signed_command",
        normalization_contract=(
            "libero_eef_v2:state=physical_width:action=absolute_physical_width:"
            "canonical_state=open_fraction:canonical_action=ee_delta"
        ),
    )

