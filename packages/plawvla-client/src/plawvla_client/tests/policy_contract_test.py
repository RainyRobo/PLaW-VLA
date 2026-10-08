import copy
import dataclasses

import pytest

from plawvla_client import policy_contract


def _metadata():
    return {
        "policy_config_name": "stage3_finetuning_libero",
        "output_action_dim": 8,
        "input_spec": {
            "family": "libero",
            "image_keys": ["observation/image", "observation/wrist_image"],
            "temporal_image_keys": ["observation/image"],
            "state_key": "observation/state",
            "prompt_key": "prompt",
            "state_gripper_format": "two_finger_qpos",
            "history_time_offsets_s": [-1.0, -0.5, 0.0],
            "world_future_time_offsets_s": [0.5, 1.0],
            "action_target_time_offsets_s": [0.1, 0.2, 0.3],
            "action_pose_format": "absolute_eef_target",
            "action_quaternion_order": "wxyz",
            "action_gripper_format": "open_fraction",
            "action_arm_count": 1,
            "normalization_contract": "libero_eef_v2",
        },
    }


def test_parse_policy_contract_and_round_trip():
    contract = policy_contract.parse_policy_contract(_metadata())

    assert contract == policy_contract.PolicyContract(
        family="libero",
        image_keys=("observation/image", "observation/wrist_image"),
        temporal_image_keys=("observation/image",),
        state_key="observation/state",
        prompt_key="prompt",
        state_gripper_format="two_finger_qpos",
        history_time_offsets_s=(-1.0, -0.5, 0.0),
        world_future_time_offsets_s=(0.5, 1.0),
        action_target_time_offsets_s=(0.1, 0.2, 0.3),
        action_pose_format="absolute_eef_target",
        action_quaternion_order="wxyz",
        action_gripper_format="open_fraction",
        action_arm_count=1,
        normalization_contract="libero_eef_v2",
        output_action_dim=8,
    )
    assert contract.input_spec == policy_contract.PolicyInputSpec(
        family="libero",
        image_keys=("observation/image", "observation/wrist_image"),
        temporal_image_keys=("observation/image",),
        state_key="observation/state",
        prompt_key="prompt",
        state_gripper_format="two_finger_qpos",
        history_time_offsets_s=(-1.0, -0.5, 0.0),
        world_future_time_offsets_s=(0.5, 1.0),
        action_target_time_offsets_s=(0.1, 0.2, 0.3),
        action_pose_format="absolute_eef_target",
        action_quaternion_order="wxyz",
        action_gripper_format="open_fraction",
        action_arm_count=1,
        normalization_contract="libero_eef_v2",
    )
    assert policy_contract.parse_policy_contract(contract.to_metadata_dict()) == contract


def test_parse_accepts_nullable_observation_keys_and_integer_offsets():
    metadata = _metadata()
    metadata["output_action_dim"] = 16
    metadata["input_spec"].update(
        {
            "state_key": None,
            "prompt_key": None,
            "history_time_offsets_s": [-1, 0],
            "world_future_time_offsets_s": [],
            "action_target_time_offsets_s": [1, 2],
            "action_arm_count": 2,
        }
    )

    contract = policy_contract.parse_policy_contract(metadata)

    assert contract.state_key is None
    assert contract.prompt_key is None
    assert contract.history_time_offsets_s == (-1.0, 0.0)
    assert contract.action_target_time_offsets_s == (1.0, 2.0)


@pytest.mark.parametrize(
    ("container", "field"),
    [
        ("metadata", "output_action_dim"),
        ("input_spec", "family"),
        ("input_spec", "image_keys"),
        ("input_spec", "temporal_image_keys"),
        ("input_spec", "state_key"),
        ("input_spec", "prompt_key"),
        ("input_spec", "state_gripper_format"),
        ("input_spec", "history_time_offsets_s"),
        ("input_spec", "world_future_time_offsets_s"),
        ("input_spec", "action_target_time_offsets_s"),
        ("input_spec", "action_pose_format"),
        ("input_spec", "action_quaternion_order"),
        ("input_spec", "action_gripper_format"),
        ("input_spec", "action_arm_count"),
        ("input_spec", "normalization_contract"),
    ],
)
def test_missing_required_fields_are_rejected(container, field):
    metadata = _metadata()
    if container == "metadata":
        del metadata[field]
    else:
        del metadata["input_spec"][field]

    with pytest.raises(ValueError, match="missing required field"):
        policy_contract.parse_policy_contract(metadata)


@pytest.mark.parametrize("metadata", [None, [], "metadata", 1])
def test_metadata_must_be_a_mapping(metadata):
    with pytest.raises(ValueError, match="metadata must be a mapping"):
        policy_contract.parse_policy_contract(metadata)


@pytest.mark.parametrize("input_spec", [None, [], "input_spec", 1])
def test_input_spec_must_be_a_mapping(input_spec):
    metadata = _metadata()
    metadata["input_spec"] = input_spec

    with pytest.raises(ValueError, match=r"metadata\['input_spec'\] must be a mapping"):
        policy_contract.parse_policy_contract(metadata)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("history_time_offsets_s", [-1.0, -1.0, 0.0], "strictly increasing"),
        ("history_time_offsets_s", [-1.0, 0.1], "must be <= 0"),
        ("history_time_offsets_s", [-1.0, -0.1], "must end at 0"),
        ("history_time_offsets_s", [], "must not be empty"),
        ("world_future_time_offsets_s", [0.0, 1.0], "must be positive"),
        ("world_future_time_offsets_s", [1.0, 0.5], "strictly increasing"),
        ("action_target_time_offsets_s", [0.0, 0.1], "must be positive"),
        ("action_target_time_offsets_s", [0.2, 0.1], "strictly increasing"),
        ("action_target_time_offsets_s", [], "must not be empty"),
        ("action_pose_format", "delta_eef", "absolute_eef_target"),
        ("action_quaternion_order", "xyzw", "wxyz"),
        ("action_gripper_format", "signed_command", "open_fraction"),
        ("action_arm_count", 0, "must be positive"),
        ("action_arm_count", True, "must be an integer"),
    ],
)
def test_invalid_input_spec_values_are_rejected(field, value, message):
    metadata = _metadata()
    metadata["input_spec"][field] = value

    with pytest.raises(ValueError, match=message):
        policy_contract.parse_policy_contract(metadata)


@pytest.mark.parametrize("output_action_dim", [0, -1, True, 8.0, "8"])
def test_action_dim_must_be_a_positive_integer(output_action_dim):
    metadata = _metadata()
    metadata["output_action_dim"] = output_action_dim

    with pytest.raises(ValueError, match="output_action_dim"):
        policy_contract.parse_policy_contract(metadata)


def test_absolute_eef_action_dim_must_match_arm_count():
    metadata = _metadata()
    metadata["output_action_dim"] = 16

    with pytest.raises(ValueError, match=r"action_arm_count \* 8"):
        policy_contract.parse_policy_contract(metadata)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("family", ""),
        ("image_keys", []),
        ("image_keys", ["observation/image", "observation/image"]),
        ("temporal_image_keys", ["unknown/image"]),
        ("state_key", 7),
        ("prompt_key", ""),
        ("state_gripper_format", ""),
        ("normalization_contract", None),
        ("history_time_offsets_s", [-1.0, float("nan"), 0.0]),
        ("action_target_time_offsets_s", [float("inf")]),
    ],
)
def test_malformed_values_are_rejected(field, value):
    metadata = _metadata()
    metadata["input_spec"][field] = value

    with pytest.raises(ValueError):
        policy_contract.parse_policy_contract(metadata)


def test_non_temporal_contract_does_not_require_history_to_end_at_zero():
    metadata = _metadata()
    metadata["input_spec"]["temporal_image_keys"] = []
    metadata["input_spec"]["history_time_offsets_s"] = [-1.0]

    contract = policy_contract.parse_policy_contract(metadata)

    assert contract.history_time_offsets_s == (-1.0,)


def test_dataclasses_are_frozen_and_validate_direct_construction():
    contract = policy_contract.parse_policy_contract(_metadata())

    with pytest.raises(dataclasses.FrozenInstanceError):
        contract.action_dim = 16

    values = copy.deepcopy(contract.input_spec.to_metadata_dict())
    values["action_quaternion_order"] = "xyzw"
    with pytest.raises(ValueError, match="wxyz"):
        policy_contract.PolicyInputSpec(
            family=values["family"],
            image_keys=tuple(values["image_keys"]),
            temporal_image_keys=tuple(values["temporal_image_keys"]),
            state_key=values["state_key"],
            prompt_key=values["prompt_key"],
            state_gripper_format=values["state_gripper_format"],
            history_time_offsets_s=tuple(values["history_time_offsets_s"]),
            world_future_time_offsets_s=tuple(values["world_future_time_offsets_s"]),
            action_target_time_offsets_s=tuple(values["action_target_time_offsets_s"]),
            action_pose_format=values["action_pose_format"],
            action_quaternion_order=values["action_quaternion_order"],
            action_gripper_format=values["action_gripper_format"],
            action_arm_count=values["action_arm_count"],
            normalization_contract=values["normalization_contract"],
        )
