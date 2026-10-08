"""Dependency-free parsing of the policy server's execution contract.

The server sends most contract fields under ``metadata["input_spec"]`` and
keeps ``output_action_dim`` at the top level.  This module deliberately mirrors that
wire format without importing the training/serving ``plawvla`` package.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Tuple


INPUT_SPEC_METADATA_KEY = "input_spec"
ABSOLUTE_EEF_POSE_FORMAT = "absolute_eef_target"
WXYZ_QUATERNION_ORDER = "wxyz"
OPEN_FRACTION_GRIPPER_FORMAT = "open_fraction"
ABSOLUTE_EEF_ARM_DIM = 8


@dataclass(frozen=True)
class PolicyInputSpec:
    """Validated contents of the server's ``input_spec`` metadata field."""

    family: str
    image_keys: Tuple[str, ...]
    temporal_image_keys: Tuple[str, ...]
    state_key: Optional[str]
    prompt_key: Optional[str]
    state_gripper_format: str
    history_time_offsets_s: Tuple[float, ...]
    world_future_time_offsets_s: Tuple[float, ...]
    action_target_time_offsets_s: Tuple[float, ...]
    action_pose_format: str
    action_quaternion_order: str
    action_gripper_format: str
    action_arm_count: int
    normalization_contract: str

    def __post_init__(self) -> None:
        _validate_input_spec(self)

    def to_metadata_dict(self) -> Dict[str, Any]:
        """Return JSON/msgpack-friendly ``input_spec`` metadata."""

        return {
            "family": self.family,
            "image_keys": list(self.image_keys),
            "temporal_image_keys": list(self.temporal_image_keys),
            "state_key": self.state_key,
            "prompt_key": self.prompt_key,
            "state_gripper_format": self.state_gripper_format,
            "history_time_offsets_s": list(self.history_time_offsets_s),
            "world_future_time_offsets_s": list(self.world_future_time_offsets_s),
            "action_target_time_offsets_s": list(self.action_target_time_offsets_s),
            "action_pose_format": self.action_pose_format,
            "action_quaternion_order": self.action_quaternion_order,
            "action_gripper_format": self.action_gripper_format,
            "action_arm_count": self.action_arm_count,
            "normalization_contract": self.normalization_contract,
        }


@dataclass(frozen=True)
class PolicyContract(PolicyInputSpec):
    """Complete client-facing contract, including wire action dimension."""

    output_action_dim: int

    def __post_init__(self) -> None:
        super().__post_init__()
        _positive_integer(self.output_action_dim, "output_action_dim")
        if self.action_pose_format == ABSOLUTE_EEF_POSE_FORMAT:
            expected_dim = self.action_arm_count * ABSOLUTE_EEF_ARM_DIM
            if self.output_action_dim != expected_dim:
                raise ValueError(
                    "output_action_dim must equal action_arm_count * 8 for "
                    "%r; got output_action_dim=%d and action_arm_count=%d."
                    % (ABSOLUTE_EEF_POSE_FORMAT, self.output_action_dim, self.action_arm_count)
                )

    @property
    def input_spec(self) -> PolicyInputSpec:
        """Return the nested portion of this contract."""

        return PolicyInputSpec(
            family=self.family,
            image_keys=self.image_keys,
            temporal_image_keys=self.temporal_image_keys,
            state_key=self.state_key,
            prompt_key=self.prompt_key,
            state_gripper_format=self.state_gripper_format,
            history_time_offsets_s=self.history_time_offsets_s,
            world_future_time_offsets_s=self.world_future_time_offsets_s,
            action_target_time_offsets_s=self.action_target_time_offsets_s,
            action_pose_format=self.action_pose_format,
            action_quaternion_order=self.action_quaternion_order,
            action_gripper_format=self.action_gripper_format,
            action_arm_count=self.action_arm_count,
            normalization_contract=self.normalization_contract,
        )

    def to_metadata_dict(self) -> Dict[str, Any]:
        """Return metadata in the same shape accepted by the parser."""

        return {
            "output_action_dim": self.output_action_dim,
            INPUT_SPEC_METADATA_KEY: self.input_spec.to_metadata_dict(),
        }


def parse_policy_contract(metadata: Mapping) -> PolicyContract:
    """Parse and validate policy server metadata.

    No representation defaults are inferred.  A client must receive an
    explicit absolute-EEF pose layout, quaternion order, gripper convention,
    arm count, action dimension, and normalization contract before it can
    safely interpret action vectors.
    """

    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping.")

    raw_spec = _required(metadata, INPUT_SPEC_METADATA_KEY, "metadata")
    if not isinstance(raw_spec, Mapping):
        raise ValueError("metadata['input_spec'] must be a mapping.")

    return PolicyContract(
        family=_string(_required(raw_spec, "family", "input_spec"), "input_spec.family"),
        image_keys=_string_tuple(
            _required(raw_spec, "image_keys", "input_spec"), "input_spec.image_keys"
        ),
        temporal_image_keys=_string_tuple(
            _required(raw_spec, "temporal_image_keys", "input_spec"),
            "input_spec.temporal_image_keys",
        ),
        state_key=_optional_string(
            _required(raw_spec, "state_key", "input_spec"), "input_spec.state_key"
        ),
        prompt_key=_optional_string(
            _required(raw_spec, "prompt_key", "input_spec"), "input_spec.prompt_key"
        ),
        state_gripper_format=_string(
            _required(raw_spec, "state_gripper_format", "input_spec"),
            "input_spec.state_gripper_format",
        ),
        history_time_offsets_s=_float_tuple(
            _required(raw_spec, "history_time_offsets_s", "input_spec"),
            "input_spec.history_time_offsets_s",
        ),
        world_future_time_offsets_s=_float_tuple(
            _required(raw_spec, "world_future_time_offsets_s", "input_spec"),
            "input_spec.world_future_time_offsets_s",
        ),
        action_target_time_offsets_s=_float_tuple(
            _required(raw_spec, "action_target_time_offsets_s", "input_spec"),
            "input_spec.action_target_time_offsets_s",
        ),
        action_pose_format=_string(
            _required(raw_spec, "action_pose_format", "input_spec"),
            "input_spec.action_pose_format",
        ),
        action_quaternion_order=_string(
            _required(raw_spec, "action_quaternion_order", "input_spec"),
            "input_spec.action_quaternion_order",
        ),
        action_gripper_format=_string(
            _required(raw_spec, "action_gripper_format", "input_spec"),
            "input_spec.action_gripper_format",
        ),
        action_arm_count=_integer(
            _required(raw_spec, "action_arm_count", "input_spec"),
            "input_spec.action_arm_count",
        ),
        normalization_contract=_string(
            _required(raw_spec, "normalization_contract", "input_spec"),
            "input_spec.normalization_contract",
        ),
        output_action_dim=_integer(
            _required(metadata, "output_action_dim", "metadata"),
            "metadata.output_action_dim",
        ),
    )


def _validate_input_spec(spec: PolicyInputSpec) -> None:
    _string(spec.family, "family")
    image_keys = _validated_string_tuple(spec.image_keys, "image_keys")
    temporal_image_keys = _validated_string_tuple(
        spec.temporal_image_keys, "temporal_image_keys"
    )
    if not image_keys:
        raise ValueError("image_keys must not be empty.")
    if len(set(image_keys)) != len(image_keys):
        raise ValueError("image_keys must not contain duplicates.")
    if len(set(temporal_image_keys)) != len(temporal_image_keys):
        raise ValueError("temporal_image_keys must not contain duplicates.")
    unknown_temporal_keys = tuple(key for key in temporal_image_keys if key not in image_keys)
    if unknown_temporal_keys:
        raise ValueError("temporal_image_keys must be included in image_keys.")

    _optional_string(spec.state_key, "state_key")
    _optional_string(spec.prompt_key, "prompt_key")
    _string(spec.state_gripper_format, "state_gripper_format")

    history = _validated_float_tuple(spec.history_time_offsets_s, "history_time_offsets_s")
    if not history:
        raise ValueError("history_time_offsets_s must not be empty.")
    _strictly_increasing(history, "history_time_offsets_s")
    if any(offset > 0.0 for offset in history):
        raise ValueError("history_time_offsets_s values must be <= 0.")
    if temporal_image_keys and history[-1] != 0.0:
        raise ValueError(
            "history_time_offsets_s must end at 0 when temporal_image_keys is non-empty."
        )

    world_future = _validated_float_tuple(
        spec.world_future_time_offsets_s, "world_future_time_offsets_s"
    )
    _positive_strictly_increasing(world_future, "world_future_time_offsets_s")

    action_targets = _validated_float_tuple(
        spec.action_target_time_offsets_s, "action_target_time_offsets_s"
    )
    if not action_targets:
        raise ValueError("action_target_time_offsets_s must not be empty.")
    _positive_strictly_increasing(action_targets, "action_target_time_offsets_s")

    if _string(spec.action_pose_format, "action_pose_format") != ABSOLUTE_EEF_POSE_FORMAT:
        raise ValueError("action_pose_format must be %r." % ABSOLUTE_EEF_POSE_FORMAT)
    if (
        _string(spec.action_quaternion_order, "action_quaternion_order")
        != WXYZ_QUATERNION_ORDER
    ):
        raise ValueError("action_quaternion_order must be %r." % WXYZ_QUATERNION_ORDER)
    if _string(spec.action_gripper_format, "action_gripper_format") != OPEN_FRACTION_GRIPPER_FORMAT:
        raise ValueError("action_gripper_format must be %r." % OPEN_FRACTION_GRIPPER_FORMAT)
    _positive_integer(spec.action_arm_count, "action_arm_count")
    _string(spec.normalization_contract, "normalization_contract")


def _required(mapping: Mapping, key: str, container: str) -> Any:
    if key not in mapping:
        raise ValueError("%s is missing required field %r." % (container, key))
    return mapping[key]


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("%s must be a non-empty string." % name)
    return value


def _optional_string(value: Any, name: str) -> Optional[str]:
    if value is None:
        return None
    return _string(value, name)


def _integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("%s must be an integer." % name)
    return value


def _positive_integer(value: Any, name: str) -> int:
    value = _integer(value, name)
    if value <= 0:
        raise ValueError("%s must be positive." % name)
    return value


def _sequence(value: Any, name: str) -> Sequence:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("%s must be a sequence." % name)
    return value


def _string_tuple(value: Any, name: str) -> Tuple[str, ...]:
    return tuple(_string(item, "%s[%d]" % (name, index)) for index, item in enumerate(_sequence(value, name)))


def _validated_string_tuple(value: Any, name: str) -> Tuple[str, ...]:
    if not isinstance(value, tuple):
        raise ValueError("%s must be a tuple." % name)
    return tuple(_string(item, "%s[%d]" % (name, index)) for index, item in enumerate(value))


def _float_tuple(value: Any, name: str) -> Tuple[float, ...]:
    result = []
    for index, item in enumerate(_sequence(value, name)):
        item_name = "%s[%d]" % (name, index)
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise ValueError("%s must be a number." % item_name)
        converted = float(item)
        if not math.isfinite(converted):
            raise ValueError("%s must be finite." % item_name)
        result.append(converted)
    return tuple(result)


def _validated_float_tuple(value: Any, name: str) -> Tuple[float, ...]:
    if not isinstance(value, tuple):
        raise ValueError("%s must be a tuple." % name)
    return _float_tuple(value, name)


def _strictly_increasing(values: Tuple[float, ...], name: str) -> None:
    if any(current >= following for current, following in zip(values, values[1:])):
        raise ValueError("%s must be strictly increasing." % name)


def _positive_strictly_increasing(values: Tuple[float, ...], name: str) -> None:
    if any(value <= 0.0 for value in values):
        raise ValueError("%s values must be positive." % name)
    _strictly_increasing(values, name)


__all__ = [
    "ABSOLUTE_EEF_ARM_DIM",
    "ABSOLUTE_EEF_POSE_FORMAT",
    "INPUT_SPEC_METADATA_KEY",
    "OPEN_FRACTION_GRIPPER_FORMAT",
    "PolicyContract",
    "PolicyInputSpec",
    "WXYZ_QUATERNION_ORDER",
    "parse_policy_contract",
]
