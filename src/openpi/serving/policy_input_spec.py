from __future__ import annotations

from collections.abc import Mapping
import dataclasses
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openpi.training import config as _config


INPUT_SPEC_METADATA_KEY = "input_spec"


@dataclasses.dataclass(frozen=True)
class PolicyInputSpec:
    """Inference-facing raw input schema for a served policy."""

    family: str
    image_keys: tuple[str, ...]
    temporal_image_keys: tuple[str, ...] = ()
    state_key: str | None = None
    prompt_key: str | None = "prompt"
    history_step_offsets: tuple[int, ...] = (0,)
    future_step_offsets: tuple[int, ...] = ()
    # Optional raw environment contract.  Clients should reject incompatible
    # values instead of guessing gripper units or signs.
    state_gripper_format: str | None = None
    action_gripper_format: str | None = None
    normalization_contract: str | None = None

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "image_keys": list(self.image_keys),
            "temporal_image_keys": list(self.temporal_image_keys),
            "state_key": self.state_key,
            "prompt_key": self.prompt_key,
            "history_step_offsets": list(self.history_step_offsets),
            "future_step_offsets": list(self.future_step_offsets),
            "state_gripper_format": self.state_gripper_format,
            "action_gripper_format": self.action_gripper_format,
            "normalization_contract": self.normalization_contract,
        }


def parse_policy_input_spec(metadata: Mapping[str, Any] | None) -> PolicyInputSpec | None:
    if not metadata:
        return None

    raw_spec = metadata.get(INPUT_SPEC_METADATA_KEY)
    if not isinstance(raw_spec, Mapping):
        return None

    return PolicyInputSpec(
        family=str(raw_spec["family"]),
        image_keys=tuple(str(key) for key in raw_spec.get("image_keys", ())),
        temporal_image_keys=tuple(str(key) for key in raw_spec.get("temporal_image_keys", ())),
        state_key=None if raw_spec.get("state_key") is None else str(raw_spec["state_key"]),
        prompt_key=None if raw_spec.get("prompt_key") is None else str(raw_spec["prompt_key"]),
        history_step_offsets=tuple(int(offset) for offset in raw_spec.get("history_step_offsets", (0,))),
        future_step_offsets=tuple(int(offset) for offset in raw_spec.get("future_step_offsets", ())),
        state_gripper_format=(
            None if raw_spec.get("state_gripper_format") is None else str(raw_spec["state_gripper_format"])
        ),
        action_gripper_format=(
            None if raw_spec.get("action_gripper_format") is None else str(raw_spec["action_gripper_format"])
        ),
        normalization_contract=(
            None if raw_spec.get("normalization_contract") is None else str(raw_spec["normalization_contract"])
        ),
    )


def build_server_metadata(train_config: "_config.TrainConfig") -> dict[str, Any]:
    input_spec = build_from_train_config(train_config)
    return {
        **(train_config.policy_metadata or {}),
        "policy_config_name": train_config.name,
        "model_type": train_config.model.model_type.value,
        "action_dim": int(train_config.model.action_dim),
        "action_horizon": int(train_config.model.action_horizon),
        INPUT_SPEC_METADATA_KEY: input_spec.to_metadata_dict(),
    }


def build_from_train_config(train_config: "_config.TrainConfig") -> PolicyInputSpec:
    from openpi.policies import libero_policy as _libero_policy

    data_factory = dataclasses.replace(train_config.data, load_norm_stats=False)
    data_config = data_factory.create(train_config.assets_dirs, train_config.model)
    input_transforms = tuple(data_config.data_transforms.inputs)
    if not input_transforms:
        raise ValueError(f"Config {train_config.name!r} does not expose any policy input transforms.")

    input_transform = _resolve_policy_input_transform(input_transforms)
    if not isinstance(input_transform, _libero_policy.LiberoInputs):
        raise ValueError(
            f"Config {train_config.name!r} uses unsupported policy input transform "
            f"{type(input_transform).__name__}."
        )

    temporal_image_keys = tuple(input_transform.image_keys) if input_transform.enable_world_model else ()
    primary_image_key = temporal_image_keys[0] if temporal_image_keys else "observation/image"
    return PolicyInputSpec(
        family="libero",
        image_keys=_dedupe_keys(primary_image_key, "observation/wrist_image"),
        temporal_image_keys=temporal_image_keys,
        state_key=None if input_transform.pretrain_world_model else "observation/state",
        prompt_key="prompt",
        history_step_offsets=_history_offsets(data_config, enabled=bool(temporal_image_keys)),
        future_step_offsets=_future_offsets(data_config, enabled=bool(temporal_image_keys)),
        state_gripper_format="two_finger_qpos",
        action_gripper_format="signed_command",
        normalization_contract=data_config.normalization_contract,
    )


def _resolve_policy_input_transform(input_transforms: tuple[Any, ...]) -> Any:
    from openpi.policies import libero_policy as _libero_policy

    for transform in input_transforms:
        if isinstance(transform, _libero_policy.LiberoInputs):
            return transform

    raise ValueError("Config does not expose a LiberoInputs policy transform.")


def _dedupe_keys(*keys: str) -> tuple[str, ...]:
    ordered: list[str] = []
    for key in keys:
        if key not in ordered:
            ordered.append(key)
    return tuple(ordered)


def _history_offsets(data_config: Any, *, enabled: bool) -> tuple[int, ...]:
    if not enabled:
        return (0,)
    return tuple(offset for offset in _resolve_step_offsets(data_config) if offset <= 0)


def _future_offsets(data_config: Any, *, enabled: bool) -> tuple[int, ...]:
    if not enabled:
        return ()
    return tuple(offset for offset in _resolve_step_offsets(data_config) if offset > 0)


def _resolve_step_offsets(data_config: Any) -> tuple[int, ...]:
    world_model = data_config.world_model
    if world_model.time_offsets_s is None:
        return tuple(int(offset) for offset in world_model.resolve_frame_indices())

    if data_config.action_time_step_s is None:
        return tuple(int(offset) for offset in world_model.resolve_layout_indices())

    step_s = float(data_config.action_time_step_s)
    step_offsets: list[int] = []
    for offset_s in world_model.resolve_time_offsets(1.0 / step_s):
        raw_step_offset = float(offset_s) / step_s
        rounded_step_offset = int(round(raw_step_offset))
        if abs(raw_step_offset - rounded_step_offset) > 1e-6:
            raise ValueError(
                "World-model time offsets must align with the action timestep during live inference: "
                f"offset_s={offset_s}, action_time_step_s={step_s}."
            )
        step_offsets.append(rounded_step_offset)
    return tuple(step_offsets)
