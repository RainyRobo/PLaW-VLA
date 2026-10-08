# Derived from openpi (Copyright 2024 Physical Intelligence, Inc.; Apache-2.0).
# Modified for PLaW-VLA by the PLaW-VLA authors, 2026.
"""Serve a trained policy over the shared websocket protocol.

Pass the training config and the checkpoint directory explicitly::

    uv run scripts/serve_policy.py policy:checkpoint \
        --policy.config=stage3_finetuning_libero \
        --policy.dir=checkpoints/stage3_finetuning_libero/<run_name>/<step>
"""

import dataclasses
import enum
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import policy_input_spec as _policy_input_spec
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Policy environments accepted by the serving entrypoints."""

    LIBERO = "libero"
    ROBOTWIN = "robotwin"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "stage3_finetuning_libero").
    config: str
    # Checkpoint directory (e.g., "checkpoints/stage3_finetuning_libero/my_exp/30000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.LIBERO

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8001
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)

    pytorch_device: str | None = None


# Register named default checkpoints here when needed. Until then, pass
# `policy:checkpoint --policy.config=... --policy.dir=...` explicitly.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {}


def create_default_policy(
    env: EnvMode,
    *,
    default_prompt: str | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        train_config = _config.get_config(checkpoint.config)
        train_config = dataclasses.replace(
            train_config,
            policy_metadata=_policy_input_spec.build_server_metadata(train_config),
        )
        return _policy_config.create_trained_policy(
            train_config, checkpoint.dir, default_prompt=default_prompt, pytorch_device=pytorch_device
        )
    raise ValueError(
        f"Default policy is not configured for environment {env.value!r}. "
        "Specify a checkpoint explicitly, e.g.\n"
        "  uv run scripts/serve_policy.py policy:checkpoint "
        "--policy.config=stage3_finetuning_libero --policy.dir=<checkpoint_dir>"
    )


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            train_config = _config.get_config(args.policy.config)
            train_config = dataclasses.replace(
                train_config,
                policy_metadata=_policy_input_spec.build_server_metadata(train_config),
            )
            return _policy_config.create_trained_policy(
                train_config,
                args.policy.dir,
                default_prompt=args.default_prompt,
                pytorch_device=args.pytorch_device,
            )
        case Default():
            return create_default_policy(
                args.env,
                default_prompt=args.default_prompt,
                pytorch_device=args.pytorch_device,
            )


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except OSError:
        local_ip = "127.0.0.1"
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
