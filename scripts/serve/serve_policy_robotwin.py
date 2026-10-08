# Copyright 2026 PLaW-VLA authors. SPDX-License-Identifier: Apache-2.0
"""Serve a RoboTwin EEF checkpoint with per-task normalization routing."""

import dataclasses
import logging
import socket
import sys
from pathlib import Path

import tyro

# Share the serving arguments with ``scripts/serve/serve_policy.py``.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts import serve_policy as _serve  # noqa: E402

from plawvla.policies import robotwin_policy as _robotwin_policy  # noqa: E402
from plawvla.policies import policy_config as _policy_config  # noqa: E402
from plawvla.serving import policy_input_spec as _policy_input_spec  # noqa: E402
from plawvla.serving import websocket_policy_server  # noqa: E402
from plawvla.training import config as _config  # noqa: E402


def main(args: _serve.Args) -> None:
    if args.env != _serve.EnvMode.ROBOTWIN:
        raise SystemExit(
            "scripts/serve/serve_policy_robotwin.py is RobotWin-specific; "
            "use scripts/serve/serve_policy.py for any other environment."
        )
    if not isinstance(args.policy, _serve.Checkpoint):
        raise SystemExit(
            "scripts/serve/serve_policy_robotwin.py only supports `policy:checkpoint`. Pass --policy.config / --policy.dir."
        )

    ckpt_dir = Path(args.policy.dir).expanduser().resolve()
    assets_path = ckpt_dir / "assets"
    # ``_scan_norm_stats`` walks the assets tree recursively and collects every
    # leaf directory that contains a ``norm_stats.json``. Returns
    # {asset_id (relative path under assets/) -> {state, actions -> NormStats}}.
    norm_stats_map = _robotwin_policy._scan_norm_stats(str(assets_path))  # noqa: SLF001
    if not norm_stats_map:
        raise SystemExit(
            f"No norm_stats.json found under {assets_path}; cannot serve RobotWin "
            "checkpoint without per-task normalisation statistics."
        )
    default_asset_id = sorted(norm_stats_map.keys())[0]
    logging.info(
        "Discovered %d per-task norm_stats under %s (default=%s)",
        len(norm_stats_map),
        assets_path,
        default_asset_id,
    )

    train_config = _config.get_config(args.policy.config)
    train_config = dataclasses.replace(train_config, data=dataclasses.replace(train_config.data, load_norm_stats=False))
    train_config = dataclasses.replace(
        train_config,
        policy_metadata=_policy_input_spec.build_server_metadata(train_config),
    )

    base_policy = _policy_config.create_trained_policy(
        train_config,
        str(ckpt_dir),
        default_prompt=args.default_prompt,
        pytorch_device=args.pytorch_device,
        norm_stats=norm_stats_map[default_asset_id],
    )

    if len(norm_stats_map) > 1:
        policy = _robotwin_policy.RobotwinPolicy(
            base_policy,
            norm_stats_map,
            default_asset_id=default_asset_id,
        )
        logging.info("Wrapped policy with RobotwinPolicy router (%d tasks).", len(norm_stats_map))
    else:
        policy = base_policy

    policy_metadata = base_policy.metadata

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except OSError:
        local_ip = "127.0.0.1"
    logging.info("Creating server (host: %s, ip: %s, port: %d)", hostname, local_ip, args.port)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(_serve.Args))
