# Derived from openpi (Copyright 2024 Physical Intelligence, Inc.; Apache-2.0).
# Modified for PLaW-VLA by the PLaW-VLA authors, 2026.
from collections.abc import Mapping
import dataclasses
import logging
import math
import pathlib
import time

import numpy as np
from plawvla_client import websocket_client_policy as _websocket_client_policy
from plawvla_client import policy_contract as _policy_contract
import polars as pl
import rich
import tqdm
import tyro

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    """Command line arguments."""

    # Host and port to connect to the server.
    host: str = "localhost"
    # Port of the policy server. serve_policy.py listens on 8001.
    port: int = 8001
    # API key to use for the server.
    api_key: str | None = None
    # Maximum wait for a connection, in seconds.
    connect_timeout: float = 30.0
    # Maximum wait for one inference response, in seconds.
    inference_timeout: float = 60.0
    # Number of steps to run the policy for.
    num_steps: int = 20
    # Path to save the timings to a parquet file. (e.g., timing.parquet)
    timing_file: pathlib.Path | None = None


class TimingRecorder:
    """Records timing measurements for different keys."""

    def __init__(self) -> None:
        self._timings: dict[str, list[float | None]] = {}
        self._num_steps = 0

    def record(self, measurements: Mapping[str, float]) -> None:
        """Record one inference step, preserving gaps in optional timing fields."""
        for key in self._timings.keys() | measurements.keys():
            self._timings.setdefault(key, [None] * self._num_steps).append(measurements.get(key))
        self._num_steps += 1

    def get_stats(self, key: str) -> dict[str, float]:
        """Get statistics for the given key."""
        times = [value for value in self._timings[key] if value is not None]
        return {
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
            "p25": float(np.quantile(times, 0.25)),
            "p50": float(np.quantile(times, 0.50)),
            "p75": float(np.quantile(times, 0.75)),
            "p90": float(np.quantile(times, 0.90)),
            "p95": float(np.quantile(times, 0.95)),
            "p99": float(np.quantile(times, 0.99)),
        }

    def print_all_stats(self) -> None:
        """Print statistics for all keys in a concise format."""

        table = rich.table.Table(
            title="[bold blue]Timing Statistics[/bold blue]",
            show_header=True,
            header_style="bold white",
            border_style="blue",
            title_justify="center",
        )

        # Add metric column with custom styling
        table.add_column("Metric", style="cyan", justify="left", no_wrap=True)

        # Add statistical columns with consistent styling
        stat_columns = [
            ("Mean", "yellow", "mean"),
            ("Std", "yellow", "std"),
            ("P25", "magenta", "p25"),
            ("P50", "magenta", "p50"),
            ("P75", "magenta", "p75"),
            ("P90", "magenta", "p90"),
            ("P95", "magenta", "p95"),
            ("P99", "magenta", "p99"),
        ]

        for name, style, _ in stat_columns:
            table.add_column(name, justify="right", style=style, no_wrap=True)

        # Add rows for each metric with formatted values
        for key in sorted(self._timings.keys()):
            stats = self.get_stats(key)
            values = [f"{stats[key]:.1f}" for _, _, key in stat_columns]
            table.add_row(key, *values)

        # Print with custom console settings
        console = rich.console.Console(width=None, highlight=True)
        console.print(table)

    def write_parquet(self, path: pathlib.Path) -> None:
        """Save the timings to a parquet file."""
        logger.info(f"Writing timings to {path}")
        frame = pl.DataFrame(self._timings)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)


def main(args: Args) -> None:
    if args.num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if not 0 < args.port <= 65535:
        raise ValueError("port must be between 1 and 65535.")
    for name in ("connect_timeout", "inference_timeout"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")

    with _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
        connect_timeout=args.connect_timeout,
        inference_timeout=args.inference_timeout,
    ) as policy:
        metadata = policy.get_server_metadata()
        logger.info(f"Server metadata: {metadata}")
        input_spec = _load_input_spec(metadata)

        # Send a few observations to make sure the model is loaded.
        for _ in range(2):
            policy.infer(_random_observation_libero(input_spec))

        timing_recorder = TimingRecorder()

        for _ in tqdm.trange(args.num_steps, desc="Running policy"):
            observation = _random_observation_libero(input_spec)
            inference_start = time.perf_counter()
            action = policy.infer(observation)
            measurements = {"client_infer_ms": 1000 * (time.perf_counter() - inference_start)}
            for key, value in action.get("server_timing", {}).items():
                measurements[f"server_{key}"] = value
            for key, value in action.get("policy_timing", {}).items():
                measurements[f"policy_{key}"] = value
            timing_recorder.record(measurements)

    timing_recorder.print_all_stats()

    if args.timing_file is not None:
        timing_recorder.write_parquet(args.timing_file)


def _load_input_spec(metadata: dict) -> Mapping:
    contract = _policy_contract.parse_policy_contract(metadata)
    if contract.family != "libero":
        raise ValueError(f"The synthetic client requires a LIBERO policy, got {contract.family!r}.")
    if contract.state_gripper_format != "two_finger_qpos":
        raise ValueError("The LIBERO policy must accept two_finger_qpos observations.")
    if contract.action_pose_format != "absolute_eef_target":
        raise ValueError("The LIBERO policy must return absolute EEF targets.")
    image_keys = set(contract.image_keys)
    if not image_keys or not image_keys.issubset({"observation/image", "observation/wrist_image"}):
        raise ValueError(f"Unsupported LIBERO image keys: {contract.image_keys!r}.")
    temporal_keys = set(contract.temporal_image_keys)
    if not temporal_keys.issubset(image_keys):
        raise ValueError("Temporal image keys must be included in image_keys.")
    return contract


def _random_observation_libero(input_spec: _policy_contract.PolicyContract) -> dict:
    """Create raw LIBERO inputs, including every history frame requested by the server."""
    temporal_keys = set(input_spec.temporal_image_keys)
    observation = {}
    for key in input_spec.image_keys:
        shape = (
            (len(input_spec.history_time_offsets_s), 256, 256, 3)
            if key in temporal_keys
            else (224, 224, 3)
        )
        observation[key] = np.random.randint(256, size=shape, dtype=np.uint8)
    if input_spec.state_key is not None:
        finger_opening = np.random.uniform(0.0, 0.04)
        observation[input_spec.state_key] = np.concatenate(
            (
                np.random.uniform([0.3, -0.2, 0.1], [0.6, 0.2, 0.4]),
                np.random.uniform(-0.2, 0.2, size=3),
                [finger_opening, -finger_opening],
            )
        ).astype(np.float32)
    if input_spec.prompt_key is not None:
        observation[input_spec.prompt_key] = "pick up the object"
    return observation


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
