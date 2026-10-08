# Derived from openpi (Copyright 2024 Physical Intelligence, Inc.; Apache-2.0).
# Modified for PLaW-VLA by the PLaW-VLA authors, 2026.
# Simulator paths and rendering settings must precede third-party imports.
# ruff: noqa: E402
from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import datetime
import json
import logging
import math
import os
import pathlib
import re
import sys
import tempfile
from typing import Any

import imageio
import numpy as np
import tyro

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = pathlib.Path(__file__).resolve().parent
LIBERO_PLUS_SRC_ROOT = PROJECT_ROOT / "third_party" / "libero-plus"
LIBERO_PLUS_ROOT = LIBERO_PLUS_SRC_ROOT / "libero" / "libero"
LIBERO_PLUS_DATASETS_ROOT = LIBERO_PLUS_SRC_ROOT / "libero" / "datasets"
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(LIBERO_PLUS_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(LIBERO_PLUS_SRC_ROOT))


def _ensure_libero_plus_config() -> None:
    config_root = pathlib.Path(
        os.environ.get("LIBERO_PLUS_CONFIG_PATH", "~/.cache/robot_policy/libero-plus")
    ).expanduser()
    if not config_root.is_absolute():
        config_root = PROJECT_ROOT / config_root

    os.environ["LIBERO_CONFIG_PATH"] = str(config_root)
    config_root.mkdir(parents=True, exist_ok=True)
    config_content = (
        "\n".join(
            (
                f"benchmark_root: {LIBERO_PLUS_ROOT}",
                f"bddl_files: {LIBERO_PLUS_ROOT / 'bddl_files'}",
                f"init_states: {LIBERO_PLUS_ROOT / 'init_files'}",
                f"datasets: {LIBERO_PLUS_DATASETS_ROOT}",
                f"assets: {LIBERO_PLUS_ROOT / 'assets'}",
            )
        )
        + "\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", dir=config_root, encoding="utf-8", delete=False) as handle:
        handle.write(config_content)
        temporary = pathlib.Path(handle.name)
    temporary.replace(config_root / "config.yaml")


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
_ensure_libero_plus_config()

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from plawvla_client import execution as _execution
from plawvla_client import image_tools
from plawvla_client import policy_contract as _policy_contract
from plawvla_client import websocket_client_policy as _websocket_client_policy

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
LIBERO_CONTROL_HZ = 20.0
TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}

PERTURBATION_CATEGORIES = [
    "Camera Viewpoints",
    "Robot Initial States",
    "Language Instructions",
    "Light Conditions",
    "Background Textures",
    "Sensor Noise",
    "Objects Layout",
]

COL_NAMES = {
    "Camera Viewpoints": "Camera",
    "Robot Initial States": "Robot",
    "Language Instructions": "Language",
    "Light Conditions": "Light",
    "Background Textures": "Background",
    "Sensor Noise": "Noise",
    "Objects Layout": "Layout",
}

LIBERO_PLUS_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

_SUPPORTED_FRONT_KEYS = {
    "observation/image",
    "observation/front_image",
}
_SUPPORTED_WRIST_KEYS = {
    "observation/wrist_image",
}


@dataclasses.dataclass
class Args:
    host: str = "localhost"
    port: int = 8001
    connect_timeout: float = 30.0
    inference_timeout: float = 60.0
    resize_size: int = 224
    # Match examples/libero/main.py. Executing fewer steps would replan more often.
    replan_steps: int = 5
    task_suite_name: str | None = None
    task_suite_names: list[str] = dataclasses.field(default_factory=lambda: list(LIBERO_PLUS_SUITES))
    task_ids: tuple[int, ...] = ()
    num_steps_wait: int = 10
    # LIBERO-Plus uses one trial per task. Upstream LIBERO uses 50.
    num_trials_per_task: int = 1
    result_root: str = "results/libero_plus"
    checkpoint_path: str | None = None
    record_video: str = "failure"
    video_out_path: str = "results/libero_plus/videos"
    seed: int = 42
    task_classification_path: str | None = None
    # Retained for CLI compatibility, but unsupported for absolute EEF trajectories.
    action_ensembling: bool = False
    ensemble_window: int = 8
    ensemble_decay: float = 0.1


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    w = float(np.clip(quat[3], -1.0, 1.0))
    den = np.sqrt(1.0 - w * w)
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(w)) / den


def _preprocess_image(img: np.ndarray, size: int) -> np.ndarray:
    img = np.ascontiguousarray(img[::-1, ::-1])
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(img, size, size))


def _load_policy_input_spec(metadata: dict[str, Any]) -> _policy_contract.PolicyContract:
    input_spec = _policy_contract.parse_policy_contract(metadata)
    if input_spec.family not in {"libero", "libero_plus"}:
        raise ValueError(f"Unsupported LIBERO-Plus input family: {input_spec.family!r}")
    if len(input_spec.temporal_image_keys) > 1:
        raise ValueError("LIBERO-Plus evaluation supports at most one temporal image input.")
    if not input_spec.image_keys:
        raise ValueError("Policy input_spec must contain image keys.")
    for key in input_spec.image_keys:
        if key not in _SUPPORTED_FRONT_KEYS and key not in _SUPPORTED_WRIST_KEYS:
            raise ValueError(f"Unsupported LIBERO-Plus image key {key!r}.")
    for key in input_spec.temporal_image_keys:
        if key not in _SUPPORTED_FRONT_KEYS:
            raise ValueError(f"Unsupported LIBERO-Plus temporal image key {key!r}.")
    if not set(input_spec.temporal_image_keys).issubset(input_spec.image_keys):
        raise ValueError("Temporal image keys must be included in image_keys.")
    if input_spec.state_gripper_format != "two_finger_qpos":
        raise ValueError(
            f"LIBERO-Plus client provides two_finger_qpos; server expects {input_spec.state_gripper_format!r}."
        )
    if input_spec.action_gripper_format != "open_fraction":
        raise ValueError(
            "LIBERO-Plus execution requires open_fraction gripper targets; "
            f"server returns {input_spec.action_gripper_format!r}."
        )
    if input_spec.action_pose_format != "absolute_eef_target":
        raise ValueError(
            "LIBERO-Plus requires absolute_eef_target actions; "
            f"server returns {input_spec.action_pose_format!r}."
        )
    if input_spec.action_quaternion_order != "wxyz" or input_spec.action_arm_count != 1:
        raise ValueError("LIBERO-Plus requires one arm with wxyz absolute EEF targets.")
    if input_spec.output_action_dim != 8:
        raise ValueError(
            f"LIBERO-Plus requires output_action_dim=8, got {input_spec.output_action_dim}."
        )
    return input_spec


def _libero_image_for_key(key: str, front_image: np.ndarray, wrist_image: np.ndarray) -> np.ndarray:
    if key in _SUPPORTED_WRIST_KEYS:
        return wrist_image
    if key in _SUPPORTED_FRONT_KEYS:
        return front_image
    raise ValueError(f"Unsupported LIBERO image key {key!r}.")


def _libero_raw_state(obs: dict[str, Any]) -> np.ndarray:
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)
    if state.shape != (8,) or not np.all(np.isfinite(state)):
        raise ValueError(f"Expected a finite 8D raw LIBERO state, got {state!r}.")
    return state


def _control_dt_s(env) -> float:
    control_dt_s = float(env.env.control_timestep)
    expected_dt_s = 1.0 / LIBERO_CONTROL_HZ
    if not math.isfinite(control_dt_s) or not math.isclose(
        control_dt_s, expected_dt_s, rel_tol=1e-6, abs_tol=1e-9
    ):
        raise ValueError(
            f"LIBERO-Plus requires a {LIBERO_CONTROL_HZ:g} Hz control timestep "
            f"({expected_dt_s:g} s), got {control_dt_s!r} s."
        )
    return control_dt_s


def _replan_duration_s(input_spec: _policy_contract.PolicyContract, replan_steps: int) -> float:
    if replan_steps > len(input_spec.action_target_time_offsets_s):
        raise ValueError(
            f"replan_steps={replan_steps} exceeds action horizon "
            f"{len(input_spec.action_target_time_offsets_s)}."
        )
    return float(input_spec.action_target_time_offsets_s[replan_steps - 1])


def _get_libero_env(task, resolution: int, seed: int):
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task.language


def _load_task_classification(suite_name: str, path: str | None = None) -> dict[str, dict[str, Any]]:
    if path is None:
        task_classification_path = LIBERO_PLUS_ROOT / "benchmark" / "task_classification.json"
    else:
        task_classification_path = pathlib.Path(path)
    if not task_classification_path.exists():
        return {}
    with open(task_classification_path, encoding="utf-8") as handle:
        data = json.load(handle)
    return {entry["name"]: entry for entry in data.get(suite_name, [])}


def _select_task_ids(task_suite, requested_task_ids: Sequence[int]) -> list[int]:
    if not requested_task_ids:
        return list(range(task_suite.n_tasks))

    selected: list[int] = []
    seen: set[int] = set()
    for task_id in requested_task_ids:
        if task_id < 0 or task_id >= task_suite.n_tasks:
            raise ValueError(
                f"Task id {task_id} is out of range for this suite; expected 0 <= task_id < {task_suite.n_tasks}."
            )
        if task_id in seen:
            continue
        seen.add(task_id)
        selected.append(task_id)
    return selected


def _rate(results: list[bool]) -> float | None:
    return sum(results) / len(results) * 100 if results else None


def _load_checkpoint(path: pathlib.Path, seed: int) -> list[dict[str, Any]]:
    if path.exists():
        with open(path, encoding="utf-8") as handle:
            checkpoint = json.load(handle)
        if checkpoint.get("seed") != seed:
            raise ValueError(f"Checkpoint {path} was not saved for seed {seed}; use a separate result directory.")
        records = checkpoint["records"]
        logging.info("Resumed from checkpoint: %s (%d records)", path, len(records))
        return records
    return []


def _save_checkpoint(path: pathlib.Path, records: list[dict[str, Any]], seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(json.dumps({"seed": seed, "records": records}), encoding="utf-8")
    temporary_path.replace(path)


def _rebuild_state(records: list[dict[str, Any]]):
    suite_results: dict[str, dict[str, list[bool]]] = {}
    suite_totals: dict[str, dict[str, int]] = {}
    done_keys: set[str] = set()
    for record in records:
        suite = record["suite"]
        suite_results.setdefault(suite, {category: [] for category in PERTURBATION_CATEGORIES})
        suite_totals.setdefault(suite, {"successes": 0, "episodes": 0})
        suite_totals[suite]["episodes"] += 1
        if record["success"]:
            suite_totals[suite]["successes"] += 1
        category = record.get("category")
        if category and category in suite_results[suite]:
            suite_results[suite][category].append(record["success"])
        done_keys.add(f"{suite}|{record['task_id']}|{record['episode_idx']}")
    return suite_results, suite_totals, done_keys


def run_single_episode(
    env,
    task_description: str,
    client: _websocket_client_policy.WebsocketClientPolicy,
    initial_state: np.ndarray,
    args: Args,
    input_spec: _policy_contract.PolicyContract,
    max_steps: int,
):
    env.reset()
    obs = env.set_init_state(initial_state)
    temporal_keys = set(input_spec.temporal_image_keys)
    trajectory = None
    trajectory_elapsed_s = 0.0
    image_history = _execution.TimedHistoryBuffer()
    simulation_time_s = 0.0
    control_dt_s = _control_dt_s(env)
    replan_duration_s = _replan_duration_s(input_spec, args.replan_steps)
    osc_adapter = _execution.RobosuiteOSCAdapter()
    if temporal_keys:
        image_history.append(_preprocess_image(obs["agentview_image"], 256), simulation_time_s)
    replay_images = []

    # Let contacts settle while collecting timestamped history at the simulator timestep.
    for _ in range(args.num_steps_wait):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
        simulation_time_s += control_dt_s
        if temporal_keys:
            image_history.append(_preprocess_image(obs["agentview_image"], 256), simulation_time_s)

    for _ in range(max_steps):
        front_image = _preprocess_image(obs["agentview_image"], args.resize_size)
        wrist_image = _preprocess_image(obs["robot0_eye_in_hand_image"], args.resize_size)
        if args.record_video != "none":
            replay_images.append(front_image)

        if trajectory is None or trajectory_elapsed_s >= replan_duration_s - 1e-9:
            robot_state = _libero_raw_state(obs)
            request = {}
            for image_key in input_spec.image_keys:
                if image_key in temporal_keys:
                    request[image_key], request[f"{image_key}_is_pad"] = image_history.sample_offsets(
                        simulation_time_s,
                        np.asarray(input_spec.history_time_offsets_s, dtype=np.float64),
                    )
                else:
                    request[image_key] = _libero_image_for_key(image_key, front_image, wrist_image)
            if input_spec.state_key is not None:
                request[input_spec.state_key] = robot_state
            if input_spec.prompt_key is not None:
                request[input_spec.prompt_key] = task_description
            action_chunk = np.asarray(client.infer(request)["actions"], dtype=np.float32)
            if (
                action_chunk.ndim != 2
                or action_chunk.shape[0] == 0
                or action_chunk.shape[1] != input_spec.output_action_dim
            ):
                raise ValueError(
                    "Policy server returned invalid action chunk shape "
                    f"{action_chunk.shape}; expected [positive horizon, {input_spec.output_action_dim}]."
                )
            if action_chunk.shape[0] != len(input_spec.action_target_time_offsets_s):
                raise ValueError(
                    "Policy server returned an action horizon inconsistent with metadata: "
                    f"{action_chunk.shape[0]} targets for "
                    f"{len(input_spec.action_target_time_offsets_s)} action_target_time_offsets_s."
                )
            if not np.isfinite(action_chunk).all():
                raise ValueError("Policy returned nonfinite LIBERO actions.")
            trajectory = _execution.AbsoluteEefTrajectory(
                np.asarray(input_spec.action_target_time_offsets_s, dtype=np.float64),
                action_chunk,
                arm_count=1,
                initial_pose=_execution.libero_state_to_canonical(robot_state),
            )
            trajectory_elapsed_s = 0.0

        desired_target = trajectory.sample(trajectory_elapsed_s + control_dt_s)
        live_canonical_pose = _execution.libero_state_to_canonical(_libero_raw_state(obs))
        action = osc_adapter.adapt(live_canonical_pose, desired_target)
        obs, _, done, _ = env.step(action.tolist())
        trajectory_elapsed_s += control_dt_s
        simulation_time_s += control_dt_s
        if temporal_keys:
            image_history.append(_preprocess_image(obs["agentview_image"], 256), simulation_time_s)
        if done:
            return True, replay_images
    return False, replay_images


def _format_report_table(report: dict[str, Any]) -> str:
    pooled = report["pooled"]
    grid = report["per_suite"]
    suites = report["suites"]

    cols = [COL_NAMES[category] for category in PERTURBATION_CATEGORIES] + ["Suite Total"]
    header = f"{'Suite':18s}" + "".join(f"{column:>12s}" for column in cols)
    sep = "-" * len(header)

    lines = ["=" * len(header), "LIBERO-plus Report", "=" * len(header), header, sep]
    for suite in suites:
        row = f"{suite:18s}"
        for column in cols:
            value = grid[suite][column]["rate"]
            row += f"{value:11.1f}%" if value is not None else f"{'N/A':>12s}"
        lines.append(row)
    lines.append(sep)

    row = f"{'Pooled':18s}"
    for column in [COL_NAMES[category] for category in PERTURBATION_CATEGORIES] + ["Total"]:
        value = pooled[column]["rate"]
        row += f"{value:11.1f}%" if value is not None else f"{'N/A':>12s}"
    lines.append(row)
    lines.append("=" * len(header))
    return "\n".join(lines)


def _format_suite_log(suite_name: str, category_results: dict[str, list[bool]], totals: dict[str, int]) -> str:
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"\n[{timestamp}] Suite: {suite_name}"]
    if totals["episodes"]:
        lines.append(
            f"  Success Rate: {totals['successes']}/{totals['episodes']} "
            f"({totals['successes'] / totals['episodes']:.2%})"
        )
    for category in PERTURBATION_CATEGORIES:
        results = category_results.get(category, [])
        if results:
            lines.append(
                f"  {COL_NAMES[category]:12s}: {sum(results) / len(results) * 100:5.1f}%  ({sum(results)}/{len(results)})"
            )
        else:
            lines.append(f"  {COL_NAMES[category]:12s}: N/A")
    return "\n".join(lines) + "\n"


def _build_report(
    suite_results: dict[str, dict[str, list[bool]]], suite_totals: dict[str, dict[str, int]]
) -> dict[str, Any]:
    grid: dict[str, dict[str, dict[str, Any]]] = {}
    for suite, category_results in suite_results.items():
        grid[suite] = {}
        for category in PERTURBATION_CATEGORIES:
            results = category_results[category]
            grid[suite][COL_NAMES[category]] = {
                "success": sum(results),
                "total": len(results),
                "rate": round(_rate(results), 1) if results else None,
            }
        totals = suite_totals[suite]
        grid[suite]["Suite Total"] = {
            "success": totals["successes"],
            "total": totals["episodes"],
            "rate": round(totals["successes"] / totals["episodes"] * 100, 1) if totals["episodes"] else None,
        }

    pooled = {}
    for category in PERTURBATION_CATEGORIES:
        merged: list[bool] = []
        for category_results in suite_results.values():
            merged.extend(category_results[category])
        pooled[COL_NAMES[category]] = {
            "success": sum(merged),
            "total": len(merged),
            "rate": round(_rate(merged), 1) if merged else None,
        }
    total_successes = sum(totals["successes"] for totals in suite_totals.values())
    total_episodes = sum(totals["episodes"] for totals in suite_totals.values())
    pooled["Total"] = {
        "success": total_successes,
        "total": total_episodes,
        "rate": round(total_successes / total_episodes * 100, 1) if total_episodes else None,
    }
    return {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "suites": list(suite_results.keys()),
        "per_suite": grid,
        "pooled": pooled,
    }


def _save_report(report: dict[str, Any], log_dir: pathlib.Path) -> None:
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = log_dir / f"results_{timestamp}.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    text_path = log_dir / f"results_{timestamp}.txt"
    with open(text_path, "w", encoding="utf-8") as handle:
        handle.write(_format_report_table(report))
        paper_columns = [COL_NAMES[category] for category in PERTURBATION_CATEGORIES] + ["Total"]
        handle.write("\n\nMarkdown Table:\n")
        handle.write("| Model | " + " | ".join(paper_columns) + " |\n")
        handle.write("|" + "|".join(["---"] * (len(paper_columns) + 1)) + "|\n")
        values = [
            f"{report['pooled'][column]['rate']:.1f}" if report["pooled"][column]["rate"] is not None else "N/A"
            for column in paper_columns
        ]
        handle.write("| **Ours** | " + " | ".join(values) + " |\n")

    logging.info("Results saved to: %s", text_path)


def _log_results_block(report: dict[str, Any]) -> None:
    logging.info("[RESULTS_TABLE_START]")
    for suite in report["suites"]:
        suite_total = report["per_suite"][suite]["Suite Total"]
        rate = suite_total["rate"]
        rate_str = f"{rate:.1f}" if rate is not None else "N/A"
        logging.info(
            "suite_total=%s|%d/%d|%s",
            suite,
            suite_total["success"],
            suite_total["total"],
            rate_str,
        )
    pooled_total = report["pooled"]["Total"]
    pooled_rate = pooled_total["rate"]
    pooled_rate_str = f"{pooled_rate:.1f}" if pooled_rate is not None else "N/A"
    logging.info(
        "pooled_total=%d/%d|%s",
        pooled_total["success"],
        pooled_total["total"],
        pooled_rate_str,
    )
    logging.info("[RESULTS_TABLE_END]")


def eval_libero_plus(args: Args) -> None:
    for name in ("resize_size", "replan_steps", "num_trials_per_task", "ensemble_window"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if args.action_ensembling:
        raise ValueError(
            "action_ensembling=True is unsupported: the previous normalized-command "
            "ensembling implementation is incompatible with absolute EEF target trajectories."
        )
    if args.num_steps_wait < 0 or args.seed < 0:
        raise ValueError("num_steps_wait and seed must be nonnegative.")
    if args.record_video not in {"none", "failure", "all"}:
        raise ValueError(f"record_video must be one of none/failure/all, got {args.record_video!r}.")
    if not math.isfinite(args.ensemble_decay) or args.ensemble_decay < 0:
        raise ValueError("ensemble_decay must be finite and nonnegative.")
    for name in ("connect_timeout", "inference_timeout"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    np.random.seed(args.seed)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    result_root = pathlib.Path(args.result_root)
    result_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_dir = result_root / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "eval.log"
    checkpoint_path = pathlib.Path(args.checkpoint_path) if args.checkpoint_path else result_root / "checkpoint.json"
    video_dir = None
    if args.record_video != "none":
        video_dir = pathlib.Path(args.video_out_path)
        video_dir.mkdir(parents=True, exist_ok=True)

    suite_names = [args.task_suite_name] if args.task_suite_name else list(args.task_suite_names)
    benchmark_dict = benchmark.get_benchmark_dict()
    for suite_name in suite_names:
        if suite_name not in benchmark_dict:
            raise ValueError(f"Unknown task suite: {suite_name}")

    records = _load_checkpoint(checkpoint_path, args.seed)
    suite_results, suite_totals, done_keys = _rebuild_state(records)
    client = _websocket_client_policy.WebsocketClientPolicy(
        args.host, args.port, connect_timeout=args.connect_timeout, inference_timeout=args.inference_timeout
    )
    try:
        input_spec = _load_policy_input_spec(client.get_server_metadata())
        logging.info("Resolved policy input spec: %s", input_spec)

        for suite_name in suite_names:
            task_suite = benchmark_dict[suite_name]()
            task_classification = _load_task_classification(suite_name, args.task_classification_path)
            selected_task_ids = _select_task_ids(task_suite, args.task_ids)

            suite_results.setdefault(suite_name, {category: [] for category in PERTURBATION_CATEGORIES})
            suite_totals.setdefault(suite_name, {"successes": 0, "episodes": 0})
            category_results = suite_results[suite_name]
            totals = suite_totals[suite_name]

            completed = sum(
                1
                for task_id in selected_task_ids
                for episode_idx in range(args.num_trials_per_task)
                if f"{suite_name}|{task_id}|{episode_idx}" in done_keys
            )
            total_episodes = len(selected_task_ids) * args.num_trials_per_task
            if completed >= total_episodes:
                logging.info("\n[%s] Already completed (%d/%d), skipping.", suite_name, completed, total_episodes)
                continue

            logging.info(
                "\n%s\nSuite: %s (%d/%d tasks selected, %d episodes done)\n%s",
                "=" * 50,
                suite_name,
                len(selected_task_ids),
                task_suite.n_tasks,
                completed,
                "=" * 50,
            )

            max_steps = TASK_MAX_STEPS.get(suite_name, 300)
            start_time = datetime.datetime.now(datetime.timezone.utc)
            tasks_run = 0

            for task_id in selected_task_ids:
                if all(
                    f"{suite_name}|{task_id}|{episode_idx}" in done_keys
                    for episode_idx in range(args.num_trials_per_task)
                ):
                    continue

                task = task_suite.get_task(task_id)
                initial_states = task_suite.get_task_init_states(task_id)
                if args.num_trials_per_task > len(initial_states):
                    raise ValueError(
                        f"Task {task_id} has {len(initial_states)} initial states; requested {args.num_trials_per_task} trials."
                    )
                env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
                metadata = task_classification.get(task.name)
                category = metadata.get("category") if metadata else None

                try:
                    for episode_idx in range(args.num_trials_per_task):
                        record_key = f"{suite_name}|{task_id}|{episode_idx}"
                        if record_key in done_keys:
                            continue

                        success, replay_images = run_single_episode(
                            env,
                            str(task_description),
                            client,
                            initial_states[episode_idx],
                            args,
                            input_spec,
                            max_steps,
                        )

                        should_save_video = (
                            video_dir is not None
                            and replay_images
                            and (args.record_video == "all" or (args.record_video == "failure" and not success))
                        )
                        if should_save_video:
                            result = "success" if success else "failure"
                            safe_task_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", task.name).strip("_")[:120] or "task"
                            video_path = (
                                video_dir
                                / (
                                    f"{suite_name}_task{task_id}_seed{args.seed}_{safe_task_name}"
                                    f"_episode{episode_idx}_{result}.mp4"
                                )
                            )
                            imageio.mimwrite(
                                video_path,
                                [np.asarray(image) for image in replay_images],
                                fps=10,
                                macro_block_size=1,
                            )
                            logging.info("Video saved to: %s", video_path)

                        if success:
                            totals["successes"] += 1
                        totals["episodes"] += 1
                        if category and category in category_results:
                            category_results[category].append(success)

                        records.append(
                            {
                                "suite": suite_name,
                                "task_id": task_id,
                                "episode_idx": episode_idx,
                                "task_name": task.name,
                                "category": category,
                                "success": success,
                            }
                        )
                        done_keys.add(record_key)
                        _save_checkpoint(checkpoint_path, records, args.seed)

                finally:
                    env.close()
                tasks_run += 1
                remaining = max(total_episodes - totals["episodes"], 0)
                elapsed = max((datetime.datetime.now(datetime.timezone.utc) - start_time).total_seconds(), 1.0)
                avg_seconds = elapsed / max(tasks_run, 1)
                eta = str(datetime.timedelta(seconds=int(avg_seconds * remaining)))
                acc = totals["successes"] / totals["episodes"] if totals["episodes"] else 0.0
                category_tag = f" [{category}]" if category else ""
                logging.info(
                    "[%s] %d/%d | Task %d%s | Acc=%.1f%% | ETA %s",
                    suite_name,
                    totals["episodes"],
                    total_episodes,
                    task_id,
                    category_tag,
                    acc * 100.0,
                    eta,
                )

            if totals["episodes"]:
                logging.info(
                    "\n[%s] %d/%d (%.2f%%)",
                    suite_name,
                    totals["successes"],
                    totals["episodes"],
                    totals["successes"] / totals["episodes"] * 100.0,
                )
            for category in PERTURBATION_CATEGORIES:
                results = category_results[category]
                if results:
                    logging.info(
                        "  %-12s: %5.1f%% (%d/%d)", COL_NAMES[category], _rate(results), sum(results), len(results)
                    )
            with open(log_file, "a", encoding="utf-8") as handle:
                handle.write(_format_suite_log(suite_name, category_results, totals))

        report = _build_report(suite_results, suite_totals)
        table = _format_report_table(report)
        logging.info("\n%s", table)
        _log_results_block(report)
        _save_report(report, log_dir)
        with open(log_file, "a", encoding="utf-8") as handle:
            handle.write(f"\n{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M:%S}\n{table}\n")
        logging.info("\nCheckpoint: %s\nLog: %s", checkpoint_path, log_file)
    finally:
        client.close()


if __name__ == "__main__":
    eval_libero_plus(tyro.cli(Args))
