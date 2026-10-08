# Derived from openpi (Copyright 2024 Physical Intelligence, Inc.; Apache-2.0).
# Modified for PLaW-VLA by the PLaW-VLA authors, 2026.
# Simulator paths and rendering settings must precede third-party imports.
# ruff: noqa: E402
from __future__ import annotations

import collections
from collections.abc import Sequence
import contextlib
import dataclasses
import gc
import logging
import math
import multiprocessing as mp
import os
import pathlib
import queue
import re
import sys
import tempfile
import time
import traceback
from typing import Any, List, Tuple
import warnings

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["NUMBA_DISABLE_PERFORMANCE_WARNINGS"] = "1"
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import imageio
import numpy as np
import tyro

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
LIBERO_SRC_ROOT = PROJECT_ROOT / "third_party" / "libero"
LIBERO_ROOT = LIBERO_SRC_ROOT / "libero" / "libero"
LIBERO_DATASETS_ROOT = LIBERO_SRC_ROOT / "libero" / "datasets"
MPLCONFIGDIR_ROOT = pathlib.Path("/tmp/matplotlib")
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(LIBERO_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(LIBERO_SRC_ROOT))


def _ensure_libero_config() -> None:
    config_root = pathlib.Path(os.environ.get("LIBERO_CONFIG_PATH", "~/.cache/robot_policy/libero")).expanduser()
    if not config_root.is_absolute():
        config_root = PROJECT_ROOT / config_root

    os.environ["LIBERO_CONFIG_PATH"] = str(config_root)
    config_root.mkdir(parents=True, exist_ok=True)
    config_text = (
        "\n".join(
            (
                f"benchmark_root: {LIBERO_ROOT}",
                f"bddl_files: {LIBERO_ROOT / 'bddl_files'}",
                f"init_states: {LIBERO_ROOT / 'init_files'}",
                f"datasets: {LIBERO_DATASETS_ROOT}",
                f"assets: {LIBERO_ROOT / 'assets'}",
            )
        )
        + "\n"
    )
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=config_root, prefix=".config-", suffix=".yaml", delete=False
    ) as config_file:
        temp_path = pathlib.Path(config_file.name)
        config_file.write(config_text)
    try:
        temp_path.replace(config_root / "config.yaml")
    finally:
        temp_path.unlink(missing_ok=True)


MPLCONFIGDIR_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR_ROOT))
_ensure_libero_config()

from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

from openpi.serving import policy_input_spec as _policy_input_spec

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclasses.dataclass
class Args:
    host: str = "localhost"
    port: int = 8001
    resize_size: int = 224
    replan_steps: int = 5
    connect_timeout: float = 30.0
    inference_timeout: float = 60.0
    # Total wall-clock budget for initialization and a complete rollout.
    episode_timeout: float = 600.0

    task_suite_name: str = "libero_spatial"
    task_ids: Tuple[int, ...] = ()
    num_steps_wait: int = 10
    num_trials_per_task: int = 50

    record_video: str = "failure"
    video_out_path: str = "results/libero/videos"
    seed: int = 42


_LIBERO_FRONT_KEY = "observation/image"
_LIBERO_WRIST_KEY = "observation/wrist_image"


def _suppress_warnings() -> None:
    import warnings as _warnings

    _warnings.filterwarnings("ignore")
    _warnings.filterwarnings("ignore", category=DeprecationWarning)
    os.environ["PYTHONWARNINGS"] = "ignore"
    logging.getLogger("robosuite_logs").setLevel(logging.CRITICAL)
    logging.getLogger("numba").setLevel(logging.CRITICAL)
    logging.getLogger("OpenGL").setLevel(logging.CRITICAL)


def _log(msg: str) -> None:
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def _progress(msg: str) -> None:
    sys.stderr.write(f"\r\033[K{msg}")
    sys.stderr.flush()


def _lazy_imports():
    _suppress_warnings()
    from libero.libero import benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    return benchmark, get_libero_path, OffScreenRenderEnv


def preprocess_image(img: np.ndarray, target_size: int) -> np.ndarray:
    img_processed = np.ascontiguousarray(img[::-1, ::-1])
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(img_processed, target_size, target_size))


def _history_buffer_len(step_offsets: Sequence[int]) -> int:
    if not step_offsets:
        return 1
    return 1 + max(-min(step_offsets), 0)


def _select_temporal_frames(frames: Sequence[np.ndarray], step_offsets: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Sample the requested history, masking offsets before the first observation."""
    if not frames:
        raise ValueError("Expected at least one frame in the history buffer.")

    newest_idx = len(frames) - 1
    frame_indices = [newest_idx + offset for offset in step_offsets]
    return (
        np.stack([frames[max(index, 0)] for index in frame_indices], axis=0),
        np.asarray([index < 0 for index in frame_indices], dtype=bool),
    )


def _load_policy_input_spec(metadata: dict[str, Any]) -> _policy_input_spec.PolicyInputSpec:
    input_spec = _policy_input_spec.parse_policy_input_spec(metadata)
    if input_spec is None:
        raise RuntimeError(
            "Policy server metadata is missing 'input_spec'. Start the server with scripts/serve_policy.py."
        )

    if input_spec.family != "libero":
        raise ValueError(f"Unsupported LIBERO input family: {input_spec.family!r}")
    if not input_spec.image_keys:
        raise ValueError("LIBERO policy input_spec must include image_keys.")
    if len(input_spec.temporal_image_keys) > 1:
        raise ValueError(
            "LIBERO evaluation currently supports at most one temporal image input, "
            f"got {input_spec.temporal_image_keys}."
        )
    for image_key in input_spec.image_keys:
        if image_key not in {_LIBERO_FRONT_KEY, _LIBERO_WRIST_KEY}:
            raise ValueError(f"Unsupported LIBERO image key {image_key!r} in policy input spec.")
    for temporal_key in input_spec.temporal_image_keys:
        if temporal_key != _LIBERO_FRONT_KEY:
            raise ValueError(
                f"LIBERO evaluation only supports temporal histories for the front camera, got {temporal_key!r}."
            )
    if not set(input_spec.temporal_image_keys).issubset(input_spec.image_keys):
        raise ValueError("Temporal image keys must be included in image_keys.")
    offsets = input_spec.history_step_offsets
    if input_spec.temporal_image_keys and (not offsets or offsets[-1] != 0 or offsets != tuple(sorted(set(offsets)))):
        raise ValueError("History step offsets must be strictly increasing and end in zero.")

    if input_spec.state_gripper_format != "two_finger_qpos":
        raise ValueError(
            "Policy/client LIBERO state gripper contract mismatch: server expects "
            f"{input_spec.state_gripper_format!r}, client provides 'two_finger_qpos'."
        )
    if input_spec.action_gripper_format != "signed_command":
        raise ValueError(
            "Policy/client LIBERO action gripper contract mismatch: server returns "
            f"{input_spec.action_gripper_format!r}, environment expects 'signed_command'."
        )

    return input_spec


def _libero_image_for_key(key: str, front_image: np.ndarray, wrist_image: np.ndarray) -> np.ndarray:
    if key == _LIBERO_WRIST_KEY:
        return wrist_image
    if key == _LIBERO_FRONT_KEY:
        return front_image
    raise ValueError(f"Unsupported LIBERO image key {key!r}.")


def run_single_episode(
    env,
    task_description: str,
    client: _websocket_client_policy.WebsocketClientPolicy,
    initial_state: np.ndarray,
    args: Args,
    input_spec: _policy_input_spec.PolicyInputSpec,
) -> Tuple[bool, List[np.ndarray]]:
    env.reset()
    obs = env.set_init_state(initial_state)

    temporal_keys = set(input_spec.temporal_image_keys)
    action_plan = collections.deque()
    queue_image_history = collections.deque(maxlen=_history_buffer_len(input_spec.history_step_offsets))
    if temporal_keys:
        queue_image_history.append(preprocess_image(obs["agentview_image"], 256))
    replay_images: List[np.ndarray] = []

    max_steps = TASK_MAX_STEPS.get(args.task_suite_name, 300)

    # Let contacts settle while collecting history at the simulator timestep.
    # The policy samples this buffer at its declared history step offsets.
    for _ in range(args.num_steps_wait):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
        if temporal_keys:
            queue_image_history.append(preprocess_image(obs["agentview_image"], 256))

    success = False
    try:
        for _ in range(max_steps):
            front_image = preprocess_image(obs["agentview_image"], args.resize_size)
            wrist_image = preprocess_image(obs["robot0_eye_in_hand_image"], args.resize_size)
            if args.record_video != "none":
                replay_images.append(front_image)

            if not action_plan:
                robot_state = np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                ).astype(np.float32)
                if robot_state.shape != (8,) or not np.all(np.isfinite(robot_state)):
                    raise ValueError(f"Expected a finite 8D raw LIBERO state, got {robot_state!r}.")
                request = {}
                for image_key in input_spec.image_keys:
                    if image_key in temporal_keys:
                        request[image_key], request[f"{image_key}_is_pad"] = _select_temporal_frames(
                            list(queue_image_history),
                            input_spec.history_step_offsets,
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
                    or action_chunk.shape[1] != len(LIBERO_DUMMY_ACTION)
                ):
                    raise ValueError(
                        "Policy server returned invalid action chunk shape "
                        f"{action_chunk.shape}; expected [positive horizon, {len(LIBERO_DUMMY_ACTION)}]."
                    )
                if not np.all(np.isfinite(action_chunk)):
                    raise ValueError("Policy server returned non-finite LIBERO actions.")
                action_plan.extend(action_chunk[: args.replan_steps])

            action = action_plan.popleft()
            obs, _, done, _ = env.step(action.tolist())
            if temporal_keys:
                queue_image_history.append(preprocess_image(obs["agentview_image"], 256))

            if done:
                success = True
                break

    except Exception as exc:
        logging.error("Episode failed with error: %s", exc)
        traceback.print_exc()
        # Let the subprocess boundary report the traceback to the parent.  The
        # worker redirects stdout/stderr to keep MuJoCo quiet, so swallowing the
        # exception here otherwise turns infrastructure errors into ordinary
        # policy failures and produces a misleading 0% success rate.
        raise

    return success, replay_images


def _episode_worker(
    task_bddl_file: str,
    task_description: str,
    initial_state: np.ndarray,
    host: str,
    port: int,
    args_dict: dict[str, Any],
    video_path: str,
    result_queue,
) -> None:
    _suppress_warnings()
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                _, _, OffScreenRenderEnv = _lazy_imports()
                args = Args(**args_dict)
                np.random.seed(args.seed)
                env = OffScreenRenderEnv(
                    bddl_file_name=task_bddl_file,
                    camera_heights=LIBERO_ENV_RESOLUTION,
                    camera_widths=LIBERO_ENV_RESOLUTION,
                )
                try:
                    env.seed(args.seed)
                    with _websocket_client_policy.WebsocketClientPolicy(
                        host,
                        port,
                        connect_timeout=args.connect_timeout,
                        inference_timeout=args.inference_timeout,
                    ) as client:
                        input_spec = _load_policy_input_spec(client.get_server_metadata())
                        is_success, replay_images = run_single_episode(
                            env=env,
                            task_description=task_description,
                            client=client,
                            initial_state=initial_state,
                            args=args,
                            input_spec=input_spec,
                        )
                finally:
                    env.close()
                    del env
                    gc.collect()

                should_save = (
                    bool(video_path)
                    and replay_images
                    and (args.record_video == "all" or (args.record_video == "failure" and not is_success))
                )
                if should_save:
                    imageio.mimwrite(video_path, [np.asarray(x) for x in replay_images], fps=10, macro_block_size=1)
        result_queue.put((is_success, ""))
    except Exception:
        result_queue.put((False, traceback.format_exc()))


def _run_episode_in_subprocess(
    task_suite_name: str,
    task_bddl_file: pathlib.Path,
    task_description: str,
    initial_state: np.ndarray,
    host: str,
    port: int,
    args: Args,
    video_path: str = "",
    timeout: float = 600.0,
) -> bool:
    args_dict = {field.name: getattr(args, field.name) for field in dataclasses.fields(args)}
    args_dict["task_suite_name"] = task_suite_name

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(
        target=_episode_worker,
        args=(
            str(task_bddl_file),
            task_description,
            initial_state,
            host,
            port,
            args_dict,
            video_path,
            result_queue,
        ),
    )
    started = False
    try:
        proc.start()
        started = True
        deadline = time.monotonic() + timeout
        # Drain the queue while the worker runs. A large traceback can fill its
        # pipe and keep the worker's feeder thread alive until the parent reads.
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"LIBERO episode exceeded its {timeout:g}s subprocess timeout.")
            try:
                is_success, error_text = result_queue.get(timeout=min(0.1, remaining))
                break
            except queue.Empty as exc:
                if proc.is_alive():
                    continue
                if proc.exitcode != 0:
                    raise RuntimeError(f"LIBERO episode subprocess exited with code {proc.exitcode}.") from exc
                raise RuntimeError("LIBERO episode subprocess exited without reporting a result.") from exc
        proc.join(timeout=max(0, deadline - time.monotonic()))
        if proc.is_alive():
            raise TimeoutError(f"LIBERO episode exceeded its {timeout:g}s subprocess timeout.")
        if proc.exitcode != 0:
            raise RuntimeError(f"LIBERO episode subprocess exited with code {proc.exitcode}.")
        if error_text:
            raise RuntimeError("LIBERO episode subprocess failed:\n" + error_text.rstrip())
        return bool(is_success)
    finally:
        if started and proc.is_alive():
            proc.kill()
            proc.join()
        result_queue.close()
        result_queue.join_thread()
        if started:
            proc.close()


def _validate_args(args: Args) -> None:
    for name in ("port", "resize_size", "replan_steps", "num_trials_per_task"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive, got {getattr(args, name)}.")
    if args.port > 65535:
        raise ValueError(f"port must be at most 65535, got {args.port}.")
    if args.num_steps_wait < 0:
        raise ValueError(f"num_steps_wait must be nonnegative, got {args.num_steps_wait}.")
    for name in ("connect_timeout", "inference_timeout", "episode_timeout"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive, got {value}.")
    if not 0 <= args.seed < 2**32:
        raise ValueError(f"seed must be between 0 and {2**32 - 1}, got {args.seed}.")
    if args.record_video not in {"none", "failure", "all"}:
        raise ValueError(f"record_video must be one of none/failure/all, got {args.record_video!r}.")
    if args.task_suite_name not in TASK_MAX_STEPS:
        raise ValueError(f"Unknown task suite: {args.task_suite_name!r}.")
    if len(set(args.task_ids)) != len(args.task_ids):
        raise ValueError(f"task_ids must not contain duplicates, got {args.task_ids}.")


def eval_libero(args: Args) -> None:
    _validate_args(args)
    if not (LIBERO_SRC_ROOT / "libero").is_dir():
        raise FileNotFoundError(
            f"LIBERO submodule is missing at {LIBERO_SRC_ROOT}. Run: git submodule update --init third_party/libero"
        )
    mp.set_start_method("spawn", force=True)
    np.random.seed(args.seed)

    benchmark, get_libero_path, _ = _lazy_imports()
    benchmark_dict = benchmark.get_benchmark_dict()
    if args.task_suite_name not in benchmark_dict:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    task_suite = benchmark_dict[args.task_suite_name]()
    _log(f"Initialized Task Suite: {args.task_suite_name} with {task_suite.n_tasks} tasks.")

    task_id_list = list(args.task_ids) if args.task_ids else list(range(task_suite.n_tasks))
    for task_id in task_id_list:
        if task_id < 0 or task_id >= task_suite.n_tasks:
            raise ValueError(f"task_id {task_id} out of range [0, {task_suite.n_tasks})")
    _log(f"Running task IDs: {task_id_list}")
    _log(f"Evaluation seed: {args.seed}; requested trials per task: {args.num_trials_per_task}")
    with _websocket_client_policy.WebsocketClientPolicy(
        args.host,
        args.port,
        connect_timeout=args.connect_timeout,
        inference_timeout=args.inference_timeout,
    ) as client:
        server_metadata = client.get_server_metadata()
        input_spec = _load_policy_input_spec(server_metadata)
    _log(f"Server metadata: {server_metadata}")
    _log(f"Resolved policy input spec: {input_spec}")

    video_dir = None
    if args.record_video != "none":
        video_dir = pathlib.Path(args.video_out_path)
        video_dir.mkdir(parents=True, exist_ok=True)

    total_episodes, total_successes = 0, 0
    task_results: list[tuple[int, str, int, int, float]] = []
    num_tasks = len(task_id_list)
    suite_t0 = time.time()

    for task_idx, task_id in enumerate(task_id_list):
        task = task_suite.get_task(task_id)
        task_description = str(task.language)
        initial_states = task_suite.get_task_init_states(task_id)
        task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        if not task_bddl_file.is_file():
            raise FileNotFoundError(f"LIBERO task definition is missing: {task_bddl_file}")
        trial_count = min(args.num_trials_per_task, len(initial_states))
        if trial_count == 0:
            raise RuntimeError(f"LIBERO task {task_id} does not provide any initial states.")
        if trial_count < args.num_trials_per_task:
            _log(
                f"[WARN] Task {task_id} only provides {len(initial_states)} initial states; "
                f"clamping from {args.num_trials_per_task} to {trial_count}."
            )

        task_successes = 0
        for episode_idx in range(trial_count):
            video_path = ""
            if video_dir is not None:
                safe_task_desc = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_description).strip("_")[:120] or "task"
                video_path = str(
                    video_dir / f"task{task_id}_seed{args.seed}_{safe_task_desc}_{episode_idx}_pending.mp4"
                )

            _log(f"Rollout: suite={args.task_suite_name} task={task_id} trial={episode_idx} seed={args.seed}")

            is_success = _run_episode_in_subprocess(
                task_suite_name=args.task_suite_name,
                task_bddl_file=task_bddl_file,
                task_description=task_description,
                initial_state=initial_states[episode_idx],
                host=args.host,
                port=args.port,
                args=args,
                video_path=video_path,
                timeout=args.episode_timeout,
            )
            _log(f"Rollout result: task={task_id} trial={episode_idx} seed={args.seed} success={is_success}")

            if is_success:
                task_successes += 1
                total_successes += 1
            total_episodes += 1

            if video_path:
                pending_path = pathlib.Path(video_path)
                if pending_path.exists():
                    suffix = "success" if is_success else "failure"
                    final_name = pending_path.name.replace("_pending.mp4", f"_{suffix}.mp4")
                    pending_path.rename(pending_path.parent / final_name)

            task_acc = task_successes / (episode_idx + 1)
            total_acc = total_successes / total_episodes
            elapsed = time.time() - suite_t0
            _progress(
                f"  [{args.task_suite_name}] Task {task_idx + 1}/{num_tasks} "
                f"Ep {episode_idx + 1}/{trial_count} task_acc={task_acc:.0%} "
                f"total_acc={total_acc:.0%} ({elapsed:.0f}s)"
            )

        task_rate = task_successes / trial_count if trial_count > 0 else 0.0
        task_results.append((task_id, task_description, task_successes, trial_count, task_rate))
        _log(
            f"Task {task_id} Finished. Success Rate: {task_successes}/{trial_count} "
            f"({task_rate:.1%}) | {task_description}"
        )

    _progress("")

    final_rate = total_successes / total_episodes if total_episodes > 0 else 0.0
    elapsed = time.time() - suite_t0

    _log("=" * 50)
    _log(f"Final Total Success Rate: {final_rate:.4f}  ({total_successes}/{total_episodes})  [{elapsed:.0f}s]")
    _log("=" * 50)
    _log("")
    _log("[RESULTS_TABLE_START]")
    _log(f"suite={args.task_suite_name}")
    for task_id, task_description, successes, episodes, task_rate in task_results:
        _log(f"task={task_id}|{task_description}|{successes}/{episodes}|{task_rate:.4f}")
    _log(f"suite_total={total_successes}/{total_episodes}|{final_rate:.4f}")
    _log("[RESULTS_TABLE_END]")


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    w = float(np.clip(quat[3], -1.0, 1.0))
    den = np.sqrt(1.0 - w * w)
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(w)) / den


if __name__ == "__main__":
    eval_libero(tyro.cli(Args))
