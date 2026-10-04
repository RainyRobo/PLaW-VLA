from __future__ import annotations

import collections
import dataclasses
import gc
import logging
import math
import multiprocessing as mp
import os
import pathlib
import sys
import time
import traceback
import warnings
from collections.abc import Sequence
from typing import Any, List, Tuple

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["NUMBA_DISABLE_PERFORMANCE_WARNINGS"] = "1"
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import imageio
import numpy as np
import tyro

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = pathlib.Path(__file__).resolve().parent
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
    config_root = pathlib.Path(os.environ.get("LIBERO_CONFIG_PATH", EXAMPLE_ROOT / ".libero")).expanduser()
    if not config_root.is_absolute():
        config_root = PROJECT_ROOT / config_root

    os.environ["LIBERO_CONFIG_PATH"] = str(config_root)
    config_root.mkdir(parents=True, exist_ok=True)
    (config_root / "config.yaml").write_text(
        "\n".join(
            (
                f"benchmark_root: {LIBERO_ROOT}",
                f"bddl_files: {LIBERO_ROOT / 'bddl_files'}",
                f"init_states: {LIBERO_ROOT / 'init_files'}",
                f"datasets: {LIBERO_DATASETS_ROOT}",
                f"assets: {LIBERO_ROOT / 'assets'}",
            )
        )
        + "\n",
        encoding="utf-8",
    )


MPLCONFIGDIR_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR_ROOT))
_ensure_libero_config()

from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

from openpi.serving import policy_input_spec as _policy_input_spec

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
# Compatibility constants for checkpoints trained against the legacy
# ``libero_v3_eef`` layout.  That dataset stores the scalar gripper state in
# metres (roughly 0..0.04) and the action as a signed robosuite command
# (-1=open, +1=close).  The current policy transform otherwise presents a
# normalized 0..1 state and reverses that command while decoding it as a
# canonical gripper delta.
LEGACY_LIBERO_GRIPPER_OPENNESS_SCALE = 0.04
LIBERO_GRIPPER_ACTION_INDEX = 6
TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8001
    resize_size: int = 224
    replan_steps: int = 5

    task_suite_name: str = "libero_spatial"
    task_ids: Tuple[int, ...] = ()
    num_steps_wait: int = 10
    num_trials_per_task: int = 50

    # Optional local training-config name. Only used as a fallback to build the policy
    # input spec when the server's metadata does not include one (e.g. older servers).
    # In normal operation the spec — including history step offsets and stride — is
    # sourced from the server's `input_spec` metadata.
    policy_config: str | None = None

    record_video: str = "failure"
    video_out_path: str = "results/libero/videos"
    seed: int = 7

    # Evaluate checkpoints trained with the legacy ``libero_v3_eef`` gripper
    # representation.  This is intentionally client-side and opt-in so it
    # cannot silently alter results for correctly converted datasets.
    legacy_gripper_compat: bool = False


_SUPPORTED_LIBERO_FRONT_KEYS = {
    "image",
    "front_image",
    "observation/image",
    "observation/front_image",
}
_SUPPORTED_LIBERO_WRIST_KEYS = {
    "wrist_image",
    "observation/wrist_image",
}


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
    return image_tools.convert_to_uint8(
        image_tools.resize_with_pad(img_processed, target_size, target_size)
    )


def _history_buffer_len(step_offsets: Sequence[int]) -> int:
    if not step_offsets:
        return 1
    return 1 + max(-min(step_offsets), 0)


def _select_temporal_frames(frames: Sequence[np.ndarray], step_offsets: Sequence[int]) -> np.ndarray:
    if not frames:
        raise ValueError("Expected at least one frame in the history buffer.")

    newest_idx = len(frames) - 1
    min_available_offset = -newest_idx
    available_offsets = [offset for offset in step_offsets if offset >= min_available_offset]
    if not available_offsets:
        available_offsets = [0]

    return np.stack([frames[newest_idx + offset] for offset in available_offsets], axis=0)


def _load_policy_input_spec(
    metadata: dict[str, Any],
    policy_config: str | None,
) -> _policy_input_spec.PolicyInputSpec:
    input_spec = _policy_input_spec.parse_policy_input_spec(metadata)
    if input_spec is None and policy_config is not None:
        try:
            from openpi.training import config as _config

            input_spec = _policy_input_spec.build_from_train_config(_config.get_config(policy_config))
        except Exception as exc:
            raise RuntimeError(
                "Failed to resolve the local policy_config fallback. "
                "Use a server started from the updated serve_policy.py, or run the client in an environment "
                "with the full PLaW-VLA training dependencies installed."
            ) from exc
    if input_spec is None:
        raise RuntimeError(
            "Policy server metadata is missing the 'input_spec' field and no --policy-config "
            "fallback was provided. Restart the server with an updated serve_policy.py, or pass "
            "--policy-config <train_config_name> so the client can build the spec locally."
        )

    if input_spec.family != "libero":
        raise ValueError(f"Unsupported LIBERO input family: {input_spec.family!r}")
    if len(input_spec.temporal_image_keys) > 1:
        raise ValueError(
            "LIBERO evaluation currently supports at most one temporal image input, "
            f"got {input_spec.temporal_image_keys}."
        )
    for image_key in input_spec.image_keys:
        if image_key not in _SUPPORTED_LIBERO_FRONT_KEYS and image_key not in _SUPPORTED_LIBERO_WRIST_KEYS:
            raise ValueError(f"Unsupported LIBERO image key {image_key!r} in policy input spec.")
    for temporal_key in input_spec.temporal_image_keys:
        if temporal_key not in _SUPPORTED_LIBERO_FRONT_KEYS:
            raise ValueError(
                "LIBERO evaluation only supports temporal histories for the front camera, "
                f"got {temporal_key!r}."
            )

    if input_spec.state_gripper_format not in {None, "two_finger_qpos"}:
        raise ValueError(
            "Policy/client LIBERO state gripper contract mismatch: server expects "
            f"{input_spec.state_gripper_format!r}, client provides 'two_finger_qpos'."
        )
    if input_spec.action_gripper_format not in {None, "signed_command"}:
        raise ValueError(
            "Policy/client LIBERO action gripper contract mismatch: server returns "
            f"{input_spec.action_gripper_format!r}, environment expects 'signed_command'."
        )

    return input_spec


def _libero_image_for_key(key: str, front_image: np.ndarray, wrist_image: np.ndarray) -> np.ndarray:
    if key in _SUPPORTED_LIBERO_WRIST_KEYS:
        return wrist_image
    if key in _SUPPORTED_LIBERO_FRONT_KEYS:
        return front_image
    raise ValueError(f"Unsupported LIBERO image key {key!r}.")


def _legacy_gripper_state_for_policy(robot_state: np.ndarray) -> np.ndarray:
    """Make online LIBERO state match the legacy checkpoint's training scale.

    The server canonicalizes the two finger positions by collapsing them and
    dividing by 0.04.  Scaling both raw finger positions by 0.04 here cancels
    that division, so the server receives the 0..0.04 scalar used to compute
    this checkpoint's normalization statistics.
    """
    state = np.asarray(robot_state, dtype=np.float32).copy()
    if state.ndim != 1 or state.shape[0] < 8:
        raise ValueError(f"Expected LIBERO robot state with at least 8 values, got {state.shape}.")
    state[-2:] *= LEGACY_LIBERO_GRIPPER_OPENNESS_SCALE
    return state


def _legacy_gripper_actions_for_env(action_chunk: np.ndarray) -> np.ndarray:
    """Restore the signed robosuite gripper command learned by the checkpoint."""
    actions = np.asarray(action_chunk).copy()
    if actions.ndim != 2 or actions.shape[1] <= LIBERO_GRIPPER_ACTION_INDEX:
        raise ValueError(f"Expected LIBERO action chunk shaped [horizon, >=7], got {actions.shape}.")
    actions[:, LIBERO_GRIPPER_ACTION_INDEX] *= -1
    return actions


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

    action_plan = collections.deque()
    queue_image_history = collections.deque(maxlen=_history_buffer_len(input_spec.history_step_offsets))
    replay_images: List[np.ndarray] = []

    max_steps = TASK_MAX_STEPS.get(args.task_suite_name, 300)

    # During warm-up we step the simulator with a no-op action so gravity, contacts, etc.
    # settle. We also feed every settled observation into the history queue so that the
    # first real inference call sees a full-length history that matches the training
    # distribution (training always provides `history_num_frames` valid frames; without
    # this pre-fill the first 2 infers would see only 1 / 3 frames).
    for _ in range(args.num_steps_wait):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
        queue_image_history.append(preprocess_image(obs["agentview_image"], 256))

    success = False
    try:
        for _ in range(max_steps):
            front_image = preprocess_image(obs["agentview_image"], args.resize_size)
            wrist_image = preprocess_image(obs["robot0_eye_in_hand_image"], args.resize_size)
            front_history_image = preprocess_image(obs["agentview_image"], 256)

            replay_images.append(front_image)
            queue_image_history.append(front_history_image)

            if not action_plan:
                robot_state = np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                )
                if args.legacy_gripper_compat:
                    robot_state = _legacy_gripper_state_for_policy(robot_state)

                request = {}
                temporal_keys = set(input_spec.temporal_image_keys)
                for image_key in input_spec.image_keys:
                    if image_key in temporal_keys:
                        request[image_key] = _select_temporal_frames(
                            list(queue_image_history),
                            input_spec.history_step_offsets,
                        )
                    else:
                        request[image_key] = _libero_image_for_key(image_key, front_image, wrist_image)
                if input_spec.state_key is not None:
                    request[input_spec.state_key] = robot_state
                if input_spec.prompt_key is not None:
                    request[input_spec.prompt_key] = task_description

                action_chunk = np.asarray(client.infer(request)["actions"])
                if action_chunk.ndim != 2 or action_chunk.shape[1] < len(LIBERO_DUMMY_ACTION):
                    raise ValueError(
                        "Policy server returned invalid action chunk shape "
                        f"{action_chunk.shape}; expected [horizon, >= {len(LIBERO_DUMMY_ACTION)}]."
                    )
                if args.legacy_gripper_compat:
                    action_chunk = _legacy_gripper_actions_for_env(action_chunk)
                action_plan.extend(action_chunk[: args.replan_steps])

            action = action_plan.popleft()
            obs, _, done, _ = env.step(action.tolist())

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
    task_suite_name: str,
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
    devnull = open(os.devnull, "w", encoding="utf-8")
    sys.stdout = devnull
    sys.stderr = devnull
    _, _, OffScreenRenderEnv = _lazy_imports()

    try:
        args = Args(**args_dict)
        env = OffScreenRenderEnv(
            bddl_file_name=task_bddl_file,
            camera_heights=LIBERO_ENV_RESOLUTION,
            camera_widths=LIBERO_ENV_RESOLUTION,
        )
        env.seed(args.seed)
        client = _websocket_client_policy.WebsocketClientPolicy(host, port)
        input_spec = _load_policy_input_spec(client.get_server_metadata(), args.policy_config)

        is_success, replay_images = run_single_episode(
            env=env,
            task_description=task_description,
            client=client,
            initial_state=initial_state,
            args=args,
            input_spec=input_spec,
        )
        env.close()
        del env
        gc.collect()

        should_save = bool(video_path) and replay_images and (
            args.record_video == "all" or (args.record_video == "failure" and not is_success)
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
    timeout: int = 600,
) -> bool:
    args_dict = {field.name: getattr(args, field.name) for field in dataclasses.fields(args)}
    args_dict["task_suite_name"] = task_suite_name

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(
        target=_episode_worker,
        args=(
            task_suite_name,
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
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        _log("[WARN] Episode subprocess timed out; killing it and treating the rollout as failure.")
        proc.kill()
        proc.join()
        return False

    if proc.exitcode != 0:
        _log(f"[WARN] Episode subprocess exited with code {proc.exitcode}; treating the rollout as failure.")
        return False

    if result_queue.empty():
        return False

    result = result_queue.get_nowait()
    # Accept the historical bool payload for compatibility with older workers.
    if isinstance(result, tuple):
        is_success, error_text = result
        if error_text:
            _log("[ERROR] Episode subprocess failed:\n" + error_text.rstrip())
        return bool(is_success)
    return bool(result)


def eval_libero(args: Args) -> None:
    if not (LIBERO_SRC_ROOT / "libero").is_dir():
        raise FileNotFoundError(
            f"LIBERO submodule is missing at {LIBERO_SRC_ROOT}. "
            "Run: git submodule update --init --recursive"
        )
    mp.set_start_method("spawn", force=True)
    np.random.seed(args.seed)

    if args.replan_steps <= 0:
        raise ValueError(f"replan_steps must be > 0, got {args.replan_steps}")
    if args.num_trials_per_task <= 0:
        raise ValueError(f"num_trials_per_task must be > 0, got {args.num_trials_per_task}")
    if args.record_video not in {"none", "failure", "all"}:
        raise ValueError(f"record_video must be one of none/failure/all, got {args.record_video!r}")

    benchmark, get_libero_path, _ = _lazy_imports()

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    server_metadata = client.get_server_metadata()
    input_spec = _load_policy_input_spec(server_metadata, args.policy_config)
    legacy_required = bool(server_metadata.get("legacy_gripper_compat_required", False))
    if legacy_required != args.legacy_gripper_compat:
        required_flag = "--legacy-gripper-compat" if legacy_required else "--no-legacy-gripper-compat"
        raise RuntimeError(
            "Policy/client legacy gripper mode mismatch: "
            f"server legacy_required={legacy_required}, client legacy_enabled={args.legacy_gripper_compat}. "
            f"Restart the client with {required_flag}."
        )
    _log(f"Server metadata: {server_metadata}")
    _log(f"Resolved policy input spec: {input_spec}")
    if args.legacy_gripper_compat:
        _log(
            "[compat] legacy LIBERO gripper mode enabled: policy state uses the training-time "
            "0..0.04 scale and decoded gripper commands are sign-corrected."
        )

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
        trial_count = min(args.num_trials_per_task, len(initial_states))
        if trial_count < args.num_trials_per_task:
            _log(
                f"[WARN] Task {task_id} only provides {len(initial_states)} initial states; "
                f"clamping from {args.num_trials_per_task} to {trial_count}."
            )

        task_successes = 0
        for episode_idx in range(trial_count):
            video_path = ""
            if video_dir is not None:
                safe_task_desc = task_description.replace(" ", "_")
                video_path = str(video_dir / f"rollout_{safe_task_desc}_{episode_idx}_pending.mp4")

            is_success = _run_episode_in_subprocess(
                task_suite_name=args.task_suite_name,
                task_bddl_file=task_bddl_file,
                task_description=task_description,
                initial_state=initial_states[episode_idx],
                host=args.host,
                port=args.port,
                args=args,
                video_path=video_path,
            )

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
