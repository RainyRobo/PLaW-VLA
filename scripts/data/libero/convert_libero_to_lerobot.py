# Copyright 2026 PLaW-VLA authors. SPDX-License-Identifier: Apache-2.0
"""Convert official LIBERO HDF5 demonstrations to canonical LeRobot EEF data."""

from pathlib import Path
import json
import sys

import numpy as np
import tyro


def build_absolute_next_targets(state: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Build next-sampled absolute EEF targets and their validity mask.

    The returned arrays correspond to ``state[::stride]``.  Every valid target
    is the following sampled state.  The final sample has no future target, so
    it is marked invalid and its target is kept equal to its current state.

    Args:
        state: Canonical LIBERO EEF states with shape ``[num_frames, 8]``.
        stride: Positive integer sampling stride in source-frame units.

    Returns:
        A pair ``(targets, validity)`` with shapes ``[num_samples, 8]`` and
        ``[num_samples]``.  Targets preserve the input dtype.
    """
    if not isinstance(state, np.ndarray):
        raise TypeError(f"state must be a numpy.ndarray, got {type(state).__name__}")
    if state.ndim != 2 or state.shape[1] != 8:
        raise ValueError(f"state must have shape [num_frames, 8], got {state.shape}")
    if state.shape[0] == 0:
        raise ValueError("state must contain at least one frame")
    if isinstance(stride, (bool, np.bool_)) or not isinstance(stride, (int, np.integer)):
        raise TypeError(f"stride must be a positive integer, got {type(stride).__name__}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    sampled_state = state[:: int(stride)]
    targets = sampled_state.copy()
    validity = np.zeros(sampled_state.shape[0], dtype=np.bool_)
    if sampled_state.shape[0] > 1:
        targets[:-1] = sampled_state[1:]
        validity[:-1] = True
    return targets, validity


def main(data_dir: Path, output_dir: Path, *, source_fps: int = 20, fps: int = 10) -> None:
    import h5py
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    from plawvla.policies import ee_pose_utils

    files = sorted(data_dir.expanduser().rglob("*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No LIBERO HDF5 demonstrations found in {data_dir}")
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    if fps <= 0 or source_fps < fps or source_fps % fps:
        raise ValueError("source_fps must be a positive integer multiple of fps")
    with h5py.File(files[0], "r") as raw:
        first_obs = raw["data"][next(iter(raw["data"]))]["obs"]
        image_shape = tuple(first_obs["agentview_rgb"].shape[1:])
        wrist_image_shape = tuple(first_obs["eye_in_hand_rgb"].shape[1:])
    names = ["x", "y", "z", "quaternion.w", "quaternion.x", "quaternion.y", "quaternion.z", "gripper.width"]
    features = {
        "observation.images.image": {
            "dtype": "video",
            "shape": image_shape,
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist_image": {
            "dtype": "video",
            "shape": wrist_image_shape,
            "names": ["height", "width", "channels"],
        },
        "observation.state": {"dtype": "float32", "shape": (8,), "names": names},
        "action": {"dtype": "float32", "shape": (8,), "names": names},
        "transition_action_valid": {"dtype": "bool", "shape": (1,)},
    }
    dataset = LeRobotDataset.create(
        repo_id="local/libero",
        root=output_dir,
        robot_type="panda",
        fps=fps,
        features=features,
        use_videos=True,
        image_writer_threads=4,
    )
    stride = source_fps // fps
    try:
        for path in files:
            with h5py.File(path, "r") as raw:
                data = raw["data"]
                problem = data.attrs.get("problem_info", "{}")
                if isinstance(problem, bytes):
                    problem = problem.decode()
                task = json.loads(problem).get("language_instruction", path.stem.replace("_demo", "").replace("_", " "))
                for key in sorted(data):
                    obs = data[key]["obs"]
                    if "ee_states" in obs:
                        pose = np.asarray(obs["ee_states"])
                    else:
                        pose = np.concatenate([obs["ee_pos"], obs["ee_ori"]], axis=-1)
                    fingers = np.asarray(obs["gripper_states"])
                    raw_state = np.concatenate([pose, fingers], axis=-1)
                    state = np.concatenate(
                        [
                            raw_state[:, :3],
                            ee_pose_utils.rotation_vector_to_quaternion(raw_state[:, 3:6]),
                            ee_pose_utils.collapse_opposing_gripper_fingers(raw_state[:, 6:8])[:, None],
                        ],
                        axis=-1,
                    ).astype(np.float32)
                    sampled_indices = list(range(0, len(state), stride))
                    targets, target_validity = build_absolute_next_targets(state, stride)
                    for sample_position, index in enumerate(sampled_indices):
                        dataset.add_frame(
                            {
                                "observation.images.image": np.ascontiguousarray(
                                    obs["agentview_rgb"][index][::-1, ::-1]
                                ),
                                "observation.images.wrist_image": np.ascontiguousarray(
                                    obs["eye_in_hand_rgb"][index][::-1, ::-1]
                                ),
                                "observation.state": state[index],
                                "action": targets[sample_position],
                                "transition_action_valid": np.asarray(
                                    [target_validity[sample_position]], dtype=np.bool_
                                ),
                                "task": task,
                            }
                        )
                    dataset.save_episode()
        dataset.finalize()
        info_path = output_dir / "meta" / "info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        info.update(
            {
                "eef_schema": "eef_absolute_next_observation_wxyz_v2",
                "xq_action_alignment": "next_observation_absolute_pose",
                "quaternion_order": "wxyz",
                "state_units": {"translation": "metre", "gripper": "metre"},
                "action_units": {"translation": "metre", "gripper": "metre"},
                "action_target_lead_s": 1.0 / fps,
                "invalid_transition_semantics": "masked_no_target",
            }
        )
        info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    finally:
        dataset.stop_image_writer()
    print(f"Converted LIBERO to {output_dir}")


if __name__ == "__main__":
    tyro.cli(main)
