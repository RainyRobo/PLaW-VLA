# Copyright 2026 PLaW-VLA authors. SPDX-License-Identifier: Apache-2.0
"""Convert official LIBERO HDF5 demonstrations to canonical LeRobot EEF data."""

from pathlib import Path
import json
import sys

import numpy as np
import tyro


def main(data_dir: Path, output_dir: Path, *, source_fps: int = 20, fps: int = 10) -> None:
    import h5py
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from openpi.policies import ee_pose_utils

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
                    for index in range(0, len(state), stride):
                        target = min(index + stride, len(state) - 1)
                        dataset.add_frame(
                            {
                                "observation.images.image": np.ascontiguousarray(
                                    obs["agentview_rgb"][index][::-1, ::-1]
                                ),
                                "observation.images.wrist_image": np.ascontiguousarray(
                                    obs["eye_in_hand_rgb"][index][::-1, ::-1]
                                ),
                                "observation.state": state[index],
                                "action": state[target],
                                "task": task,
                            }
                        )
                    dataset.save_episode()
        dataset.finalize()
    finally:
        dataset.stop_image_writer()
    print(f"Converted LIBERO to {output_dir}")


if __name__ == "__main__":
    tyro.cli(main)
