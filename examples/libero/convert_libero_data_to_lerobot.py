# Derived from openpi (Copyright 2024 Physical Intelligence, Inc.; Apache-2.0).
# Modified for PLaW-VLA by the PLaW-VLA authors, 2026.
"""Convert raw LIBERO RLDS episodes to a LeRobot v3 dataset.

The output contains:

- ``observation.images.image``
- ``observation.images.wrist_image``
- ``observation.state`` (8D raw end-effector state)
- ``action`` (7D controller command with a signed gripper value)

Use a custom ``TrainConfig`` that matches these raw features and declares
``dataset_state_input_format="two_finger_qpos"`` and
``dataset_action_gripper_format="signed_command"``. Query the first action at
the current observation timestamp. ``converted_libero_data`` provides these
settings. The canonical EEF recipe uses a different layout; see ``docs/data.md``
for the HDF5 converter.

Usage:

    python examples/libero/convert_libero_data_to_lerobot.py --data-dir /path/to/rlds --output-dir /path/to/converted/libero

Raw LIBERO RLDS lives at https://huggingface.co/datasets/openvla/modified_libero_rlds.
Run this example in a conversion environment with LeRobot, TensorFlow, and
TensorFlow Datasets installed. Output remains local.
"""

import pathlib

import numpy as np
import tyro

REPO_NAME = "local/libero"
RAW_DATASET_NAMES = [
    "libero_10_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
]

# LeRobot v3 feature spec. Shapes stay tuples so they match numpy frames.
LEROBOT_FEATURES = {
    "observation.images.image": {
        "dtype": "image",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.wrist_image": {
        "dtype": "image",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (8,),
        "names": ["state"],
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["action"],
    },
}


def _language_instruction(value) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def convert_libero(data_dir: str, output_dir: pathlib.Path) -> pathlib.Path:
    """Write one LeRobot dataset from the raw LIBERO RLDS splits in ``data_dir``."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import tensorflow_datasets as tfds

    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}; choose a new output directory.")

    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        root=output_dir,
        robot_type="panda",
        fps=10,
        features=LEROBOT_FEATURES,
        use_videos=False,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    try:
        for raw_dataset_name in RAW_DATASET_NAMES:
            raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
            for episode in raw_dataset:
                for step in episode["steps"].as_numpy_iterator():
                    dataset.add_frame(
                        {
                            "observation.images.image": step["observation"]["image"],
                            "observation.images.wrist_image": step["observation"]["wrist_image"],
                            "observation.state": np.asarray(step["observation"]["state"], dtype=np.float32),
                            "action": np.asarray(step["action"], dtype=np.float32),
                            "task": _language_instruction(step["language_instruction"]),
                        }
                    )
                dataset.save_episode()
        dataset.finalize()
    finally:
        dataset.stop_image_writer()

    return output_dir


def main(data_dir: str, output_dir: pathlib.Path | None = None) -> None:
    from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME

    destination = output_dir if output_dir is not None else HF_LEROBOT_HOME / REPO_NAME
    written = convert_libero(data_dir, pathlib.Path(destination))
    print(f"Wrote LeRobot dataset to {written}")


if __name__ == "__main__":
    tyro.cli(main)
