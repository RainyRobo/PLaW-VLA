"""Convert raw LIBERO RLDS episodes to a LeRobot v3 dataset.

The written keys match ``LeRobotLiberoDataConfig`` defaults:

- ``observation.images.image``
- ``observation.images.wrist_image``
- ``observation.state`` (8D raw end-effector state)
- ``action`` (7D signed gripper command)

That is not the released ``libero_v3_eef`` layout. Point a new ``TrainConfig`` at
the output directory and set ``dataset_action_gripper_format="signed_command"``.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data-dir /path/to/your/data

Raw LIBERO RLDS lives at https://huggingface.co/datasets/openvla/modified_libero_rlds.
This script needs TensorFlow Datasets:

    uv pip install tensorflow tensorflow_datasets
"""

import pathlib
import shutil

import numpy as np
import tyro

REPO_NAME = "your_hf_username/libero"
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


def convert_libero(data_dir: str, output_dir: pathlib.Path, *, push_to_hub: bool) -> pathlib.Path:
    """Write one LeRobot dataset from the raw LIBERO RLDS splits in ``data_dir``."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import tensorflow_datasets as tfds

    if output_dir.exists():
        shutil.rmtree(output_dir)

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

    dataset.stop_image_writer()
    dataset.finalize()

    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=False,
            license="apache-2.0",
        )
    return output_dir


def main(data_dir: str, output_dir: pathlib.Path | None = None, *, push_to_hub: bool = False) -> None:
    from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME

    destination = output_dir if output_dir is not None else HF_LEROBOT_HOME / REPO_NAME
    written = convert_libero(data_dir, pathlib.Path(destination), push_to_hub=push_to_hub)
    print(f"Wrote LeRobot dataset to {written}")


if __name__ == "__main__":
    tyro.cli(main)
