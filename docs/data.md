# Preparing local datasets

The pretraining mixture is not distributed. Obtain each dataset from its original provider, follow its access and use terms, and convert local files with the tools below. The [licensing guide](licenses.md) distinguishes dataset terms from the code license. Converters default to local output and do not publish converted data automatically.

All commands assume the repository root. Install FFmpeg and enough storage for both source data and converted shards. Each converter has a separate Python 3.12 environment pinned to the same LeRobot revision as training; its `uv.lock` is committed alongside `pyproject.toml`.

## Obtain source data

| Dataset | Provider | Local input |
| --- | --- | --- |
| InternData-A1 | [InternRobotics](https://huggingface.co/datasets/InternRobotics/InternData-A1) | Simulation LeRobot tar shards |
| AgiBotWorld | [Alpha](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha), [Beta](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta) | Raw task videos and HDF5 proprioception |
| RoboTwin | [Official project](https://github.com/RoboTwin-Platform/RoboTwin) | EEF LeRobot v2.1 task datasets collected/exported with RoboTwin |
| LIBERO | [Official LIBERO data](https://github.com/Lifelong-Robot-Learning/LIBERO#datasets) | LIBERO demonstration HDF5 files |
| EgoDex | [Apple](https://github.com/apple-aiml-research/ml-egodex) | Paired MP4 and HDF5 episodes |

InternData-A1 and AgiBotWorld require accepting provider terms and signing in before access. Download the relevant simulation shards using the provider's instructions; extraction tools do not bypass access controls. Optional download helpers are:

```bash
bash examples/agibotworld/download_agibotworld_data.sh --output-dir data/raw/agibotworld
bash examples/egodex/download_egodex.sh
```

Select `data/raw/egodex` as the EgoDex helper's output directory. Keep train/test splits distinct when selecting data for pretraining. The RoboTwin EEF converter upgrades existing local LeRobot datasets; collecting the source trajectories is a separate upstream workflow.

## InternData-A1

For v2.1 simulation archives:

```bash
uv sync --project examples/intern_a1 --python 3.12 --frozen
uv run --project examples/intern_a1 --frozen python examples/intern_a1/extract_interndata_a1.py \
  --source-root data/raw/intern_a1/sim_updated --output-dir data/raw/intern_a1/extracted
uv run --project examples/intern_a1 --frozen python examples/intern_a1/convert_interndata_a1_to_lerobot.py \
  --input-root data/raw/intern_a1/extracted --output-dir data/pretrain/intern_a1
```

For the provider's v3.0 archives, use `extract_interndata_a1_v30.py --source-root ... --output-dir ...`, followed by `merge_interndata_a1_v30.py --input-root ... --output-dir data/pretrain/intern_a1`. Outputs are grouped by embodiment. EEF poses use scalar-first quaternions; joint-space source layouts retain their declared semantics. Camera masks identify missing views.

## AgiBotWorld

```bash
uv sync --project examples/agibotworld --python 3.12 --frozen
uv run --project examples/agibotworld --frozen python examples/agibotworld/convert_agibotworld_to_lerobot.py \
  --src-path data/raw/agibotworld --output-dir data/pretrain/agibotworld
```

The converter discovers gripper and dexterous-hand tasks, groups outputs by effector, and writes shared camera names with each effector's state/action layout recorded in metadata. `--task-ids`, `--episodes-per-task`, and `--max-tasks` select local subsets. The raw format is recorded at its source frame rate; the training loader applies time-based sampling.

## RoboTwin EEF datasets and format upgrades

```bash
uv sync --project scripts/data --python 3.12 --frozen
uv run --project scripts/data --frozen python scripts/data/convert_robotwin_to_lerobot.py \
  --input-roots data/raw/robotwin/clean data/raw/robotwin/aug \
  --output-dir data/pretrain/robotwin
```

Each input root contains task-level v2.1 EEF datasets. The outputs are grouped by embodiment, with bimanual 16D layouts ordered as `left [xyz, qw, qx, qy, qz, gripper]`, then `right [xyz, qw, qx, qy, qz, gripper]`. Confirm the source export has this quaternion and gripper convention before conversion; format upgrades do not change action semantics automatically.

General LeRobot format upgrades live in [`scripts/data`](../scripts/data). Run either command in the same environment:

```bash
uv run --project scripts/data --frozen python scripts/data/convert_lerobot_v21_to_v30.py \
  --input-root data/raw/lerobot_v21 --output-dir data/converted/lerobot_v3
uv run --project scripts/data --frozen python scripts/data/convert_lerobot_v20_to_v30.py \
  --input-root data/raw/lerobot_v20 --output-dir data/converted/lerobot_v3_from_v20
```

Use `--help` for explicit column/vector remapping. Choose a new output directory unless you intend to replace an existing conversion.

## LIBERO

Install the shared converter environment, which includes HDF5 support:

```bash
uv sync --project scripts/data --python 3.12 --frozen
uv run --project scripts/data --frozen python examples/libero/convert_libero_to_lerobot.py \
  --data-dir data/raw/libero --output-dir data/pretrain/libero
```

The default converts 20 Hz demonstrations to 10 Hz. It writes 8D absolute EEF states and next-sample EEF action targets, with scalar-first quaternions and physical gripper widths. Images are rotated to match the LIBERO inference client. Actions are converted to deltas from the current state by the Stage II training transforms. Set `--source-fps` and `--fps` only when they match your source data.

[`convert_libero_data_to_lerobot.py`](../examples/libero/convert_libero_data_to_lerobot.py) separately demonstrates raw RLDS conversion. It requires TensorFlow/TensorFlow Datasets and writes signed 7D controller commands; that output needs a matching custom data config and is not interchangeable with the LIBERO EEF output. The provided Stage III recipe instead downloads [the prepared LIBERO EEF dataset](https://huggingface.co/datasets/RainyBot/libero_v3_eef).

## EgoDex

```bash
uv sync --project examples/egodex --python 3.12 --frozen
uv run --project examples/egodex --frozen python examples/egodex/convert_egodex_to_lerobot.py \
  --data-dir data/raw/egodex --output-dir data/pretrain/egodex --subdirs part1
```

List the downloaded training splits with `--subdirs part1 part2 ...` and select tasks with `--task-names`. Without `--subdirs`, all available splits are scanned, including test data if present. EgoDex is action-free input for Stage I. Its data terms restrict commercial use and sharing adaptations; keep converted output local unless you have the required rights.

## Output and training statistics

A LeRobot v3 dataset contains `meta/info.json`, task/episode metadata, and `data/` Parquet shards. Camera data is written to `videos/` shards; the v2.0 format upgrade retains source images inline. Grouped source roots contain one dataset per embodiment/effector. State, action, and image feature names are recorded in `meta/info.json`; the source adapters validate layouts before mixing them.

Conversion and merge commands support selected subsets and, where listed in `--help`, `--resume` or explicit `--overwrite`. Preserve source frame rates and episode boundaries. Point `DATA_ROOT` at the parent of the five converted source directories, then compute [training normalization statistics](norm_stats.md) before [Stage II](pretraining.md).
