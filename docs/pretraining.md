# Stage I and Stage II: data conversion to training

This guide is the end-to-end path for reproducing the PLaW-VLA pretraining pipeline:

```text
raw provider data
    ↓ repository converters
local LeRobot v3 datasets
    ↓ Stage II normalization statistics
Stage I (π₀.₅ → world-model alignment)
    ↓ full checkpoint handoff
Stage II (joint world-model + action pretraining)
    ↓ full checkpoint handoff
Stage III (LIBERO adaptation)
```

Run all commands from the repository root after [installation](../README.md#installation). PLaW-VLA Stage I/II weights and the converted pretraining mixture are not distributed. The repository does provide the conversion, normalization, training, checkpoint, resume, and handoff code.

If you only need downstream adaptation without preparing the five-source
mixture, use the
[downstream fine-tuning workflow](../README.md#1-downstream-task-fine-tuning).
The repository provides LIBERO as the complete reference example.

## 1. Choose the local layout

The default converted-data root is `data/pretrain`. Set `DATA_ROOT` explicitly so conversion, statistics, and training all refer to the same location:

```bash
export DATA_ROOT="$PWD/data/pretrain"
mkdir -p "$DATA_ROOT" data/raw
```

The final layout must be:

```text
data/pretrain/
├── intern_a1/<embodiment>/{meta,data,videos}/
├── agibotworld/<effector>/{meta,data,videos}/
├── robotwin/<embodiment>/{meta,data,videos}/
├── libero/{meta,data,videos}/
└── egodex/{meta,data,videos}/
```

A source may be one LeRobot dataset or a parent containing several child datasets. Every leaf dataset must contain `meta/info.json`. Do not point `DATA_ROOT` directly at one source directory.

## 2. Obtain and convert every source

Obtain each source from its provider under its own access and license terms. The commands below write only local output and do not upload converted datasets. See the [data guide](data.md) for source URLs, raw layouts, action/gripper conventions, subset options, and format-specific caveats.

### 2.1 InternData-A1

Accept the provider terms and place the simulation archives under `data/raw/intern_a1/`. Use **one** workflow matching the downloaded format.

LeRobot v2.1 archives:

```bash
uv sync --project scripts/data/intern_a1 --python 3.12 --frozen
uv run --project scripts/data/intern_a1 --frozen python \
  scripts/data/intern_a1/extract_interndata_a1.py \
  --source-root data/raw/intern_a1/sim_updated \
  --output-dir data/raw/intern_a1/extracted
uv run --project scripts/data/intern_a1 --frozen python \
  scripts/data/intern_a1/convert_interndata_a1_to_lerobot.py \
  --input-root data/raw/intern_a1/extracted \
  --output-dir "$DATA_ROOT/intern_a1"
```

LeRobot v3.0 archives:

```bash
uv sync --project scripts/data/intern_a1 --python 3.12 --frozen
uv run --project scripts/data/intern_a1 --frozen python \
  scripts/data/intern_a1/extract_interndata_a1_v30.py \
  --source-root data/raw/intern_a1/sim_updated_lerobotv30 \
  --output-dir data/raw/intern_a1/extracted_v30
uv run --project scripts/data/intern_a1 --frozen python \
  scripts/data/intern_a1/merge_interndata_a1_v30.py \
  --input-root data/raw/intern_a1/extracted_v30 \
  --output-dir "$DATA_ROOT/intern_a1"
```

### 2.2 AgiBotWorld

Download Alpha or Beta, extract its tar shards, then convert the extracted tree. This example uses Beta:

```bash
PATH="$PWD/.venv/bin:$PATH" bash \
  scripts/data/agibotworld/download_agibotworld_data.sh \
  --output-dir data/raw/agibotworld

uv sync --project scripts/data/agibotworld --python 3.12 --frozen
uv run --project scripts/data/agibotworld --frozen python \
  scripts/data/agibotworld/extract_agibotworld.py \
  --input-root data/raw/agibotworld/AgiBotWorld-Beta
uv run --project scripts/data/agibotworld --frozen python \
  scripts/data/agibotworld/convert_agibotworld_to_lerobot.py \
  --src-path data/raw/agibotworld/AgiBotWorld-Beta \
  --output-dir "$DATA_ROOT/agibotworld"
```

Use the same task subset for extraction and conversion. If calibrated fully-open gripper widths are available, pass them with `--gripper-max-width-mm`; changing that choice requires reconversion and recomputing normalization statistics.

### 2.3 RoboTwin

First collect or export local RoboTwin EEF datasets using the upstream project. The converter upgrades local task-level LeRobot v2.1 datasets:

```bash
uv sync --project scripts/data --python 3.12 --frozen
uv run --project scripts/data --frozen python \
  scripts/data/robotwin/convert_robotwin_to_lerobot.py \
  --input-roots data/raw/robotwin/clean data/raw/robotwin/aug \
  --output-dir "$DATA_ROOT/robotwin"
```

Confirm that the source uses the EEF quaternion and gripper convention documented in [data.md](data.md#robotwin-eef-datasets-and-format-upgrades). A format upgrade cannot infer different action semantics.

### 2.4 LIBERO for pretraining

This is the locally converted absolute-EEF dataset used in the Stage I/II
mixture; it is separate from the prepared direct Stage III dataset at
`data/libero_v3_eef`.

```bash
uv sync --project scripts/data --python 3.12 --frozen
uv run --project scripts/data --frozen python \
  scripts/data/libero/convert_libero_to_lerobot.py \
  --data-dir data/raw/libero \
  --output-dir "$DATA_ROOT/libero"
```

The default conversion changes 20 Hz demonstrations to 10 Hz and writes scalar-first quaternion poses and physical gripper widths. Keep the default unless the raw source actually uses another frame rate.

### 2.5 EgoDex

Download the desired training parts and list them explicitly. Avoid accidentally including test data:

```bash
bash scripts/data/egodex/download_egodex.sh
uv sync --project scripts/data/egodex --python 3.12 --frozen
uv run --project scripts/data/egodex --frozen python \
  scripts/data/egodex/convert_egodex_to_lerobot.py \
  --data-dir data/raw/egodex \
  --output-dir "$DATA_ROOT/egodex" \
  --subdirs part1
```

EgoDex is action-free and is used only by Stage I.

## 3. Verify the converted layout

List every detected LeRobot leaf dataset:

```bash
find "$DATA_ROOT" -path '*/meta/info.json' -print | sort
```

You should see leaves under all five top-level source directories. The stage wrappers perform another existence check before launching. To instantiate every Stage I source with its configured transforms and load one real sample from every leaf dataset, run:

```bash
.venv/bin/pytest -q -m manual \
  src/plawvla/training/tests/data_loader_manual_test.py \
  --dataset-config-name stage1_world_model_pretraining
```

This is a real-data validation command, not a unit-test substitute. It can decode video and may take time. A missing source, incompatible metadata layout, camera mismatch, or invalid action-space declaration fails before full training.

The default sampling weights are:

| Source | Stage I | Stage II |
| --- | ---: | ---: |
| InternData-A1 | 0.20 | 0.20 |
| AgiBotWorld | 0.30 | 0.30 |
| RoboTwin | 0.15 | 0.15 |
| LIBERO | 0.08 | 0.10 |
| EgoDex | 0.15 | not used |

Weights are relative and are normalized across active source groups. Frames are sampled uniformly within the selected group.

## 4. Compute Stage II normalization statistics

Stage I predicts visual latents and does not need action normalization. Stage II does, so compute its statistics after all action-bearing sources have been converted:

```bash
export DATA_ROOT="$PWD/data/pretrain"
.venv/bin/python scripts/train/compute_norm_stats.py \
  --config-name stage2_pretraining \
  --batch-size 32
```

The command expands child datasets and applies the same canonical action transforms used in training before calculating statistics. Converter-level `meta/stats.json` describes raw stored features and does not replace these training statistics. See [normalization](norm_stats.md).

`--batch-size` controls only the CPU statistics scan and does not change the training recipe. Reduce it if decoding or transformed state/action batches exceed host memory. `--num-workers` can also be adjusted for the available CPU and storage throughput.

Recompute statistics whenever data, subsets, action conventions, or gripper calibration change.

After the statistics are written, validate every Stage II leaf dataset and its normalization contract:

```bash
.venv/bin/pytest -q -m manual \
  src/plawvla/training/tests/data_loader_manual_test.py \
  --dataset-config-name stage2_pretraining
```

## 5. Understand checkpoint initialization

The wrappers use two explicit loading modes:

| Transition | Mode | Required checkpoint |
| --- | --- | --- |
| π₀.₅ → Stage I | `foundation` | PaliGemma/action tensors; world-model tensors must be absent |
| Stage I → Stage II | `full` | Complete PLaW-VLA model checkpoint |
| Stage II → Stage III | `full` | Complete PLaW-VLA model checkpoint |
| π₀.₅ → direct LIBERO fine-tuning, current fallback | `foundation` | Same foundation rule as Stage I |
| Official PLaW-VLA base → direct LIBERO fine-tuning, after release | `full` | Complete pretrained PLaW-VLA checkpoint |

This prevents an incomplete Stage I/II checkpoint from being silently mistaken for π₀.₅ initialization.

## 6. Stage I: world-model alignment

Stage I prepares the π₀.₅ base checkpoint, V-JEPA 2, and the tokenizer. It loads π₀.₅ into PaliGemma and the action expert; the PLaW-VLA world-model modules retain their fresh initialization.

Launch the official default recipe (100,000 steps, global batch 512):

```bash
export DATA_ROOT="$PWD/data/pretrain"
NUM_GPUS=8 EXP_NAME=stage1_world_model \
  bash scripts/train/run_stage1_world_model_pretraining.sh
```

Useful initialization overrides:

```bash
# Reuse an existing converted π₀.₅ directory containing model.safetensors.
STAGE1_INIT_WEIGHT=/path/to/pi05_base_pytorch \
  bash scripts/train/run_stage1_world_model_pretraining.sh

# Select another registered π₀.₅ source.
BASE_CHECKPOINT=pi05_base \
  bash scripts/train/run_stage1_world_model_pretraining.sh
```

## 7. Stage II: joint world-model and action pretraining

Stage II must start from a **complete Stage I step directory**. By default the wrapper selects the latest numeric step under the named Stage I experiment.

Launch the official default recipe (100,000 steps, global batch 256) from the completed Stage I experiment:

```bash
export DATA_ROOT="$PWD/data/pretrain"
NUM_GPUS=8 STAGE1_EXP_NAME=stage1_world_model EXP_NAME=stage2_joint \
  bash scripts/train/run_stage2_pretraining.sh
```

To select a specific Stage I step instead of automatic latest-step discovery:

```bash
STAGE2_INIT_WEIGHT=/path/to/stage1/checkpoint/100000 \
NUM_GPUS=8 EXP_NAME=stage2_joint \
  bash scripts/train/run_stage2_pretraining.sh
```

Outputs are written to:

```text
checkpoints/stage2_pretraining/stage2_joint/<step>/
```

## 8. Stage III: LIBERO adaptation after Stage II

The paper handoff starts Stage III from a complete Stage II checkpoint. The wrapper automatically prepares the Stage III LIBERO EEF dataset and matching normalization statistics:

```bash
NUM_GPUS=8 STAGE2_EXP_NAME=stage2_joint EXP_NAME=libero \
  bash scripts/train/run_stage3_finetuning_libero.sh
```

Or select a specific checkpoint:

```bash
STAGE3_INIT_WEIGHT=/path/to/stage2/checkpoint/100000 \
NUM_GPUS=8 EXP_NAME=libero \
  bash scripts/train/run_stage3_finetuning_libero.sh
```

Stage III defaults to 50,000 steps and global batch 256. See [LIBERO evaluation](../examples/libero/README.md) for serving and evaluating its checkpoint.

## 9. Training options, outputs, and resume

`NUM_GPUS` defaults to the visible CUDA device count and respects `CUDA_VISIBLE_DEVICES`. It must not exceed the visible count and must divide the global `--batch-size`. PyTorch DDP keeps a complete model and optimizer state on each GPU.

Every wrapper forwards extra options to `scripts/train/train_pytorch.py`. Inspect them without downloading data or starting training:

```bash
bash scripts/train/run_stage1_world_model_pretraining.sh --help
bash scripts/train/run_stage2_pretraining.sh --help
bash scripts/train/run_stage3_finetuning_libero.sh --help
```

Common options include `--batch-size`, `--num-train-steps`, `--save-interval`, `--num-workers`, `--resume`, and `--no-wandb-enabled`. Weights & Biases uses project `plaw-vla` by default.

Each saved step contains model weights, copied normalization assets, optimizer state, metadata, and resumable training state. Keep the whole step directory. Resume with the same recipe, experiment name, process count, batch size, and original training overrides:

```bash
export DATA_ROOT="$PWD/data/pretrain"
NUM_GPUS=8 EXP_NAME=stage2_joint \
  bash scripts/train/run_stage2_pretraining.sh --resume
```

If logging was disabled or the original run changed the batch size, repeat those flags on resume. `--resume` restores the latest valid step of the current experiment and does not use the earlier-stage initialization checkpoint.

Use `CHECKPOINT_DIR=/another/root` to change the checkpoint root. Use `STAGE1_EXP_NAME` or `STAGE2_EXP_NAME` only for a new downstream-stage handoff; `EXP_NAME` always names the run being created or resumed.

## 10. Temporal recipe

The default recipe uses six history frames including the current frame and six future frames at 0.2-second spacing, plus actions at a 0.1-second interval. Time-based offsets allow sources with different frame rates to share the schedule; sampling clamps at episode boundaries. Keep the same model and temporal configuration when handing off checkpoints.
