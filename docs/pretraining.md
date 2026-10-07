# Pretraining

Run all commands from the repository root after [installation](../README.md#installation). The released recipe supports latent world-model pretraining (Stage I), joint world-model/action pretraining (Stage II), and LIBERO fine-tuning (Stage III). Prepare the data yourself using [the data guide](data.md); the converted mixture is not provided.

## Data layout and sampling

`DATA_ROOT` selects the local converted data root and defaults to `<repository>/data/pretrain`. A source directory may contain one dataset or several child datasets with their own `meta/info.json` files:

```text
data/pretrain/
├── intern_a1/<embodiment>/{meta,data,videos}/
├── agibotworld/<effector>/{meta,data,videos}/
├── robotwin/<embodiment>/{meta,data,videos}/
├── libero/{meta,data,videos}/
└── egodex/{meta,data,videos}/
```

| Source | Stage I weight | Stage II weight |
| --- | --- | --- |
| InternData-A1 | 0.20 | 0.20 |
| AgiBotWorld | 0.30 | 0.30 |
| RoboTwin | 0.15 | 0.15 |
| LIBERO | 0.08 | 0.10 |
| EgoDex | 0.15 | — |

These are relative weights. The sampler normalizes them across active source groups, selects a group by weight, and samples a frame uniformly within that group. Stage I consumes videos without action supervision. Stage II uses action-annotated robot trajectories and excludes EgoDex.

Temporal sampling is expressed in seconds so different source frame rates can share a schedule. The default recipe uses six history frames (including the current frame) and six future frames, spaced by 0.2 seconds, with a 0.1-second action interval. Data loaders resolve these offsets against each dataset's frame rate and clamp queries at episode boundaries. Configure other schedules in [config.py](../src/openpi/training/config.py) and keep the same configuration when loading a checkpoint.

## Normalization

Compute Stage II statistics **after** conversion, using the root training environment:

```bash
export DATA_ROOT=/path/to/converted/pretrain
.venv/bin/python scripts/compute_norm_stats.py --config-name stage2_pretraining
```

The script expands every source and child dataset and applies the training action transforms before computing its statistics. Stage I does not require action normalization. Converter-level statistics describe the written raw features and do not replace training statistics. See [normalization](norm_stats.md).

## Stage I: latent world-model pretraining

The wrapper prepares the π₀.₅ base checkpoint, V-JEPA 2 encoder, and tokenizer, then trains `stage1_world_model_pretraining`:

```bash
EXP_NAME=world_model \
bash scripts/run_stage1_world_model_pretraining.sh --batch-size 256
```

Set `STAGE1_INIT_WEIGHT=/path/to/pytorch/base` to select an existing PyTorch initialization directory containing `model.safetensors`; the wrapper then skips the base-checkpoint download. `BASE_CHECKPOINT` chooses an entry in [the base-checkpoint registry](../src/openpi/training/base_checkpoints.py); the default is `pi05_base`.

## Stage II: joint pretraining

Supply a Stage I checkpoint step directory, or let the wrapper find the latest checkpoint for a named Stage I experiment:

```bash
EXP_NAME=joint STAGE1_EXP_NAME=world_model \
bash scripts/run_stage2_pretraining.sh

# Alternatively, select a specific step.
EXP_NAME=joint STAGE2_INIT_WEIGHT=/path/to/stage1/checkpoint/step \
bash scripts/run_stage2_pretraining.sh
```

The default configuration is `stage2_pretraining`. All source datasets and their computed statistics must be present before training starts.

## Stage III: LIBERO adaptation

```bash
EXP_NAME=libero STAGE2_EXP_NAME=joint \
bash scripts/run_stage3_finetuning_libero.sh
```

Alternatively set `STAGE3_INIT_WEIGHT=/path/to/stage2/checkpoint/step`. The wrapper prepares LIBERO EEF data and its normalization statistics. See [LIBERO evaluation](../examples/libero/README.md) for using the resulting checkpoint.

## Training options and outputs

`NUM_GPUS` defaults to `torch.cuda.device_count()` and respects `CUDA_VISIBLE_DEVICES`. It must not exceed the visible device count and must divide the total batch size, including a `--batch-size` override. Adjust the batch size for the available GPU memory. Every wrapper forwards additional arguments to `scripts/train_pytorch.py`; inspect that script's `--help` for options such as `--num-train-steps`, `--resume`, and `--no-wandb-enabled`.

`CONFIG` and `EXP_NAME` select a recipe and experiment name. `CHECKPOINT_DIR` changes the checkpoint root (default `checkpoints/`). With the example names above, step directories are saved under:

```text
checkpoints/stage1_world_model_pretraining/world_model/<step>/
checkpoints/stage2_pretraining/joint/<step>/
checkpoints/stage3_finetuning_libero/libero/<step>/
```

Use the step directory, including its saved `assets/`, for checkpoint handoff and serving. The training script also saves optimizer and training state for resuming an interrupted run. Weights & Biases logging is enabled by default with project `plaw-vla`.

Resume an experiment directly with the same recipe, experiment name, and original training overrides:

```bash
.venv/bin/python scripts/train_pytorch.py stage2_pretraining --exp-name joint --resume
```

Keep the original data sources, sampling weights, temporal schedule, batch size, worker count, number of processes, model, optimizer, and learning-rate schedule. The target `--num-train-steps` and logging/checkpoint intervals may change. Resumable checkpoints include `training_state.pt`, which restores each process's random state, temporal sampling, and next data batch; normalization comes from that step's original `assets/`. Standard Python, NumPy, and PyTorch randomness in workers is replayed. Custom transforms with external state must restore that state themselves.

Checkpoints without `training_state.pt` cannot resume training; use their weights to initialize a new experiment. Inference does not require this file, but still needs the checkpoint's matching model configuration and normalization assets.
