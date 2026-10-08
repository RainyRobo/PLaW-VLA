# Pretraining

Run all commands from the repository root after [installation](../README.md#installation). The released recipe supports latent world-model pretraining (Stage I), joint world-model/action pretraining (Stage II), and LIBERO fine-tuning (Stage III). Prepare the data yourself using [the data guide](data.md); the converted mixture is not provided.

The default global batches are 512 for Stage I and 256 for Stages II and III; the default training lengths are 100,000, 100,000, and 50,000 steps. The examples below select one GPU and a global batch of one for initial setup checks with the complete model. Choose the batch size and training schedule for your full run using [the training options](#training-options-and-outputs).

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
export NUM_GPUS=1
export BATCH_SIZE=1
.venv/bin/python scripts/compute_norm_stats.py --config-name stage2_pretraining
```

The script expands every source and child dataset and applies the training action transforms before computing its statistics. Stage I does not require action normalization. Converter-level statistics describe the written raw features and do not replace training statistics. See [normalization](norm_stats.md).

## Stage I: latent world-model pretraining

The wrapper prepares the π₀.₅ base checkpoint, V-JEPA 2 encoder, and tokenizer, then trains `stage1_world_model_pretraining`:

```bash
EXP_NAME=world_model \
bash scripts/run_stage1_world_model_pretraining.sh --batch-size "$BATCH_SIZE"
```

Set `STAGE1_INIT_WEIGHT=/path/to/pytorch/base` to select an existing PyTorch initialization directory containing `model.safetensors`; the wrapper then skips the base-checkpoint download. `BASE_CHECKPOINT` chooses an entry in [the base-checkpoint registry](../src/openpi/training/base_checkpoints.py); the default is `pi05_base`.

The default conversion writes the π₀.₅ initialization to `checkpoints/pi05_base_pytorch/`. JAX checkpoint and tokenizer downloads use `~/.cache/robot_policy` or `DATA_HOME`; V-JEPA 2 uses the Hugging Face cache or `HF_HOME`. Keep enough storage for the downloaded checkpoint, converted weights, and training outputs.

## Stage II: joint pretraining

Supply a Stage I checkpoint step directory, or let the wrapper find the latest checkpoint for a named Stage I experiment:

```bash
EXP_NAME=joint STAGE1_EXP_NAME=world_model \
bash scripts/run_stage2_pretraining.sh --batch-size "$BATCH_SIZE"

# Alternatively, select a specific step.
EXP_NAME=joint STAGE2_INIT_WEIGHT=/path/to/stage1/checkpoint/step \
bash scripts/run_stage2_pretraining.sh --batch-size "$BATCH_SIZE"
```

The default configuration is `stage2_pretraining`. All source datasets and their computed statistics must be present before training starts.

## Stage III: LIBERO adaptation

```bash
EXP_NAME=libero STAGE2_EXP_NAME=joint \
bash scripts/run_stage3_finetuning_libero.sh --batch-size "$BATCH_SIZE"
```

Alternatively set `STAGE3_INIT_WEIGHT=/path/to/stage2/checkpoint/step`. The wrapper prepares LIBERO EEF data and its normalization statistics. See [LIBERO evaluation](../examples/libero/README.md) for using the resulting checkpoint.

## Training options and outputs

`NUM_GPUS` defaults to `torch.cuda.device_count()` and respects `CUDA_VISIBLE_DEVICES`. It must not exceed the visible device count and must divide the total batch size, including a `--batch-size` override. Adjust the batch size for the available GPU memory. Every wrapper forwards additional arguments to `scripts/train_pytorch.py`. Pass `--help` to any wrapper to view its training options without downloading resources or starting training, including `--num-train-steps`, `--resume`, and `--no-wandb-enabled`.

PyTorch DDP splits the global batch across processes; each GPU holds a complete model and optimizer state for its trainable parameters. Gradient checkpointing is enabled by default to reduce activation memory. `--num-workers` controls data-loader workers per process.

For a short startup and checkpoint-handoff check, add `--num-train-steps 1 --save-interval 1 --no-wandb-enabled` to each wrapper above. This runs the complete model for one optimization step per stage. Use the intended training length, batch size, and learning-rate schedule to obtain a trained policy.

`CONFIG` and `EXP_NAME` select a recipe and experiment name. `CHECKPOINT_DIR` changes the checkpoint root (default `checkpoints/`). With the example names above, step directories are saved under:

```text
checkpoints/stage1_world_model_pretraining/world_model/<step>/
checkpoints/stage2_pretraining/joint/<step>/
checkpoints/stage3_finetuning_libero/libero/<step>/
```

Checkpoint handoff loads `model.safetensors` from the selected step; the destination stage uses the statistics for its own training data and action transforms. Keep Stage II and Stage III checkpoints' saved `assets/` with their weights for resuming and inference. The training script also saves optimizer and training state for resuming an interrupted run. Weights & Biases logging is enabled by default with project `plaw-vla`.

Resume with the same process count, recipe, experiment name, and original training overrides. For the Stage II example above:

```bash
EXP_NAME=joint \
bash scripts/run_stage2_pretraining.sh \
  --batch-size "$BATCH_SIZE" --resume
```

The wrapper restores the latest checkpoint for the current recipe and experiment, including its original normalization assets. Earlier-stage initialization checkpoints are not needed for a resume. Keep the original `CHECKPOINT_DIR` if the run used a custom checkpoint root, and retain `--no-wandb-enabled` if logging was disabled.

Keep the original data sources, sampling weights, temporal schedule, batch size, worker count, number of processes, model, optimizer, and learning-rate schedule. The target `--num-train-steps` and logging/checkpoint intervals may change. Resumable checkpoints include `training_state.pt`, which restores each process's random state, temporal sampling, and next data batch; normalization comes from that step's original `assets/`. Standard Python, NumPy, and PyTorch randomness in workers is replayed. Custom transforms with external state must restore that state themselves.

Checkpoints without `training_state.pt` cannot resume training; use their weights to initialize a new experiment. Inference does not require this file, but still needs the checkpoint's matching model configuration and normalization assets.
