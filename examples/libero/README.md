# LIBERO Benchmark

This example runs the LIBERO simulator in a dedicated Python 3.8 `uv` project and sends observations to a PLaW-VLA policy server over WebSocket.

All commands below assume your current working directory is the repo root.

## Before You Start

- Finish the repo-root installation in [`README.md`](../../README.md#installation) if you have not already. The policy server uses the repo-root `uv` environment.
- Initialize submodules:

```bash
git submodule update --init --recursive
```

- The LIBERO client environment is defined in [`pyproject.toml`](./pyproject.toml) and locked in [`uv.lock`](./uv.lock).
- [`examples/libero/main.py`](./main.py) now writes a repo-local LIBERO config automatically before importing LIBERO, so you do not need to manage `PYTHONPATH` or answer LIBERO's first-run path prompt by hand.

## One-Time LIBERO Client Setup

Sync the dedicated `uv` project in `examples/libero`:

```bash
uv sync --project examples/libero --python 3.8 --frozen
```

This creates and syncs `examples/libero/.venv` with Python 3.8. If `uv` cannot find a local Python 3.8 interpreter, it will prompt to download one unless you disabled managed Python downloads.

## Quick Start

1. Start a policy server in one terminal with a custom checkpoint.

```bash
uv run scripts/serve_policy.py --env LIBERO policy:checkpoint \
  --policy.config=stage3_finetuning_libero \
  --policy.dir=<path/to/your/checkpoint>
```

The server listens on port `8001` by default.

2. Start the LIBERO client in a second terminal.

```bash
uv run --project examples/libero --frozen python examples/libero/main.py
```

Useful variations:

```bash
# Use a different task suite.
uv run --project examples/libero --frozen python examples/libero/main.py --task-suite-name libero_10

# Connect to a remote server.
uv run --project examples/libero --frozen python examples/libero/main.py --host <server-host> --port 8001

# Use this if Mujoco/EGL initialization fails on your machine.
MUJOCO_GL=glx uv run --project examples/libero --frozen python examples/libero/main.py
```

Videos are written to `results/libero/videos` by default.

## Batch Evaluation

Use the single batch entrypoint [`eval.sh`](./eval.sh).

Run it with no arguments to launch the interactive wizard:

```bash
bash examples/libero/eval.sh
```

If `whiptail` is installed, the wizard now stays in the dialog UI end-to-end:

- the first page lets you choose `quick`, `custom`, or `defaults`
- `quick` asks only for the required options, `custom` keeps the full flow and adds inline explanations on each page, and `defaults` lets you edit the wizard's prefilled values before starting
- suite selection uses a checklist
- GPU selection uses a checklist, excludes the policy GPU from eval choices, and preselects as many eval GPUs as the number of selected suites when possible
- execution mode uses a menu
- benchmark checkpoint steps use a checklist
- the remaining fields use dialog input boxes and menus instead of dropping back to plain prompts

Modes:

- `serial`: start a local policy server from `CHECKPOINT_DIR`, then run the selected suites sequentially on `CLIENT_GPU`.
- `parallel`: run suites through a GPU worker queue sized by `GPU_LIST`. If `CHECKPOINT_DIR` is set, the script starts a local policy server first. If `CHECKPOINT_DIR` is unset, it connects to an already running server at `HOST:PORT`. If there are more suites than GPU slots, the extra suites wait in the queue automatically.
- `benchmark`: sweep `CKPT_BASE/<step>` for each step in `CKPT_STEPS`. Each checkpoint gets its own local server. Set `BENCHMARK_MODE=serial|parallel` to choose whether suites run sequentially or through the same GPU worker queue for that checkpoint.
- `--debug`: optional quick smoke mode. It sets `TASK_SUITES=libero_spatial`, `TASK_IDS=0`, `TRIALS=1`, and `RECORD=none` unless you override them explicitly.

Common commands:

```bash
# serial: local server + sequential evaluation
CONFIG=stage3_finetuning_libero \
CHECKPOINT_DIR=<path/to/your/checkpoint> \
SERVER_GPU=0 \
CLIENT_GPU=1 \
TASK_SUITES="libero_spatial libero_goal" \
bash examples/libero/eval.sh serial

# parallel: auto-start a local server from CHECKPOINT_DIR
CONFIG=stage3_finetuning_libero \
CHECKPOINT_DIR=<path/to/your/checkpoint> \
SERVER_GPU=0 \
GPU_LIST="1 2 3 4" \
TASK_SUITES="libero_spatial libero_object libero_goal libero_10" \
bash examples/libero/eval.sh parallel

# parallel: connect to an already running server
TASK_SUITES="libero_spatial libero_goal" \
HOST=127.0.0.1 \
PORT=8001 \
GPU_LIST="0 1" \
bash examples/libero/eval.sh parallel

# benchmark: sweep checkpoints and evaluate suites in parallel for each step
CONFIG=stage3_finetuning_libero \
CKPT_BASE=<path/to/your/checkpoint> \
CKPT_STEPS="29000 30000" \
BENCHMARK_MODE=parallel \
SERVER_GPU=0 \
GPU_LIST="1 2 3 4" \
bash examples/libero/eval.sh benchmark
```

Results are written under `results/libero/`. Benchmark runs now also keep per-checkpoint logs and summaries in `results/libero/benchmark_<timestamp>/<step>/`.

## Reference Result

Baseline numbers reported by [openpi](https://github.com/Physical-Intelligence/openpi)
for the π₀.₅ model, provided for context only. These are **not** PLaW-VLA results —
see the paper for PLaW-VLA numbers.

| Model | Libero Spatial | Libero Object | Libero Goal | Libero 10 | Average |
|-------|---------------|---------------|-------------|-----------|---------|
| π₀.₅ @ 30k (upstream baseline) | 98.8 | 98.2 | 98.0 | 92.4 | 96.85 |

## Dataset Conversion

For a minimal raw RLDS LIBERO -> LeRobot conversion example, see
[`convert_libero_data_to_lerobot.py`](./convert_libero_data_to_lerobot.py).

This script is the reference template for preparing your own data: it shows the
column layout (`image`, `wrist_image`, `state`, `actions`, `task`) and the frame /
episode writing loop that the training pipeline expects. Adapt it for other
datasets, then point the `repo_id` in your stage config at the output directory.
