# LIBERO

The LIBERO simulator runs in a dedicated Python 3.8 environment and sends observations to the PLaW-VLA policy server over WebSocket. Run commands from the repository root.

## Setup

Complete the [root installation](../../README.md#installation), then initialize only LIBERO and install its client:

```bash
git submodule update --init third_party/libero
uv sync --project examples/libero --python 3.8 --frozen
```

The client creates its LIBERO path configuration automatically in `~/.cache/robot_policy/libero`. Set `LIBERO_CONFIG_PATH` to use another location. The policy server runs in the root environment.

## Evaluation

A checkpoint trained for LIBERO must include its `model.safetensors`, original `assets/` statistics, and matching training configuration. PLaW-VLA weights are pending release; use the output of [Stage III](../../README.md#libero-fine-tuning).

Start the server:

```bash
.venv/bin/python scripts/serve_policy.py --env LIBERO policy:checkpoint \
  --policy.config=stage3_finetuning_libero --policy.dir=/path/to/checkpoint/step
```

Then start a client in another terminal:

```bash
MUJOCO_GL=egl uv run --project examples/libero --frozen python examples/libero/main.py \
  --task-suite-name libero_spatial --seed 42
```

By default, the client evaluates every task in the selected suite with 50 trials per task. Use `--task-ids 0 1 2` to select tasks, `--num-trials-per-task` to set the trial count, and `--host` / `--port` to connect to another server. Supported suites are `libero_spatial`, `libero_object`, `libero_goal`, and `libero_10`. The client reads camera keys and temporal sampling from server metadata. Failed-rollout videos go to `results/libero/videos`; use `--record-video all|failure|none` to control recording and `--video-out-path` to change the output directory.

Wait for the server's listening message before starting evaluation. `--connect-timeout` and `--inference-timeout` adjust the connection and response waits in seconds; defaults are 30 and 60, respectively. `--episode-timeout` sets the total time allowed for simulator initialization and one complete rollout, with a default of 600 seconds.

The server uses eager execution by default. For compiled inference, prefix its command with `TORCH_COMPILE_MODE=max-autotune` and set `--inference-timeout 600 --episode-timeout 1200` on the client; the first inference includes compilation. Batch evaluation forwards these options to the client as extra arguments after the selected mode.

For systems using an X display instead of EGL, set `MUJOCO_GL=glx`.

## Batch evaluation

[`eval.sh`](eval.sh) supports sequential suites, concurrent simulator clients, and checkpoint sweeps. With no arguments it opens an interactive setup; `--help` lists options. For example:

```bash
CONFIG=stage3_finetuning_libero CHECKPOINT_DIR=/path/to/checkpoint/step \
SERVER_GPU=0 CLIENT_GPU=1 TASK_SUITES="libero_spatial libero_goal" \
bash examples/libero/eval.sh serial
```

The parallel example connects to an already running server at `HOST:PORT`. Select the GPUs available to the simulator clients:

```bash
HOST=127.0.0.1 PORT=8001 GPU_LIST="0 1" \
TASK_SUITES="libero_spatial libero_object libero_goal libero_10" \
bash examples/libero/eval.sh parallel
```

Use the direct client command above on a single GPU. Batch outputs and logs are written under `results/libero/`.

## Data

[The data guide](../../docs/data.md) describes the official LIBERO HDF5 converter for pretraining. [`convert_libero_data_to_lerobot.py`](convert_libero_data_to_lerobot.py) is a separate raw RLDS example; it writes signed controller commands and requires a matching custom config. [Normalization](../../docs/norm_stats.md) explains why those actions differ from the canonical EEF dataset used for Stage III.
