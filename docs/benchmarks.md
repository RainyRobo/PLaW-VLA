# Optional benchmarks

RoboTwin and LIBERO-Plus have dedicated clients and remain **planned support**. They are not installed by the default LIBERO setup. Use checkpoints whose action space, temporal sampling, and original normalization assets match the selected client.

## RoboTwin

The client targets canonical 16D dual-arm EEF actions (`left xyz, qwqxqyqz, gripper`, then `right`). It requires Linux, the CUDA 12.8 development toolkit, a compatible GPU driver, and `unzip` for the asset archives. Set `CUDA_HOME` to your toolkit directory if `nvcc` is not on `PATH`. PyTorch3D and cuRobo are built from pinned commits in their official repositories; build caches default to `~/.cache/robot_policy/robotwin/`.

```bash
git submodule update --init third_party/robotwin
bash examples/robotwin/_install.sh
bash examples/robotwin/_download_assets.sh
```

The installer applies the pinned upstream's SAPIEN/MPLib compatibility changes inside the client environment. Obtain simulation assets under their accompanying terms. The downloader leaves archives and extracted assets in the external benchmark checkout; these are not packaged with PLaW-VLA.

Use a checkpoint trained for this embodiment, including its original normalization assets:

```bash
POLICY_DIR=/path/to/robotwin/checkpoint/step \
bash examples/robotwin/launch_server.sh

# In another terminal:
SEED=42 TEST_NUM=1 TASK_CONFIG=demo_clean \
bash examples/robotwin/launch_client.sh /path/to/results adjust_bottle
```

`POLICY_SERVER_HOST` and `POLICY_PORT` select the server. The client sends three camera views, temporal history, and the dual-arm state. For checkpoints with per-task statistics, the server selects them using task and task-config metadata. If multiple statistics directories use another naming scheme, set `ROBOTWIN_ASSET_ID` to the matching directory's path relative to the checkpoint's `assets/`. Evaluation starts with the requested seed; `expert_check: true` in `deploy_policy.yml` enables upstream expert-solvability filtering and can advance it.

`ROBOTWIN_VIDEO_MODE=none|failed|all` controls recorded videos. On GPUs unsupported by SAPIEN's bundled OIDN denoiser, set `ROBOTWIN_DENOISER=none` to preserve ray-traced rendering without that denoiser. `eval.sh --help` documents optional multi-task scheduling.

Summarize saved task metrics with:

```bash
python3 examples/robotwin/calc_stat.py /path/to/results
```

## LIBERO-Plus

LIBERO-Plus evaluates a LIBERO-trained checkpoint directly, without additional fine-tuning. Install its separate simulator environment:

```bash
git submodule update --init third_party/libero-plus
uv sync --project examples/libero_plus --python 3.8 --frozen
```

Install the ImageMagick runtime library (`libmagickwand-dev` on Ubuntu) for the upstream texture loader. Follow [the upstream asset instructions](https://github.com/sylvestf/LIBERO-plus) and place the extracted `assets` directory at `third_party/libero-plus/libero/libero/assets`. An external asset directory can be symlinked there; upstream scene loaders require that location. The client path configuration defaults to `~/.cache/robot_policy/libero-plus`, or `LIBERO_PLUS_CONFIG_PATH`.

Use the LIBERO policy server from [the main guide](../README.md#libero-evaluation), then run:

```bash
MUJOCO_GL=egl bash examples/libero_plus/eval.sh \
  --task-suite-name libero_spatial --task-ids 0 1 2 --num-trials-per-task 1 --seed 42
```

Set `--host`, `--port`, and `--result-root` as needed. The client reads image keys, action conventions, and history offsets from the server metadata.

The client saves episode records to `checkpoint.json` under the result root and resumes from that file. Use a separate result root for a different checkpoint or evaluation setup. Summarize recorded perturbation categories with:

```bash
python3 examples/libero_plus/aggregate.py /path/to/results --counts --save
```
