# Optional benchmarks

RoboTwin and LIBERO-Plus have dedicated clients but are not installed by the
default LIBERO setup. Use checkpoints whose action space, temporal sampling,
and original normalization assets match the selected client.

## RoboTwin

The policy wire contract is canonical 16D dual-arm **absolute EEF targets**: `left [xyz, qw, qx, qy, qz, open_fraction]`, then `right`. The RoboTwin-specific adapter validates that representation, normalizes each quaternion, and passes the two absolute poses to upstream `TASK_ENV.take_action(..., action_type="ee")`. It does **not** apply the LIBERO/robosuite OSC position or rotation scales: the pinned RoboTwin implementation already sends an absolute gripper-centre pose to its IK/path planner, and its gripper uses normalized `[0, 1]` values with `1=open`.

This boundary was audited against the pinned RoboTwin submodule commit. A different upstream revision, robot embodiment, dataset export, or controller must confirm all four native semantics before reuse: pose frame, quaternion order, gripper direction/range, and target timing. If any differs, add a dedicated observation encoder and execution adapter; do not change the model contract and do not silently reinterpret values.

RoboTwin timing is waypoint-based. One policy target corresponds to one upstream `take_action` call, whose planner executes a variable number of inner simulator steps. Therefore `deploy_policy.yml` explicitly declares `robotwin_execution_timing: one_target_per_take_action`; removing or changing that declaration fails closed. The published action timestamps are used to validate policy sampling cadence, not as a claim that RoboTwin executes at a fixed 10 Hz. Fixed-rate interpolation requires a new native controller adapter and must not be guessed from LIBERO.

The client requires Linux, the CUDA 12.8 development toolkit, a compatible GPU driver, and `unzip` for the asset archives. Set `CUDA_HOME` to your toolkit directory if `nvcc` is not on `PATH`. PyTorch3D and cuRobo are built from pinned commits in their official repositories; build caches default to `~/.cache/robot_policy/robotwin/`.

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
TASK_CONFIG=demo_clean \
bash examples/robotwin/launch_client.sh /path/to/results adjust_bottle
```

The client executes 5 absolute EEF waypoints before replanning. `deploy_policy.yml` sets `replan_steps: 5`; this counts policy waypoints, not inner physics steps or seconds. Episode count, language, seed, and expert filtering follow upstream RoboTwin: 100 episodes, `instruction_type: unseen`, and `expert_check: true`. Seed 0 starts at episode 100000, from `100000 * (1 + seed)`. Failed scripted seeds are skipped.

`POLICY_SERVER_HOST` and `POLICY_PORT` select the server. The client sends three camera views, temporal history, and canonical dual-arm absolute EEF state. It rejects servers that omit or disagree on `absolute_eef_target`, `wxyz`, `open_fraction`, two arms, exact 16D wire output, or time offsets. For checkpoints with per-task statistics, the server selects them using task and task-config metadata. If multiple statistics directories use another naming scheme, set `ROBOTWIN_ASSET_ID` to the matching directory's path relative to the checkpoint's `assets/`. Evaluation starts at episode `100000 * (1 + seed)`. `expert_check` is on by default and advances past seeds the scripted expert cannot solve.

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
  --task-suite-name libero_spatial --seed 42
```

Control defaults match the LIBERO client: `replan_steps` 5 and seed 42. The server returns `[H, 8]` canonical absolute EEF targets (`xyz`, `wxyz`, open fraction). At each 20 Hz simulator step, the client interpolates the timed target trajectory and converts it to a robosuite OSC command from the live robot state. `replan_steps` counts policy targets rather than simulator steps: the replanning duration is `action_target_time_offsets_s[replan_steps - 1]`. The trial count follows upstream LIBERO-Plus: one trial per task, rather than LIBERO's 50. Use `--task-ids` only to narrow a run.

Set `--host`, `--port`, and `--result-root` as needed. `--connect-timeout` and `--inference-timeout` adjust connection and response waits in seconds. The lightweight client reads image keys, `history_time_offsets_s`, `action_target_time_offsets_s`, and the absolute-action contract from server metadata. Temporal observations are sampled from a timestamped history buffer using simulation time.

Action ensembling is unsupported for LIBERO-Plus absolute EEF targets. Passing `--action-ensembling` fails closed because the previous implementation averaged normalized OSC commands and is not compatible with the new trajectory contract.

Failed rollouts are recorded by default under `results/libero_plus/videos`. Video recording can be configured with:

```bash
MUJOCO_GL=egl bash examples/libero_plus/eval.sh \
  --task-suite-name libero_spatial \
  --record-video all \
  --video-out-path results/libero_plus/videos
```

`--record-video` accepts `none`, `failure`, or `all`. Saved MP4 filenames include the suite, task ID, seed, task name, episode index, and `success` or `failure`. Set `--record-video none` to avoid retaining rollout frames in memory.

The client saves episode records to `checkpoint.json` under the result root and resumes from that file. Resuming requires the same seed. Use a separate result root for a different seed, checkpoint, or evaluation setup. Summarize recorded perturbation categories with:

```bash
python3 examples/libero_plus/aggregate.py /path/to/results --counts --save
```
