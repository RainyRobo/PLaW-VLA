# Normalization and action conventions

Training normalizes state and action targets using statistics computed after the recipe's data transforms. Inference must load the same statistics and action convention from the checkpoint's `assets/` directory.

## Computing statistics

After preparing the local mixture, run:

```bash
.venv/bin/python scripts/compute_norm_stats.py --config-name stage2_pretraining
.venv/bin/python scripts/compute_norm_stats.py --config-name stage3_finetuning_libero
```

The first command expands all four Stage II sources and their child datasets. Each child gets statistics under `assets/<config>/<asset_id>/norm_stats.json`. The Stage III wrapper prepares its LIBERO dataset and computes `assets/stage3_finetuning_libero/libero_v3_eef/norm_stats.json` automatically. Stage I uses video supervision and does not require action statistics.

Asset IDs are relative checkpoint paths. A local absolute dataset root is normalized by removing its leading slash; a dataset may instead declare a stable `AssetsConfig(asset_id=...)`. The training checkpoint saves all required statistics under its `assets/` subtree.

Converter statistics describe stored raw features. Recompute training statistics with `compute_norm_stats.py` whenever the action representation, time sampling, or dataset changes. `--max-frames` can limit a local experiment; use the full training data for the final training statistics.

## EEF representations

Canonical EEF poses use position plus a scalar-first quaternion (`qw, qx, qy, qz`). A single arm with one gripper value has 8 dimensions; a dual-arm pose has 16. Adapters retain declared joint-space layouts where a source dataset provides joint targets instead of EEF poses.

For EEF action training, every target in a chunk is expressed relative to the **same current observation state**: position and gripper are subtracted from that state, and orientation uses the rotation from its current quaternion. The output transform reconstructs absolute targets from that same state before benchmark-specific controller conversion.

The Stage III LIBERO EEF dataset stores physical gripper widths. Its adapter canonicalizes them before the delta transform and records the convention in a normalization contract alongside the statistics. Raw LIBERO signed controller commands are a different representation and require an explicitly matching data config. Do not mix widths, open fractions, or signed commands without the corresponding adapter.

## Loading a checkpoint

Use the checkpoint's original normalization statistics, action/gripper conventions, model dimensions, and temporal schedule. A step directory should include `model.safetensors` and its saved `assets/`; optimizer and training-state files are also needed for a training resume. Select the matching configuration with `--policy.config` when serving.
