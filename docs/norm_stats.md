# Normalization statistics

Training normalizes proprioceptive state and action targets with statistics computed on the dataset for that config. Inference loads the same file from the checkpoint's `assets/` directory.

## Where they live

`scripts/download_assets.py` runs `scripts/compute_norm_stats.py` for Stage II and Stage III before those stages start. The files are written to:

```
assets/<config_name>/libero_v3_eef/norm_stats.json
```

Stage I does not load norm stats, because it trains the world model without actions. Stage II and Stage III use different action windows (`action_time_start_s` is `0.1` in Stage II and `0.0` in Stage III), so each config keeps its own file. Do not copy a short smoke-run statistic into the release; recompute it on the full dataset.

## Computing them yourself

```bash
uv run python scripts/compute_norm_stats.py --config-name stage2_pretraining
uv run python scripts/compute_norm_stats.py --config-name stage3_finetuning_libero
```

The asset id comes from the data config (`libero_v3_eef` for the released recipe). A new dataset should set `AssetsConfig(asset_id=...)` to a relative id, then run the same script with that config name. If `asset_id` is omitted, the repo id is used. An absolute local path is stored with the leading slash removed, so `/data/my_robot` lands at `assets/<config>/data/my_robot/` and the same relative path is copied into the checkpoint.

## LIBERO end-effector actions

The released LIBERO configs use an 8D end-effector action. State gripper values are physical widths. Training actions are stored as absolute physical widths, then converted to deltas from the current end-effector state. Every step in the action chunk uses that same state: position and gripper are subtracted from it, and orientation is the quaternion rotation from the current orientation. The contract is recorded next to `norm_stats.json` so a checkpoint cannot be served against a different gripper convention.
