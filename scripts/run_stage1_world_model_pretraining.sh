#!/usr/bin/env bash
# Stage I: world-model alignment.
#
# Downloads the registered base checkpoint (default: pi05_base), converts it
# to PyTorch, downloads the default dataset, then trains.
#
#   bash scripts/run_stage1_world_model_pretraining.sh
#   NUM_GPUS=8 BASE_CHECKPOINT=pi05_base bash scripts/run_stage1_world_model_pretraining.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/train_common.sh
source "${ROOT}/scripts/train_common.sh"

CONFIG="${CONFIG:-stage1_world_model_pretraining}"
EXP_NAME="${EXP_NAME:-${CONFIG}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT}/checkpoints}"

export BASE_CHECKPOINT="${BASE_CHECKPOINT:-pi05_base}"
NUM_GPUS="$(default_num_gpus)"
require_divisible_batch "${CONFIG}" "${NUM_GPUS}"
download_stage_assets 1
if [[ -z "${STAGE1_INIT_WEIGHT:-}" ]]; then
  pytorch_dirname="$("$(train_python)" -c 'import os, openpi.training.base_checkpoints as c; print(c.get_base_checkpoint(os.environ["BASE_CHECKPOINT"]).pytorch_dirname)')"
  STAGE1_INIT_WEIGHT="${ROOT}/checkpoints/${pytorch_dirname}"
fi
cd "${ROOT}"
"$(train_torchrun)" \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NUM_GPUS}" \
    scripts/train_pytorch.py "${CONFIG}" \
    --exp_name "${EXP_NAME}" \
    --checkpoint_base_dir "${CHECKPOINT_DIR}" \
    --pytorch_weight_path "${STAGE1_INIT_WEIGHT}"
