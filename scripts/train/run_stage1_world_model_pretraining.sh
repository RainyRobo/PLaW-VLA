#!/usr/bin/env bash
# Stage I: world-model alignment.
#
# Downloads the registered base checkpoint (default: pi05_base), converts it
# to PyTorch, then trains on the converted datasets in DATA_ROOT.
#
#   bash scripts/train/run_stage1_world_model_pretraining.sh
#   NUM_GPUS=8 BASE_CHECKPOINT=pi05_base bash scripts/train/run_stage1_world_model_pretraining.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/train/train_common.sh
source "${ROOT}/scripts/train/train_common.sh"
cd "${ROOT}"

CONFIG="${CONFIG:-stage1_world_model_pretraining}"
if train_help_requested "$@"; then
  exec "$(train_python)" "${ROOT}/scripts/train/train_pytorch.py" "${CONFIG}" "$@"
fi
reject_internal_weight_args "$@"
EXP_NAME="${EXP_NAME:-${CONFIG}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT}/checkpoints}"

export BASE_CHECKPOINT="${BASE_CHECKPOINT:-pi05_base}"
NUM_GPUS="$(default_num_gpus)"
require_divisible_batch "${CONFIG}" "${NUM_GPUS}" "$@"
require_pretraining_data "${CONFIG}"
initialization_args=()
if ! train_resume_requested "$@" && [[ -n "${STAGE1_INIT_WEIGHT:-}" ]]; then
  require_checkpoint_dir "${STAGE1_INIT_WEIGHT}"
fi
download_stage_assets 1 "$@"
if ! train_resume_requested "$@"; then
  if [[ -z "${STAGE1_INIT_WEIGHT:-}" ]]; then
    pytorch_dirname="$("$(train_python)" -c 'import os, plawvla.training.base_checkpoints as c; print(c.get_base_checkpoint(os.environ["BASE_CHECKPOINT"]).pytorch_dirname)')"
    STAGE1_INIT_WEIGHT="${ROOT}/checkpoints/${pytorch_dirname}"
  fi
  require_checkpoint_dir "${STAGE1_INIT_WEIGHT}"
  initialization_args+=(
    --pytorch_weight_path "${STAGE1_INIT_WEIGHT}"
    --weight-load-mode foundation
  )
fi
"$(train_torchrun)" \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NUM_GPUS}" \
    scripts/train/train_pytorch.py "${CONFIG}" \
    --exp_name "${EXP_NAME}" \
    --checkpoint_base_dir "${CHECKPOINT_DIR}" \
    "${initialization_args[@]}" \
    "$@"
