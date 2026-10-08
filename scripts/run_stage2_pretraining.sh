#!/usr/bin/env bash
# Stage II: joint training. Continues from the latest Stage I checkpoint.
#
#   bash scripts/run_stage2_pretraining.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/train_common.sh
source "${ROOT}/scripts/train_common.sh"

CONFIG="${CONFIG:-stage2_pretraining}"
if train_help_requested "$@"; then
  exec "$(train_python)" "${ROOT}/scripts/train_pytorch.py" "${CONFIG}" "$@"
fi
EXP_NAME="${EXP_NAME:-${CONFIG}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT}/checkpoints}"
STAGE1_CONFIG="${STAGE1_CONFIG:-stage1_world_model_pretraining}"
STAGE1_EXP_NAME="${STAGE1_EXP_NAME:-${STAGE1_CONFIG}}"

NUM_GPUS="$(default_num_gpus)"
require_divisible_batch "${CONFIG}" "${NUM_GPUS}" "$@"
require_pretraining_data "${CONFIG}"
if [[ -z "${STAGE2_INIT_WEIGHT:-}" ]]; then
  STAGE2_INIT_WEIGHT="$(latest_checkpoint_dir "${CHECKPOINT_DIR}/${STAGE1_CONFIG}/${STAGE1_EXP_NAME}")" || {
    echo "No Stage I checkpoint under ${CHECKPOINT_DIR}/${STAGE1_CONFIG}/${STAGE1_EXP_NAME}." >&2
    echo "Run: bash scripts/run_stage1_world_model_pretraining.sh" >&2
    exit 1
  }
fi
require_checkpoint_dir "${STAGE2_INIT_WEIGHT}"
download_stage_assets 2
cd "${ROOT}"
"$(train_torchrun)" \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NUM_GPUS}" \
    scripts/train_pytorch.py "${CONFIG}" \
    --exp_name "${EXP_NAME}" \
    --checkpoint_base_dir "${CHECKPOINT_DIR}" \
    --pytorch_weight_path "${STAGE2_INIT_WEIGHT}" \
    "$@"
