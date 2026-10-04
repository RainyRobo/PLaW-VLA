#!/usr/bin/env bash
# Stage II: joint training. Continues from the latest Stage I checkpoint.
#
#   bash scripts/run_stage2_pretraining.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/train_common.sh
source "${ROOT}/scripts/train_common.sh"

CONFIG="${CONFIG:-stage2_pretraining}"
EXP_NAME="${EXP_NAME:-${CONFIG}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT}/checkpoints}"
STAGE1_CONFIG="${STAGE1_CONFIG:-stage1_world_model_pretraining}"

NUM_GPUS="$(default_num_gpus)"
require_divisible_batch "${CONFIG}" "${NUM_GPUS}"
download_stage_assets 2
if [[ -z "${STAGE2_INIT_WEIGHT:-}" ]]; then
  STAGE2_INIT_WEIGHT="$(latest_checkpoint_dir "${CHECKPOINT_DIR}/${STAGE1_CONFIG}/${STAGE1_CONFIG}")" || {
    echo "No Stage I checkpoint under ${CHECKPOINT_DIR}/${STAGE1_CONFIG}/${STAGE1_CONFIG}." >&2
    echo "Run: bash scripts/run_stage1_world_model_pretraining.sh" >&2
    exit 1
  }
fi
cd "${ROOT}"
"$(train_torchrun)" \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NUM_GPUS}" \
    scripts/train_pytorch.py "${CONFIG}" \
    --exp_name "${EXP_NAME}" \
    --checkpoint_base_dir "${CHECKPOINT_DIR}" \
    --pytorch_weight_path "${STAGE2_INIT_WEIGHT}"
