#!/usr/bin/env bash
# Stage III: task fine-tuning. Continues from the latest Stage II checkpoint.
#
#   bash scripts/run_stage3_finetuning_libero.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/train_common.sh
source "${ROOT}/scripts/train_common.sh"

CONFIG="${CONFIG:-stage3_finetuning_libero}"
EXP_NAME="${EXP_NAME:-${CONFIG}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT}/checkpoints}"
STAGE2_CONFIG="${STAGE2_CONFIG:-stage2_pretraining}"

NUM_GPUS="$(default_num_gpus)"
require_divisible_batch "${CONFIG}" "${NUM_GPUS}"
download_stage_assets 3
if [[ -z "${STAGE3_INIT_WEIGHT:-}" ]]; then
  STAGE3_INIT_WEIGHT="$(latest_checkpoint_dir "${CHECKPOINT_DIR}/${STAGE2_CONFIG}/${STAGE2_CONFIG}")" || {
    echo "No Stage II checkpoint under ${CHECKPOINT_DIR}/${STAGE2_CONFIG}/${STAGE2_CONFIG}." >&2
    echo "Run: bash scripts/run_stage2_pretraining.sh" >&2
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
    --pytorch_weight_path "${STAGE3_INIT_WEIGHT}"
