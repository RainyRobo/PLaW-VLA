#!/usr/bin/env bash
# Stage III: task fine-tuning. Continues from the latest Stage II checkpoint.
#
#   bash scripts/train/run_stage3_finetuning_libero.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/train/train_common.sh
source "${ROOT}/scripts/train/train_common.sh"
cd "${ROOT}"

CONFIG="${CONFIG:-stage3_finetuning_libero}"
if train_help_requested "$@"; then
  exec "$(train_python)" "${ROOT}/scripts/train/train_pytorch.py" "${CONFIG}" "$@"
fi
reject_internal_weight_args "$@"
EXP_NAME="${EXP_NAME:-${CONFIG}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT}/checkpoints}"
STAGE2_CONFIG="${STAGE2_CONFIG:-stage2_pretraining}"
STAGE2_EXP_NAME="${STAGE2_EXP_NAME:-${STAGE2_CONFIG}}"

NUM_GPUS="$(default_num_gpus)"
require_divisible_batch "${CONFIG}" "${NUM_GPUS}" "$@"
initialization_args=()
if ! train_resume_requested "$@"; then
  if [[ -z "${STAGE3_INIT_WEIGHT:-}" ]]; then
    STAGE3_INIT_WEIGHT="$(latest_checkpoint_dir "${CHECKPOINT_DIR}/${STAGE2_CONFIG}/${STAGE2_EXP_NAME}")" || {
      echo "No Stage II checkpoint under ${CHECKPOINT_DIR}/${STAGE2_CONFIG}/${STAGE2_EXP_NAME}." >&2
      echo "Run: bash scripts/train/run_stage2_pretraining.sh" >&2
      exit 1
    }
  fi
  require_checkpoint_dir "${STAGE3_INIT_WEIGHT}"
  initialization_args+=(
    --pytorch_weight_path "${STAGE3_INIT_WEIGHT}"
    --weight-load-mode full
  )
fi
download_stage_assets 3 "$@"
"$(train_torchrun)" \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NUM_GPUS}" \
    scripts/train/train_pytorch.py "${CONFIG}" \
    --exp_name "${EXP_NAME}" \
    --checkpoint_base_dir "${CHECKPOINT_DIR}" \
    "${initialization_args[@]}" \
    "$@"
