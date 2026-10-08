#!/usr/bin/env bash
# Stable direct LIBERO fine-tuning entrypoint.
#
# STAGE3_BASE_SOURCE=auto is release-controlled:
# - current release: resolves to public π₀.₅ foundation initialization;
# - after official PLaW-VLA base weights are published: resolves to that full
#   pretrained checkpoint without changing this command.
# Use STAGE3_BASE_SOURCE=pi05 or plawvla to pin the source explicitly.
#
# Official default recipe:
#   NUM_GPUS=8 EXP_NAME=libero_quickstart \
#     bash scripts/train/run_stage3_libero_quickstart.sh

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
EXP_NAME="${EXP_NAME:-libero_quickstart}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT}/checkpoints}"
STAGE3_BASE_SOURCE="${STAGE3_BASE_SOURCE:-auto}"

case "${STAGE3_BASE_SOURCE}" in
  auto|pi05|plawvla) ;;
  *)
    echo "STAGE3_BASE_SOURCE must be one of: auto, pi05, plawvla." >&2
    exit 1
    ;;
esac

if [[ -n "${BASE_PYTORCH_WEIGHT:-}" ]]; then
  if [[ -n "${STAGE3_BASE_WEIGHT:-}" ]]; then
    echo "Set only STAGE3_BASE_WEIGHT; BASE_PYTORCH_WEIGHT is the deprecated alias." >&2
    exit 1
  fi
  echo "[WARNING] BASE_PYTORCH_WEIGHT is deprecated; use STAGE3_BASE_WEIGHT." >&2
  STAGE3_BASE_WEIGHT="${BASE_PYTORCH_WEIGHT}"
fi
if [[ -n "${STAGE3_BASE_WEIGHT:-}" && "${STAGE3_BASE_SOURCE}" == "auto" ]]; then
  echo "A custom STAGE3_BASE_WEIGHT requires an explicit STAGE3_BASE_SOURCE=pi05 or plawvla." >&2
  echo "This prevents a full PLaW-VLA checkpoint from being loaded as a π₀.₅ foundation, or vice versa." >&2
  exit 1
fi

NUM_GPUS="$(default_num_gpus)"
require_divisible_batch "${CONFIG}" "${NUM_GPUS}" "$@"

initialization_args=()
if ! train_resume_requested "$@"; then
  read -r resolved_source weight_load_mode pytorch_dirname < <(
    "$(train_python)" -c \
      'import sys
from plawvla.training import base_checkpoints as c
requested, custom = sys.argv[1], sys.argv[2] == "1"
s = c.get_stage3_initialization(requested) if custom else c.resolve_stage3_initialization(requested)
print(s.name, s.weight_load_mode, s.pytorch_dirname or "")' \
      "${STAGE3_BASE_SOURCE}" "$([[ -n "${STAGE3_BASE_WEIGHT:-}" ]] && echo 1 || echo 0)"
  )

  if [[ -z "${STAGE3_BASE_WEIGHT:-}" ]]; then
    "$(train_python)" "${ROOT}/scripts/setup/download_assets.py" \
      --stage3-base-source "${STAGE3_BASE_SOURCE}"
    STAGE3_BASE_WEIGHT="${ROOT}/checkpoints/${pytorch_dirname}"
  fi
  require_checkpoint_dir "${STAGE3_BASE_WEIGHT}"

  echo "Stage III initialization"
  echo "  requested source : ${STAGE3_BASE_SOURCE}"
  echo "  resolved source  : ${resolved_source}"
  if [[ "${STAGE3_BASE_SOURCE}" == "auto" && "${resolved_source}" == "pi05" ]]; then
    echo "  reason           : official PLaW-VLA base weights are not released in this repository version"
  fi
  echo "  weight load mode : ${weight_load_mode}"
  echo "  checkpoint       : ${STAGE3_BASE_WEIGHT}"
  echo "  V-JEPA 2         : facebook/vjepa2-vitl-fpc64-256"

  initialization_args+=(
    --pytorch_weight_path "${STAGE3_BASE_WEIGHT}"
    --weight-load-mode "${weight_load_mode}"
    --initialization-source-requested "${STAGE3_BASE_SOURCE}"
    --initialization-source-resolved "${resolved_source}"
  )
fi

# Stage 3 assets include V-JEPA 2, the tokenizer, the prepared LIBERO dataset,
# and matching normalization statistics. Resume reuses checkpoint weights but
# still validates that the runtime assets are available.
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
