#!/usr/bin/env bash
# Run one task against an existing RoboTwin EEF policy server.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SAVE_ROOT="${1:-${ROOT}/results/robotwin/$(date +%Y%m%d_%H%M%S)}"
TASK_NAME="${2:-adjust_bottle}"
export POLICY_SERVER_HOST="${POLICY_SERVER_HOST:-localhost}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${HOME}/.cache/plaw-vla/robotwin/torch-extensions}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${HOME}/.cache/plaw-vla/robotwin/matplotlib}"
mkdir -p "${SAVE_ROOT}" "${TORCH_EXTENSIONS_DIR}" "${MPLCONFIGDIR}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/examples/robotwin/.venv/bin/python}"
[[ -x "${PYTHON_BIN}" ]] || { echo "Run bash examples/robotwin/_install.sh first." >&2; exit 1; }
cd "${ROOT}"
exec "${PYTHON_BIN}" examples/robotwin/main.py \
  --config "${ROOT}/examples/robotwin/deploy_policy.yml" --port "${POLICY_PORT:-8001}" \
  --save_root "${SAVE_ROOT}" --test_num "${TEST_NUM:-1}" --overrides \
  --task_name "${TASK_NAME}" --task_config "${TASK_CONFIG:-demo_clean}" \
  --train_config_name "${TRAIN_CONFIG_NAME:-stage3_finetuning_robotwin}" \
  --model_name "${MODEL_NAME:-checkpoint}" --ckpt_setting "${MODEL_NAME:-checkpoint}" \
  --seed "${SEED:-42}" --policy_name "${POLICY_NAME:-plaw_vla}"
