#!/usr/bin/env bash
# Run one task against an existing RoboTwin EEF policy server.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: bash examples/robotwin/launch_client.sh [save_root] [task_name]"
  echo "Environment: POLICY_SERVER_HOST, POLICY_PORT, TASK_CONFIG, TEST_NUM, SEED, ROBOTWIN_ASSET_ID."
  exit 0
fi
[[ $# -le 2 ]] || { echo "Expected at most save_root and task_name; use --help for usage." >&2; exit 1; }
SAVE_ROOT="${1:-${ROOT}/results/robotwin/$(date +%Y%m%d_%H%M%S)}"
TASK_NAME="${2:-adjust_bottle}"
export POLICY_SERVER_HOST="${POLICY_SERVER_HOST:-localhost}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${HOME}/.cache/robot_policy/robotwin/torch-extensions}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${HOME}/.cache/robot_policy/robotwin/matplotlib}"
mkdir -p "${SAVE_ROOT}" "${TORCH_EXTENSIONS_DIR}" "${MPLCONFIGDIR}"
SAVE_ROOT="$(cd "${SAVE_ROOT}" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/examples/robotwin/.venv/bin/python}"
[[ -x "${PYTHON_BIN}" ]] || { echo "Run bash examples/robotwin/_install.sh first." >&2; exit 1; }
cd "${ROOT}"
exec "${PYTHON_BIN}" examples/robotwin/main.py \
  --config "${ROOT}/examples/robotwin/deploy_policy.yml" --port "${POLICY_PORT:-8001}" \
  --save_root "${SAVE_ROOT}" --test_num "${TEST_NUM:-1}" --overrides \
  --task_name "${TASK_NAME}" --task_config "${TASK_CONFIG:-demo_clean}" \
  --ckpt_setting "${MODEL_NAME:-checkpoint}" \
  --seed "${SEED:-42}" --policy_name "${POLICY_NAME:-robotwin_policy}"
