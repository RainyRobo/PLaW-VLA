#!/usr/bin/env bash
# Serve a RoboTwin EEF checkpoint with per-task normalization routing.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${POLICY_DIR:?Set POLICY_DIR to a RoboTwin checkpoint step directory containing model.safetensors and assets.}"
cd "${ROOT}"
exec "${ROOT}/.venv/bin/python" scripts/serve/serve_policy_robotwin.py --env ROBOTWIN \
  --port "${POLICY_PORT:-8001}" policy:checkpoint \
  --policy.config="${POLICY_CONFIG:-stage3_finetuning_robotwin}" --policy.dir="${POLICY_DIR}" "$@"
