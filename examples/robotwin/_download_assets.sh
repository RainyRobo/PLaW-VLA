#!/usr/bin/env bash
# Obtain assets from the pinned official RoboTwin download entrypoint.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${ROOT}/examples/robotwin/.venv/bin/python"
[[ -x "${PYTHON}" ]] || { echo "Install the RoboTwin client first." >&2; exit 1; }
git -C "${ROOT}" submodule update --init third_party/robotwin
cd "${ROOT}/third_party/robotwin/assets"
"${PYTHON}" _download.py
for archive in background_texture embodiments objects; do
    unzip -n "${archive}.zip"
done
cd ..
"${PYTHON}" script/update_embodiment_config_path.py
