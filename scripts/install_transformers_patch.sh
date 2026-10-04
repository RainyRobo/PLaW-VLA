#!/usr/bin/env bash
#
# Install the local Transformers patch required by PLaW-VLA's PyTorch backend.
#
# This script:
#   1. Verifies the active environment has the expected transformers version.
#   2. Locates the installed `transformers/` package directory.
#   3. Copies every file under
#      `src/openpi/models_pytorch/transformers_replace/` over the installed
#      transformers package, preserving directory structure.
#   4. Runs the project-provided sanity check so you know the patch is live.
#
# Usage:
#   bash scripts/install_transformers_patch.sh
#   FORCE_SYNC=0 bash scripts/install_transformers_patch.sh   # skip `uv sync`
#   EXPECTED_VERSION=5.0.0 bash scripts/install_transformers_patch.sh
#
# WARNING: this rewrites files inside the active venv's site-packages. If your
# uv cache uses hardlinks (the default) the rewrite can leak into the shared
# cache and into other projects pinning the same transformers version. To roll
# back, run `uv cache clean transformers` and `uv sync` again, or rebuild the
# venv from scratch. For full isolation, prefer the Docker workflow described
# in docs/docker.md.

set -euo pipefail

EXPECTED_VERSION="${EXPECTED_VERSION:-5.0.0}"
FORCE_SYNC="${FORCE_SYNC:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PATCH_DIR="${REPO_ROOT}/src/openpi/models_pytorch/transformers_replace"

if [[ ! -d "${PATCH_DIR}" ]]; then
    echo "[ERROR] patch source directory not found: ${PATCH_DIR}" >&2
    exit 1
fi

cd "${REPO_ROOT}"

if [[ "${FORCE_SYNC}" == "1" ]]; then
    echo "[INFO] running 'uv sync' to ensure dependencies are installed..."
    uv sync
else
    echo "[INFO] skipping 'uv sync' (FORCE_SYNC=0)"
fi

installed_version="$(uv run python -c 'import transformers; print(transformers.__version__)')"
if [[ "${installed_version}" != "${EXPECTED_VERSION}" ]]; then
    echo "[ERROR] expected transformers==${EXPECTED_VERSION}, found ${installed_version}" >&2
    echo "        set EXPECTED_VERSION=<your-version> if you intentionally pinned a different one." >&2
    exit 1
fi
echo "[OK] transformers ${installed_version} detected."

transformers_dir="$(uv run python - <<'PY'
import pathlib
import site
for base in map(pathlib.Path, site.getsitepackages()):
    candidate = base / "transformers"
    if candidate.exists():
        print(candidate)
        break
else:
    raise SystemExit("Could not find the installed transformers package.")
PY
)"

if [[ -z "${transformers_dir}" || ! -d "${transformers_dir}" ]]; then
    echo "[ERROR] failed to locate installed transformers directory." >&2
    exit 1
fi
echo "[INFO] installed transformers package: ${transformers_dir}"

backup_dir="${transformers_dir}.pre_plaw_vla_patch.bak"
if [[ ! -d "${backup_dir}" ]]; then
    echo "[INFO] creating one-time backup at: ${backup_dir}"
    cp -a "${transformers_dir}" "${backup_dir}"
else
    echo "[INFO] backup already exists at ${backup_dir} (kept as-is)"
fi

echo "[INFO] applying patch from ${PATCH_DIR}/ -> ${transformers_dir}/"
cp -rv "${PATCH_DIR}"/* "${transformers_dir}"/ >/tmp/plaw_vla_patch_apply.log
echo "[OK] copied $(wc -l </tmp/plaw_vla_patch_apply.log) entries (full log at /tmp/plaw_vla_patch_apply.log)"

uv run python - <<'PY'
from transformers.models.siglip import check
assert check.check_whether_transformers_replace_is_installed_correctly(), (
    "transformers patch verification failed."
)
print("[OK] transformers patch installed successfully")
PY

cat <<'POST'

[NEXT STEPS]
- Re-run your serve/eval/train command (e.g. `uv run scripts/serve_policy.py ...`)
  and the AttributeError on `paligemma.language_model` should be gone.
- To roll back the patch, restore the backup directory printed above, or run:
    uv cache clean transformers
    uv sync
POST
