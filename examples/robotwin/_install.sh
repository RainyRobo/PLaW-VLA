#!/usr/bin/env bash
# Install the optional RoboTwin client in its own environment.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."
if [[ -n "${CUDA_HOME:-}" ]]; then export PATH="${CUDA_HOME}/bin:${PATH}"; fi
command -v nvcc >/dev/null || { echo "RoboTwin native extensions require a CUDA 12.8 toolkit (nvcc). Set CUDA_HOME if needed." >&2; exit 1; }
export UV_LINK_MODE=copy
export MAX_JOBS="${MAX_JOBS:-4}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${HOME}/.cache/plaw-vla/robotwin/torch-extensions}"
mkdir -p "${TORCH_EXTENSIONS_DIR}"
# Install Torch before building extensions whose setup imports it.
uv sync --project "${SCRIPT_DIR}" --python 3.10 --frozen --group client --group curobo \
  --no-install-package pytorch3d --no-install-package nvidia-curobo --inexact
PYTHON="${SCRIPT_DIR}/.venv/bin/python"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$("${PYTHON}" -c 'import torch; print(";".join(sorted({f"{m}.{n}" for m,n in (torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count()))})))')}"
[[ -n "${TORCH_CUDA_ARCH_LIST}" ]] || { echo "No visible CUDA GPU. Set TORCH_CUDA_ARCH_LIST for a build without a GPU." >&2; exit 1; }
uv sync --project "${SCRIPT_DIR}" --python 3.10 --frozen --group client --group curobo \
  --no-build-isolation-package pytorch3d --no-build-isolation-package nvidia-curobo
"${PYTHON}" - <<'PYCODE'
from pathlib import Path
import os
import tempfile
import importlib.util
# Compatibility changes recommended by the pinned upstream RoboTwin installer.
patches = {
    "sapien": ("wrapper/urdf_loader.py", (( 'open(urdf_file, "r")', 'open(urdf_file, "r", encoding="utf-8")'),
        ('urdf_file[:-4] + "srdf"', 'urdf_file[:-4] + ".srdf"'),
        ('open(srdf_file, "r")', 'open(srdf_file, "r", encoding="utf-8")'))),
    "mplib": ("planner.py", (('if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:',
        'if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:'),)),
}
for package, (relative, substitutions) in patches.items():
    spec = importlib.util.find_spec(package)
    path = Path(spec.origin).parent / relative
    original = path.read_text()
    modified = original
    for old, new in substitutions:
        if old not in modified and new not in modified:
            raise SystemExit(f"Unexpected {package} source at {path}; refusing to patch.")
        modified = modified.replace(old, new)
    if modified != original:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as out:
            out.write(modified); temporary = out.name
        os.replace(temporary, path)
print("RoboTwin client environment ready.")
PYCODE
