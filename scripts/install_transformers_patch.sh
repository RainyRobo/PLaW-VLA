#!/usr/bin/env bash
# Apply the PLaW Transformers patch to an isolated virtual environment.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
if [[ "${FORCE_SYNC:-1}" == 1 ]]; then
    UV_LINK_MODE=copy uv sync --frozen
fi
PYTHON="${PATCH_PYTHON:-${ROOT}/.venv/bin/python}"
[[ -x "${PYTHON}" ]] || { echo "Install the project virtual environment first." >&2; exit 1; }
"${PYTHON}" - "${ROOT}" <<'PY'
import os
from pathlib import Path
import shutil
import sys
import tempfile
import transformers

if sys.prefix == sys.base_prefix:
    raise SystemExit("The patch requires a virtual environment.")
if transformers.__version__ != "5.0.0":
    raise SystemExit(f"Expected transformers==5.0.0, found {transformers.__version__}")
source = Path(sys.argv[1]) / "src/openpi/models_pytorch/transformers_replace"
target = Path(transformers.__file__).resolve().parent
if not target.is_relative_to(Path(sys.prefix).resolve()):
    raise SystemExit("Transformers must be installed in the selected virtual environment.")
for path in source.rglob("*.py"):
    destination = target / path.relative_to(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.read_bytes() == path.read_bytes():
        continue
    # Replace the inode: never mutate a file hardlinked to a shared uv cache.
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as out:
        temporary = Path(out.name)
        out.write(path.read_bytes())
    try:
        shutil.copystat(path, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
from transformers.models.siglip import check
if not check.check_whether_transformers_replace_is_installed_correctly():
    raise SystemExit("Transformers patch verification failed.")
print(f"PLaW Transformers patch verified in {target}")
PY
