# Contributing

Thank you for improving PLaW-VLA.

## Development setup

Use the locked Python 3.12 environment described in `README.md`:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync --python 3.12 --frozen
bash scripts/setup/install_transformers_patch.sh
```

Dataset converters and benchmark simulators use the separate environments documented in `docs/data.md` and `docs/benchmarks.md`.

## Before submitting a change

- Do not commit downloaded datasets, checkpoints, benchmark assets, credentials, or generated normalization statistics.
- Preserve upstream copyright and license notices when modifying derived code.
- Document changes to dataset layouts, action/state conventions, model architecture, or checkpoint compatibility.
- Run the tests relevant to the change:

```bash
.venv/bin/pytest -q -m "not manual"
```

Tests marked `manual` require local datasets or benchmark installations. The pretraining guide documents real-data validation commands.

Run the repository pre-commit hooks on changed files. For shell changes, also run:

```bash
pre-commit run --files <changed files>
for file in scripts/train/*.sh scripts/setup/*.sh; do
  bash -n "$file"
done
```

## Pull requests

Describe the problem, intended behavior, affected configurations or datasets, validation commands, and any checkpoint compatibility implications. Keep changes focused and avoid combining unrelated generated formatting with behavior changes.

## Data and model contributions

Do not upload third-party data or weights without redistribution rights. New converters and model registries must document the upstream source, expected local layout, state/action conventions, relevant revisions, and applicable terms.
