#!/usr/bin/env bash
# Run the optional LIBERO-Plus client against an existing policy server.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
exec uv run --project examples/libero_plus --frozen python examples/libero_plus/main.py "$@"
