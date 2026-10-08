#!/usr/bin/env bash
set -euo pipefail

ALPHA_REPO="agibot-world/AgiBotWorld-Alpha"
BETA_REPO="agibot-world/AgiBotWorld-Beta"
DEFAULT_OUTPUT_DIR="./data/raw/agibotworld"

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Download AgiBot-World dataset from Hugging Face.

Options:
  --output-dir <path>   Directory to save the dataset (default: ${DEFAULT_OUTPUT_DIR})
  --variant <n>         Dataset variant: 1=Sample, 2=Alpha, 3=Beta
  --task-id <id>        Download only a specific task (e.g. 327). Only for Alpha/Beta.
  -h, --help            Show this help message

Accept the provider terms on Hugging Face before downloading gated data.
Authenticate with HF_TOKEN or run hf auth login in the root environment.
EOF
}

fail() {
    echo "Error: $*" >&2
    exit 1
}

OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}"
VARIANT=""
TASK_ID=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir|--variant|--task-id)
            [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || fail "Missing value for $1"
            case "$1" in
                --output-dir) OUTPUT_DIR="$2" ;;
                --variant) VARIANT="$2" ;;
                --task-id) TASK_ID="$2" ;;
            esac
            shift 2
            ;;
        -h|--help) usage; exit 0 ;;
        *) fail "Unknown option: $1 (see --help)" ;;
    esac
done

if [[ -z "${VARIANT}" ]]; then
    [[ -t 0 ]] || fail "Select --variant 1, 2, or 3 for a non-interactive download"
    echo "Select dataset variant: 1) Sample  2) Alpha  3) Beta"
    read -rp "Enter choice [1/2/3]: " VARIANT
fi
[[ "${VARIANT}" =~ ^[123]$ ]] || fail "Invalid variant: ${VARIANT}; expected 1, 2, or 3"
if [[ -n "${TASK_ID}" ]]; then
    [[ "${TASK_ID}" =~ ^[0-9]+$ ]] || fail "Task id must be a non-negative integer"
    [[ "${VARIANT}" != 1 ]] || fail "--task-id is only available for Alpha or Beta"
fi

# Prefer an activated environment; otherwise use the repository's root environment.
if command -v hf >/dev/null 2>&1; then
    HF_CLI="$(command -v hf)"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    HF_CLI="${REPO_ROOT}/.venv/bin/hf"
    [[ -x "${HF_CLI}" ]] || fail "Install the root environment, then run .venv/bin/hf auth login"
fi

echo "AgiBot-World datasets are gated (CC BY-NC-SA 4.0)."
echo "Accept the provider terms and authenticate with HF_TOKEN or hf auth login."
mkdir -p "${OUTPUT_DIR}"

if [[ "${VARIANT}" == 1 ]]; then
    "${HF_CLI}" download --repo-type dataset "${ALPHA_REPO}" sample_dataset.tar \
        --local-dir "${OUTPUT_DIR}"
    tar -xf "${OUTPUT_DIR}/sample_dataset.tar" -C "${OUTPUT_DIR}"
    rm -f "${OUTPUT_DIR}/sample_dataset.tar"
    echo "Sample dataset saved to: ${OUTPUT_DIR}"
else
    if [[ "${VARIANT}" == 2 ]]; then
        DATASET_REPO="${ALPHA_REPO}"
        DATASET_NAME="AgiBotWorld-Alpha"
    else
        DATASET_REPO="${BETA_REPO}"
        DATASET_NAME="AgiBotWorld-Beta"
    fi
    DATASET_DIR="${OUTPUT_DIR}/${DATASET_NAME}"
    DOWNLOAD_ARGS=(download --repo-type dataset "${DATASET_REPO}" --local-dir "${DATASET_DIR}")
    if [[ -n "${TASK_ID}" ]]; then
        DOWNLOAD_ARGS+=(
            --include "observations/${TASK_ID}/*"
            --include "task_info/task_${TASK_ID}.json"
            --include "proprio_stats/*"
            --include "parameters/*"
        )
    fi
    "${HF_CLI}" "${DOWNLOAD_ARGS[@]}"
    echo "Dataset saved to: ${DATASET_DIR}"
    echo "Extract downloaded tar shards before conversion (from repo root):"
    printf '  uv run --project scripts/data/agibotworld --frozen python scripts/data/agibotworld/extract_agibotworld.py --input-root %q' "${DATASET_DIR}"
    if [[ -n "${TASK_ID}" ]]; then
        printf ' --task-ids %s' "${TASK_ID}"
    fi
    printf '\n'
fi
