#!/usr/bin/env bash
set -e

ALPHA_REPO="agibot-world/AgiBotWorld-Alpha"
BETA_REPO="agibot-world/AgiBotWorld-Beta"
ALPHA_URL="https://huggingface.co/datasets/${ALPHA_REPO}"
BETA_URL="https://huggingface.co/datasets/${BETA_REPO}"

DEFAULT_OUTPUT_DIR="./data/raw/agibotworld"

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Download AgiBot-World dataset from Hugging Face.

Options:
  --output-dir <path>   Directory to save the dataset (default: ${DEFAULT_OUTPUT_DIR})
  --variant <n>         Dataset variant: 1=Sample (~7GB), 2=Alpha (~8.5T), 3=Beta (~43.8T)
  --task-id <id>        Download only a specific task (e.g. 327). Only for Alpha/Beta.
  -h, --help            Show this help message
EOF
    exit 0
}

OUTPUT_DIR=""
VARIANT=""
TASK_ID=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --variant)    VARIANT="$2";    shift 2 ;;
        --task-id)    TASK_ID="$2";    shift 2 ;;
        -h|--help)    usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# --- Authentication ---
echo "============================================"
echo "  AgiBot-World Dataset Downloader"
echo "============================================"
echo ""
echo "The AgiBot-World datasets are gated (CC BY-NC-SA 4.0)."
echo "You must accept the license on Hugging Face and log in."
echo ""
echo "If you haven't logged in yet, run: hf login"
echo "  Token page: https://huggingface.co/settings/tokens"
echo ""
read -rp "Have you already logged in to Hugging Face CLI? [Y/n] " hf_auth
if [[ "${hf_auth,,}" == "n" ]]; then
    echo "Running hf login..."
    hf login
fi

# --- Output directory ---
if [[ -z "${OUTPUT_DIR}" ]]; then
    read -rp "Save directory [${DEFAULT_OUTPUT_DIR}]: " user_dir
    OUTPUT_DIR="${user_dir:-${DEFAULT_OUTPUT_DIR}}"
fi
mkdir -p "${OUTPUT_DIR}"
echo "Download directory: ${OUTPUT_DIR}"

# --- Variant selection ---
if [[ -z "${VARIANT}" ]]; then
    echo ""
    echo "Select dataset variant:"
    echo "  1) Sample dataset (~7 GB, from Alpha repo)"
    echo "  2) Alpha — full dataset (~8.5T, 92k trajectories)"
    echo "  3) Beta  — full dataset (~43.8T, 1M+ trajectories)"
    echo ""
    read -rp "Enter choice [1/2/3]: " VARIANT
fi

case "${VARIANT}" in
    1)
        echo ""
        echo "Downloading sample dataset..."
        hf download \
            --repo-type dataset \
            "${ALPHA_REPO}" sample_dataset.tar \
            --local-dir "${OUTPUT_DIR}"
        echo "Extracting sample_dataset.tar..."
        tar -xf "${OUTPUT_DIR}/sample_dataset.tar" -C "${OUTPUT_DIR}"
        rm -f "${OUTPUT_DIR}/sample_dataset.tar"
        echo "Sample dataset saved to: ${OUTPUT_DIR}"
        ;;
    2|3)
        if [[ "${VARIANT}" == "2" ]]; then
            REPO_URL="${ALPHA_URL}"
            DATASET_NAME="AgiBotWorld-Alpha"
        else
            REPO_URL="${BETA_URL}"
            DATASET_NAME="AgiBotWorld-Beta"
        fi

        git lfs install

        if [[ -n "${TASK_ID}" ]]; then
            # Sparse checkout for a specific task
            CLONE_DIR="${OUTPUT_DIR}/${DATASET_NAME}"
            echo ""
            echo "Downloading task ${TASK_ID} from ${DATASET_NAME} (sparse checkout)..."
            mkdir -p "${CLONE_DIR}"
            cd "${CLONE_DIR}"

            if [[ ! -d ".git" ]]; then
                git init
                git remote add origin "${REPO_URL}"
            fi

            git sparse-checkout init
            git sparse-checkout set \
                "observations/${TASK_ID}" \
                "task_info/task_${TASK_ID}.json" \
                "scripts" \
                "proprio_stats/${TASK_ID}" \
                "parameters/${TASK_ID}"
            git pull origin main

            echo "Task ${TASK_ID} saved to: ${CLONE_DIR}"
        else
            # Full clone
            echo ""
            echo "Downloading full ${DATASET_NAME} dataset..."
            echo "This may take a very long time for large datasets."
            echo ""
            cd "${OUTPUT_DIR}"
            git clone "${REPO_URL}"
            echo "Dataset saved to: ${OUTPUT_DIR}/${DATASET_NAME}"
        fi
        ;;
    *)
        echo "Invalid choice: ${VARIANT}. Please select 1, 2, or 3."
        exit 1
        ;;
esac

echo ""
echo "Done."
