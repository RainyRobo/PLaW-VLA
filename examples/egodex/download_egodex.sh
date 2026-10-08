#!/usr/bin/env bash
#
# Interactive downloader for EgoDex dataset archives.
#
# Usage:
#   bash examples/egodex/download_egodex.sh

set -euo pipefail

DEFAULT_OUTPUT_DIR="data/raw/egodex"

declare -A URLS=(
    [part1]="https://ml-site.cdn-apple.com/datasets/egodex/part1.zip"
    [part2]="https://ml-site.cdn-apple.com/datasets/egodex/part2.zip"
    [part3]="https://ml-site.cdn-apple.com/datasets/egodex/part3.zip"
    [part4]="https://ml-site.cdn-apple.com/datasets/egodex/part4.zip"
    [part5]="https://ml-site.cdn-apple.com/datasets/egodex/part5.zip"
    [test]="https://ml-site.cdn-apple.com/datasets/egodex/test.zip"
    [extra]="https://ml-site.cdn-apple.com/datasets/egodex/extra.zip"
)

DOWNLOADS=()
PARALLEL_JOBS=1

print_header() {
    echo "============================================"
    echo "        EgoDex Dataset Downloader"
    echo "============================================"
    echo ""
}

append_download() {
    local key="$1"
    DOWNLOADS+=("$key")
}

dedupe_downloads() {
    local -A seen=()
    local deduped=()
    local item
    for item in "${DOWNLOADS[@]}"; do
        if [[ -z "${seen[$item]:-}" ]]; then
            seen["$item"]=1
            deduped+=("$item")
        fi
    done
    DOWNLOADS=("${deduped[@]}")
}

configure_parallel_jobs() {
    local value
    local default_jobs=3

    if ((${#DOWNLOADS[@]} <= 1)); then
        PARALLEL_JOBS=1
        return
    fi

    read -rp "Parallel files to download at once [${default_jobs}]: " value
    value="${value:-$default_jobs}"

    if [[ ! "$value" =~ ^[0-9]+$ ]] || ((value < 1)); then
        echo "Invalid parallel value '$value'; using ${default_jobs}."
        PARALLEL_JOBS=$default_jobs
        return
    fi
    PARALLEL_JOBS=$value
}

parse_train_part_indices() {
    local raw="$1"
    local cleaned
    local token

    cleaned="$(echo "$raw" | tr ' ' ',')"
    IFS=',' read -r -a tokens <<< "$cleaned"

    for token in "${tokens[@]}"; do
        [[ -z "$token" ]] && continue

        if [[ "$token" =~ ^([1-5])-([1-5])$ ]]; then
            local start="${BASH_REMATCH[1]}"
            local end="${BASH_REMATCH[2]}"
            if (( start > end )); then
                echo "Invalid range '$token' (start > end)."
                return 1
            fi
            for (( i=start; i<=end; i++ )); do
                append_download "part${i}"
            done
        elif [[ "$token" =~ ^[1-5]$ ]]; then
            append_download "part${token}"
        else
            echo "Invalid training set selector: '$token'"
            return 1
        fi
    done

    if ((${#DOWNLOADS[@]} == 0)); then
        echo "No training sets selected."
        return 1
    fi

    return 0
}

choose_training_sets() {
    local choice
    local train_indices

    echo ""
    echo "Training set selection:"
    echo "  1) all training set"
    echo "  2) provide indices of sets chosen"
    echo ""
    read -rp "Choice [1]: " choice
    choice="${choice:-1}"

    case "$choice" in
        1)
            append_download part1
            append_download part2
            append_download part3
            append_download part4
            append_download part5
            ;;
        2)
            echo ""
            echo "Provide part indices (1..5), comma-separated and/or ranges."
            echo "Examples: 2 | 1,3,5 | 2-4 | 1,3-5"
            read -rp "Indices: " train_indices
            parse_train_part_indices "$train_indices"
            ;;
        *)
            echo "Invalid choice: $choice"
            return 1
            ;;
    esac
}

choose_download_scope() {
    local choice

    echo "Select what to download:"
    echo ""
    echo "  1) train set"
    echo "  2) test set"
    echo "  3) additional data"
    echo "  4) all"
    echo ""
    read -rp "Choice [1]: " choice
    choice="${choice:-1}"

    case "$choice" in
        1)
            choose_training_sets
            ;;
        2)
            append_download test
            ;;
        3)
            append_download extra
            ;;
        4)
            append_download part1
            append_download part2
            append_download part3
            append_download part4
            append_download part5
            append_download test
            append_download extra
            ;;
        *)
            echo "Invalid choice: $choice"
            return 1
            ;;
    esac
}

prompt_output_dir() {
    read -rp "Save path [$DEFAULT_OUTPUT_DIR]: " OUTPUT_DIR
    OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
    OUTPUT_DIR="${OUTPUT_DIR/#\~/$HOME}"
    mkdir -p "$OUTPUT_DIR"
}

print_summary() {
    local item

    echo ""
    echo "--- Summary ---"
    echo "Output: $OUTPUT_DIR"
    echo "Downloader: curl"
    echo "Parallel files: $PARALLEL_JOBS"
    echo "Archives:"
    for item in "${DOWNLOADS[@]}"; do
        echo "  - ${item}.zip"
    done
}

download_one_archive() {
    local item="$1"
    local url="${URLS[$item]}"
    local out_file="$OUTPUT_DIR/${item}.zip"

    echo ""
    echo "Downloading $item"
    echo "URL: $url"
    echo "Output: $out_file"

    curl \
        --fail \
        --location \
        --continue-at - \
        --retry 5 \
        --retry-delay 2 \
        --retry-all-errors \
        --connect-timeout 20 \
        --speed-time 30 \
        --speed-limit 1024 \
        "$url" \
        -o "$out_file"
}

download_archives() {
    local item

    if ((PARALLEL_JOBS <= 1)); then
        for item in "${DOWNLOADS[@]}"; do
            download_one_archive "$item"
        done
        return
    fi

    for item in "${DOWNLOADS[@]}"; do
        download_one_archive "$item" &
        while (($(jobs -pr | wc -l) >= PARALLEL_JOBS)); do
            wait -n
        done
    done
    wait
}

extract_archives() {
    local choice
    local item
    local zip_file

    read -rp "Unzip downloaded archives now? [Y/n]: " choice
    choice="${choice:-Y}"
    if [[ ! "$choice" =~ ^[Yy]$ ]]; then
        return
    fi

    if ! command -v unzip >/dev/null 2>&1; then
        echo "unzip not found; skipping extraction."
        return
    fi

    for item in "${DOWNLOADS[@]}"; do
        zip_file="$OUTPUT_DIR/${item}.zip"
        echo "Extracting $zip_file"
        unzip -o "$zip_file" -d "$OUTPUT_DIR"
    done
}

main() {
    print_header
    prompt_output_dir
    choose_download_scope
    dedupe_downloads
    configure_parallel_jobs
    print_summary

    echo ""
    read -rp "Proceed? [Y/n]: " confirm
    confirm="${confirm:-Y}"
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi

    download_archives
    extract_archives

    echo ""
    echo "Done. Files are in: $OUTPUT_DIR"
    local training_splits=()
    local item
    for item in "${DOWNLOADS[@]}"; do
        if [[ "$item" =~ ^part[1-5]$ ]]; then
            training_splits+=("$item")
        fi
    done
    if ((${#training_splits[@]})); then
        echo "Next step (from repo root, for downloaded training splits):"
        printf '  uv run --project examples/egodex --frozen python examples/egodex/convert_egodex_to_lerobot.py --data-dir %q --output-dir data/pretrain/egodex --subdirs' "$OUTPUT_DIR"
        printf ' %s' "${training_splits[@]}"
        printf '\n'
    else
        echo "Select training splits explicitly with --subdirs when preparing pretraining data."
    fi
}

main
