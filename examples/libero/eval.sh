#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MAIN_SCRIPT="${PROJECT_ROOT}/examples/libero/main.py"
DEFAULT_SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")


usage() {
    cat <<'EOF'
Usage: bash examples/libero/eval.sh [global flags] [mode] [extra args for main.py]

When no mode is provided, the script launches an interactive wizard.

Global flags:
  --debug    Quick smoke-test defaults. Does not override explicit env vars.
             Defaults: TASK_SUITES=libero_spatial TASK_IDS=0 TRIALS=1 RECORD=none RUN_NAME=libero_debug

Modes:
  serial     Start a local policy server from a custom checkpoint, then evaluate suites sequentially.
  suites     Evaluate one or more suites sequentially against an existing server.
  parallel   Evaluate multiple suites with a GPU worker queue. Uses an existing server unless CHECKPOINT_DIR is set.
  benchmark  Sweep multiple checkpoints; each checkpoint can run suites serially or in parallel.

Common environment variables:
  HOST, PORT, TRIALS, RECORD, SEED, TASK_IDS, TASK_SUITES

Serial mode:
  CONFIG, CHECKPOINT_DIR, SERVER_GPU, CLIENT_GPU, RESULTS_DIR

Suites mode:
  CLIENT_GPU, START_FROM, RESULTS_DIR

Parallel mode:
  GPU_LIST, RUN_NAME, RESULTS_DIR
  Optional local-server vars: CONFIG, CHECKPOINT_DIR, SERVER_GPU
  Suites beyond the number of GPUs are queued automatically.

Benchmark mode:
  CONFIG, CKPT_BASE, CKPT_STEPS, SERVER_GPU, RESULTS_DIR
  BENCHMARK_MODE=serial|parallel
  Serial benchmark vars: CLIENT_GPU
  Parallel benchmark vars: GPU_LIST

Examples:
  bash examples/libero/eval.sh
  CHECKPOINT_DIR=/path/to/checkpoint bash examples/libero/eval.sh --debug serial
  CHECKPOINT_DIR=/path/to/checkpoint CONFIG=stage3_finetuning_libero bash examples/libero/eval.sh serial
  TASK_SUITES="libero_spatial libero_goal" CHECKPOINT_DIR=/path/to/checkpoint GPU_LIST="0 1" bash examples/libero/eval.sh parallel
  TASK_SUITES="libero_spatial libero_goal" HOST=127.0.0.1 PORT=8001 GPU_LIST="0 1" bash examples/libero/eval.sh parallel
  CKPT_BASE=/path/to/checkpoints CONFIG=stage3_finetuning_libero CKPT_STEPS="10000 20000 30000" BENCHMARK_MODE=parallel GPU_LIST="0 1 2 3" bash examples/libero/eval.sh benchmark
EOF
}


ensure_prereqs() {
    if [ ! -f "${MAIN_SCRIPT}" ]; then
        echo "[ERROR] Cannot find LIBERO eval entrypoint: ${MAIN_SCRIPT}"
        exit 1
    fi
    if [ ! -d "${PROJECT_ROOT}/third_party/libero/libero" ]; then
        echo "[ERROR] LIBERO submodule is missing from ${PROJECT_ROOT}/third_party/libero"
        echo "Run: git submodule update --init --recursive"
        exit 1
    fi
    if ! command -v uv >/dev/null 2>&1; then
        echo "[ERROR] uv is not installed or not on PATH."
        exit 1
    fi
}


elapsed_str() {
    local secs="$1"
    printf "%02d:%02d:%02d" $((secs / 3600)) $(((secs % 3600) / 60)) $((secs % 60))
}


extract_results_block() {
    local log_file="$1"
    sed -n '/\[RESULTS_TABLE_START\]/,/\[RESULTS_TABLE_END\]/p' "${log_file}" 2>/dev/null
}


extract_success_rate() {
    local log_file="$1"
    local block
    block="$(extract_results_block "${log_file}")"
    if [[ -n "${block}" ]]; then
        echo "${block}" | grep '^suite_total=' | head -1 | cut -d'|' -f2
    else
        awk '/Final Total Success Rate:/ {rate=$5} END {if (rate != "") print rate; else print "N/A"}' "${log_file}"
    fi
}


extract_task_lines() {
    local log_file="$1"
    extract_results_block "${log_file}" | grep '^task=' | sed 's/^task=//'
}


build_task_id_args() {
    TASK_ID_ARGS=()
    if [[ -n "${TASK_IDS:-}" ]]; then
        TASK_ID_ARGS+=(--task-ids)
        for tid in ${TASK_IDS}; do
            TASK_ID_ARGS+=("${tid}")
        done
    fi
}


resolve_suites() {
    local suites_str="${1:-}"
    SUITES=()
    if [[ -n "${suites_str}" ]]; then
        read -r -a SUITES <<< "${suites_str}"
    else
        SUITES=("${DEFAULT_SUITES[@]}")
    fi
    if [[ ${#SUITES[@]} -eq 0 ]]; then
        echo "[ERROR] No task suites were resolved."
        exit 1
    fi
}


apply_debug_defaults() {
    if [[ "${DEBUG_MODE}" != true ]]; then
        return
    fi

    local -a applied=()
    if [[ -z "${TASK_SUITES:-}" ]]; then
        TASK_SUITES="libero_spatial"
        applied+=("TASK_SUITES=${TASK_SUITES}")
    fi
    if [[ -z "${TASK_IDS:-}" ]]; then
        TASK_IDS="0"
        applied+=("TASK_IDS=${TASK_IDS}")
    fi
    if [[ -z "${TRIALS:-}" ]]; then
        TRIALS="1"
        applied+=("TRIALS=${TRIALS}")
    fi
    if [[ -z "${RECORD:-}" ]]; then
        RECORD="none"
        applied+=("RECORD=${RECORD}")
    fi
    if [[ -z "${RUN_NAME:-}" ]]; then
        RUN_NAME="libero_debug"
        applied+=("RUN_NAME=${RUN_NAME}")
    fi

    echo "[INFO] Debug mode enabled for a quick LIBERO smoke test."
    if [[ ${#applied[@]} -gt 0 ]]; then
        echo "  Applied defaults: ${applied[*]}"
    else
        echo "  All debug-tunable values were already provided explicitly."
    fi
}


init_suite_tracking() {
    unset SUITE_RESULTS SUITE_TIMES SUITE_ORDER
    declare -gA SUITE_RESULTS=()
    declare -gA SUITE_TIMES=()
    declare -ga SUITE_ORDER=()
}


validate_checkpoint_dir() {
    local checkpoint_dir="$1"
    if [[ -z "${checkpoint_dir}" ]]; then
        echo "[ERROR] CHECKPOINT_DIR must be set."
        exit 1
    fi
    if [[ ! -d "${checkpoint_dir}" ]]; then
        echo "[ERROR] Checkpoint directory not found: ${checkpoint_dir}"
        exit 1
    fi
}


ensure_port_available() {
    local port="$1"
    if ! python3 - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(1)
try:
    s.connect(("127.0.0.1", port))
except OSError:
    sys.exit(0)
else:
    sys.exit(1)
finally:
    s.close()
PY
    then
        echo "[ERROR] Port ${port} is already in use on 127.0.0.1."
        echo "Use a different PORT or stop the existing listener before starting a local server."
        exit 1
    fi
}


prompt_with_default() {
    local __var_name="$1"
    local prompt_text="$2"
    local default_value="$3"
    local input
    if ui_can_use_whiptail; then
        input="$(whiptail \
            --title "LIBERO Eval" \
            --inputbox "${prompt_text}" \
            12 90 "${default_value}" \
            3>&1 1>&2 2>&3)" || exit 1
        printf -v "${__var_name}" '%s' "${input:-${default_value}}"
        return
    fi
    if [[ "${prompt_text}" == *$'\n'* ]]; then
        printf '%s\n' "${prompt_text}"
        read -r -p "[${default_value}]: " input
    else
        read -r -p "${prompt_text} [${default_value}]: " input
    fi
    printf -v "${__var_name}" '%s' "${input:-${default_value}}"
}


prompt_optional() {
    local __var_name="$1"
    local prompt_text="$2"
    local input
    if ui_can_use_whiptail; then
        input="$(whiptail \
            --title "LIBERO Eval" \
            --inputbox "${prompt_text}" \
            12 90 "" \
            3>&1 1>&2 2>&3)" || exit 1
        printf -v "${__var_name}" '%s' "${input}"
        return
    fi
    if [[ "${prompt_text}" == *$'\n'* ]]; then
        printf '%s\n' "${prompt_text}"
        read -r -p "> " input
    else
        read -r -p "${prompt_text}: " input
    fi
    printf -v "${__var_name}" '%s' "${input}"
}


prompt_required() {
    local __var_name="$1"
    local prompt_text="$2"
    local current_value="${!__var_name:-}"
    local input
    while true; do
        if ui_can_use_whiptail; then
            input="$(whiptail \
                --title "LIBERO Eval" \
                --inputbox "${prompt_text}" \
                12 90 "${current_value}" \
                3>&1 1>&2 2>&3)" || exit 1
        else
            if [[ "${prompt_text}" == *$'\n'* ]]; then
                printf '%s\n' "${prompt_text}"
                if [[ -n "${current_value}" ]]; then
                    read -r -p "[${current_value}]: " input
                else
                    read -r -p "> " input
                fi
            else
                if [[ -n "${current_value}" ]]; then
                    read -r -p "${prompt_text} [${current_value}]: " input
                else
                    read -r -p "${prompt_text}: " input
                fi
            fi
        fi
        input="${input:-${current_value}}"
        if [[ -n "${input}" ]]; then
            printf -v "${__var_name}" '%s' "${input}"
            return
        fi
        if ui_can_use_whiptail; then
            whiptail --title "LIBERO Eval" --msgbox "This value is required." 8 50
        else
            echo "[ERROR] This value is required."
        fi
    done
}


append_note_to_prompt() {
    local prompt_text="$1"
    local note_text="${2:-}"
    if [[ -n "${note_text}" ]]; then
        printf '%s\n\nNote: %s' "${prompt_text}" "${note_text}"
    else
        printf '%s' "${prompt_text}"
    fi
}


is_quick_wizard() {
    [[ "${WIZARD_PROFILE:-custom}" == "quick" ]]
}


choose_wizard_profile_interactively() {
    if ui_can_use_whiptail; then
        WIZARD_PROFILE="$(whiptail \
            --title "LIBERO Wizard Style" \
            --default-item "quick" \
            --menu "Choose the wizard style" \
            18 92 3 \
            "quick" "Ask only for the required options. Ports, seeds, recording, GPUs, and extra args use defaults." \
            "custom" "Show the full wizard with advanced controls and page-by-page explanations." \
            "defaults" "Edit the prefilled defaults used by the wizard before choosing quick or custom." \
            3>&1 1>&2 2>&3)" || exit 1
        return
    fi

    local input
    echo "Choose wizard style:"
    echo "  1) quick  Ask only for required options and keep advanced settings at defaults"
    echo "  2) custom Full wizard with explanations for each page"
    echo "  3) defaults Edit the wizard's prefilled defaults first"
    read -r -p "Enter 1, 2, or 3 [1]: " input
    case "${input:-1}" in
        1) WIZARD_PROFILE="quick" ;;
        2) WIZARD_PROFILE="custom" ;;
        3) WIZARD_PROFILE="defaults" ;;
        *)
            echo "[ERROR] Invalid wizard style selection: ${input}"
            exit 1
            ;;
    esac
}


load_gpu_options() {
    local defaults_str="${1:-}"
    local line idx label
    declare -A seen=()

    unset GPU_VALUES GPU_LABELS
    declare -ga GPU_VALUES=()
    declare -ga GPU_LABELS=()

    if command -v nvidia-smi >/dev/null 2>&1; then
        while IFS= read -r line; do
            [[ -z "${line}" ]] && continue
            idx="${line%%,*}"
            idx="${idx//[[:space:]]/}"
            [[ -z "${idx}" ]] && continue
            label="${line#*,}"
            label="${label#"${label%%[![:space:]]*}"}"
            if [[ -z "${seen[${idx}]:-}" ]]; then
                GPU_VALUES+=("${idx}")
                GPU_LABELS+=("GPU ${idx}: ${label}")
                seen["${idx}"]=1
            fi
        done < <(nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null || true)
    fi

    for idx in ${defaults_str}; do
        idx="${idx//[[:space:]]/}"
        [[ -z "${idx}" ]] && continue
        if [[ -z "${seen[${idx}]:-}" ]]; then
            GPU_VALUES+=("${idx}")
            GPU_LABELS+=("GPU ${idx}")
            seen["${idx}"]=1
        fi
    done

    if [[ ${#GPU_VALUES[@]} -eq 0 ]]; then
        for idx in 0 1 2 3; do
            GPU_VALUES+=("${idx}")
            GPU_LABELS+=("GPU ${idx}")
        done
    fi
}


recommended_eval_gpu_list() {
    local requested_count="${1:-1}"
    local excluded_values="${2:-}"
    local explicit_value="${3:-}"
    local -a excluded=()
    local -a selected=()
    local gpu

    if [[ -n "${explicit_value}" ]]; then
        echo "${explicit_value}"
        return
    fi

    if ! [[ "${requested_count}" =~ ^[0-9]+$ ]] || (( requested_count < 1 )); then
        requested_count=1
    fi

    read -r -a excluded <<< "${excluded_values}"
    load_gpu_options ""

    for gpu in "${GPU_VALUES[@]}"; do
        if gpu_list_contains "${gpu}" "${excluded[@]}"; then
            continue
        fi
        selected+=("${gpu}")
        if (( ${#selected[@]} >= requested_count )); then
            break
        fi
    done

    echo "${selected[*]}"
}


choose_single_gpu_interactively() {
    local __var_name="$1"
    local prompt_text="$2"
    local default_value="$3"
    local excluded_values="${4:-}"
    local note_text="${5:-}"
    local prompt_body="${prompt_text}"

    if [[ -n "${note_text}" ]]; then
        prompt_body+=$'\n\n'"${note_text}"
    fi

    load_gpu_options "${default_value} ${excluded_values}"

    if ui_can_use_whiptail; then
        local -a default_selections=()
        local -a excluded=()
        local -a dialog_options=()
        local -a selected=()
        local state choice idx selected_gpu excluded_gpu is_excluded
        read -r -a default_selections <<< "${default_value}"
        read -r -a excluded <<< "${excluded_values}"

        while true; do
            dialog_options=()
            for idx in "${!GPU_VALUES[@]}"; do
                is_excluded=false
                for excluded_gpu in "${excluded[@]}"; do
                    if [[ "${GPU_VALUES[$idx]}" == "${excluded_gpu}" ]]; then
                        is_excluded=true
                        break
                    fi
                done
                if [[ "${is_excluded}" == true ]]; then
                    continue
                fi
                state="OFF"
                for selected_gpu in "${default_selections[@]}"; do
                    if [[ "${GPU_VALUES[$idx]}" == "${selected_gpu}" ]]; then
                        state="ON"
                        break
                    fi
                done
                dialog_options+=("${GPU_VALUES[$idx]}" "${GPU_LABELS[$idx]}" "${state}")
            done

            if [[ ${#dialog_options[@]} -eq 0 ]]; then
                whiptail --title "LIBERO GPU Selection" --msgbox "No GPUs remain after applying the current exclusions. Choose a different policy GPU or run on a machine with more GPUs." 10 96
                exit 1
            fi

            choice="$(whiptail \
                --title "LIBERO GPU Selection" \
                --separate-output \
                --checklist "${prompt_body}" \
                20 100 12 \
                "${dialog_options[@]}" \
                3>&1 1>&2 2>&3)" || exit 1
            mapfile -t selected <<< "${choice}"
            if [[ ${#selected[@]} -eq 1 ]]; then
                printf -v "${__var_name}" '%s' "${selected[0]}"
                return
            fi
            whiptail --title "LIBERO GPU Selection" --msgbox "Select exactly one GPU." 8 50
        done
    fi

    echo "Available GPUs:"
    local listed_any=false
    for idx in "${!GPU_VALUES[@]}"; do
        if [[ " ${excluded_values} " == *" ${GPU_VALUES[$idx]} "* ]]; then
            continue
        fi
        listed_any=true
        printf "  %s) %s\n" "${GPU_VALUES[$idx]}" "${GPU_LABELS[$idx]}"
    done
    if [[ "${listed_any}" != true ]]; then
        echo "[ERROR] No GPUs remain after applying the current exclusions."
        exit 1
    fi
    prompt_with_default "${__var_name}" "${prompt_body} (GPU id)" "${default_value}"
}


choose_gpu_list_interactively() {
    local __var_name="$1"
    local prompt_text="$2"
    local default_value="$3"
    local excluded_values="${4:-}"
    local note_text="${5:-}"
    local prompt_body="${prompt_text}"

    if [[ -n "${note_text}" ]]; then
        prompt_body+=$'\n\n'"${note_text}"
    fi

    load_gpu_options "${default_value} ${excluded_values}"

    if ui_can_use_whiptail; then
        local -a default_selections=()
        local -a excluded=()
        local -a dialog_options=()
        local -a selected=()
        local state choice idx selected_gpu excluded_gpu is_excluded
        read -r -a default_selections <<< "${default_value}"
        read -r -a excluded <<< "${excluded_values}"

        while true; do
            dialog_options=()
            for idx in "${!GPU_VALUES[@]}"; do
                is_excluded=false
                for excluded_gpu in "${excluded[@]}"; do
                    if [[ "${GPU_VALUES[$idx]}" == "${excluded_gpu}" ]]; then
                        is_excluded=true
                        break
                    fi
                done
                if [[ "${is_excluded}" == true ]]; then
                    continue
                fi
                state="OFF"
                for selected_gpu in "${default_selections[@]}"; do
                    if [[ "${GPU_VALUES[$idx]}" == "${selected_gpu}" ]]; then
                        state="ON"
                        break
                    fi
                done
                dialog_options+=("${GPU_VALUES[$idx]}" "${GPU_LABELS[$idx]}" "${state}")
            done

            if [[ ${#dialog_options[@]} -eq 0 ]]; then
                whiptail --title "LIBERO GPU Selection" --msgbox "No GPUs remain after applying the current exclusions. Choose a different policy GPU or run on a machine with more GPUs." 10 96
                exit 1
            fi

            choice="$(whiptail \
                --title "LIBERO GPU Selection" \
                --separate-output \
                --checklist "${prompt_body}" \
                20 100 12 \
                "${dialog_options[@]}" \
                3>&1 1>&2 2>&3)" || exit 1
            mapfile -t selected <<< "${choice}"
            if [[ ${#selected[@]} -gt 0 ]]; then
                printf -v "${__var_name}" '%s' "${selected[*]}"
                return
            fi
            whiptail --title "LIBERO GPU Selection" --msgbox "Select at least one GPU." 8 50
        done
    fi

    echo "Available GPUs:"
    local listed_any=false
    for idx in "${!GPU_VALUES[@]}"; do
        if [[ " ${excluded_values} " == *" ${GPU_VALUES[$idx]} "* ]]; then
            continue
        fi
        listed_any=true
        printf "  %s) %s\n" "${GPU_VALUES[$idx]}" "${GPU_LABELS[$idx]}"
    done
    if [[ "${listed_any}" != true ]]; then
        echo "[ERROR] No GPUs remain after applying the current exclusions."
        exit 1
    fi
    prompt_with_default "${__var_name}" "${prompt_body} (space-separated GPU ids)" "${default_value}"
}


gpu_overlap_note() {
    echo "Keep the policy server and evaluation env on different GPUs. Sharing one GPU often causes memory contention, EGL instability, and spurious evaluation failures."
}


gpu_list_contains() {
    local target="$1"
    shift
    local gpu
    for gpu in "$@"; do
        if [[ "${gpu}" == "${target}" ]]; then
            return 0
        fi
    done
    return 1
}


ensure_distinct_gpu_pair() {
    local lhs_name="$1"
    local lhs_value="$2"
    local rhs_name="$3"
    local rhs_value="$4"
    if [[ "${lhs_value}" == "${rhs_value}" ]]; then
        echo "[ERROR] ${lhs_name} and ${rhs_name} must be different."
        echo "        $(gpu_overlap_note)"
        exit 1
    fi
}


ensure_gpu_list_excludes() {
    local excluded_gpu="$1"
    local gpu_list_str="$2"
    local label="${3:-GPU_LIST}"
    local -a gpu_array=()
    read -r -a gpu_array <<< "${gpu_list_str}"
    if gpu_list_contains "${excluded_gpu}" "${gpu_array[@]}"; then
        echo "[ERROR] ${label} must not include GPU ${excluded_gpu} because it is already reserved for the policy server."
        echo "        $(gpu_overlap_note)"
        exit 1
    fi
}


ui_can_use_whiptail() {
    command -v whiptail >/dev/null 2>&1 && [[ -t 0 ]] && [[ -t 1 ]]
}


choose_record_mode_interactively() {
    local default_value="${1:-failure}"
    local note_text="${2:-}"
    local prompt_text
    prompt_text="$(append_note_to_prompt "Choose video recording behavior" "${note_text}")"
    if ui_can_use_whiptail; then
        RECORD="$(whiptail \
            --title "LIBERO Record Mode" \
            --default-item "${default_value}" \
            --menu "${prompt_text}" \
            14 80 3 \
            "none" "Do not record videos" \
            "failure" "Record failed rollouts only" \
            "all" "Record every rollout" \
            3>&1 1>&2 2>&3)" || exit 1
        return
    fi
    prompt_with_default RECORD "$(append_note_to_prompt "Record mode (none/failure/all)" "${note_text}")" "${default_value}"
}


choose_parallel_server_mode_interactively() {
    local note_text="${1:-}"
    local prompt_text
    prompt_text="$(append_note_to_prompt "Choose how to supply the policy server" "${note_text}")"
    local default_value="${PARALLEL_SERVER_MODE:-local}"
    if ui_can_use_whiptail; then
        PARALLEL_SERVER_MODE="$(whiptail \
            --title "Parallel Server Mode" \
            --default-item "${default_value}" \
            --menu "${prompt_text}" \
            14 90 2 \
            "local" "Start a local server from a checkpoint in this script" \
            "existing" "Connect to an already running server" \
            3>&1 1>&2 2>&3)" || exit 1
        return
    fi

    local input
    if [[ -n "${note_text}" ]]; then
        printf '%s\n\n' "${note_text}"
    fi
    echo "Choose policy server source:"
    echo "  1) local    Start a local server from a checkpoint"
    echo "  2) existing Connect to an already running server"
    if [[ "${default_value}" == "existing" ]]; then
        read -r -p "Enter 1 or 2 [2]: " input
        input="${input:-2}"
    else
        read -r -p "Enter 1 or 2 [1]: " input
        input="${input:-1}"
    fi
    case "${input}" in
        1) PARALLEL_SERVER_MODE="local" ;;
        2) PARALLEL_SERVER_MODE="existing" ;;
        *)
            echo "[ERROR] Invalid server mode selection: ${input}"
            exit 1
            ;;
    esac
}


choose_benchmark_mode_interactively() {
    local note_text="${1:-}"
    local prompt_text
    prompt_text="$(append_note_to_prompt "Choose how suites run for each checkpoint" "${note_text}")"
    local default_value="${BENCHMARK_MODE:-serial}"
    if ui_can_use_whiptail; then
        BENCHMARK_MODE="$(whiptail \
            --title "Benchmark Eval Mode" \
            --default-item "${default_value}" \
            --menu "${prompt_text}" \
            14 80 2 \
            "serial" "Run suites sequentially for each checkpoint" \
            "parallel" "Run suites with the GPU worker queue for each checkpoint" \
            3>&1 1>&2 2>&3)" || exit 1
        return
    fi

    local input
    if [[ -n "${note_text}" ]]; then
        printf '%s\n\n' "${note_text}"
    fi
    echo "Choose benchmark evaluation mode:"
    echo "  1) serial   Run suites sequentially for each checkpoint"
    echo "  2) parallel Run suites with the GPU worker queue for each checkpoint"
    if [[ "${default_value}" == "parallel" ]]; then
        read -r -p "Enter 1 or 2 [2]: " input
        input="${input:-2}"
    else
        read -r -p "Enter 1 or 2 [1]: " input
        input="${input:-1}"
    fi
    case "${input}" in
        1) BENCHMARK_MODE="serial" ;;
        2) BENCHMARK_MODE="parallel" ;;
        *)
            echo "[ERROR] Invalid benchmark mode selection: ${input}"
            exit 1
            ;;
    esac
}


choose_checkpoint_steps_interactively() {
    local base_dir="$1"
    local note_text="${2:-}"
    local -a step_options=()
    local -a selected_steps=()
    local -a default_steps=()
    local step

    if [[ ! -d "${base_dir}" ]]; then
        echo "[ERROR] Checkpoint base directory not found: ${base_dir}"
        exit 1
    fi

    mapfile -t step_options < <(find "${base_dir}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -V)
    if [[ ${#step_options[@]} -eq 0 ]]; then
        echo "[ERROR] No checkpoint subdirectories found in: ${base_dir}"
        exit 1
    fi

    if ui_can_use_whiptail; then
        local -a dialog_options=()
        local choice
        local prompt_text
        prompt_text="$(append_note_to_prompt "Choose checkpoint steps under ${base_dir}" "${note_text}")"
        if [[ -n "${CKPT_STEPS_STR:-}" ]]; then
            read -r -a default_steps <<< "${CKPT_STEPS_STR}"
        fi
        for step in "${step_options[@]}"; do
            local state="OFF"
            if gpu_list_contains "${step}" "${default_steps[@]}"; then
                state="ON"
            fi
            dialog_options+=("${step}" "${base_dir}/${step}" "${state}")
        done
        choice="$(whiptail \
            --title "Benchmark Checkpoints" \
            --separate-output \
            --checklist "${prompt_text}" \
            24 100 14 \
            "${dialog_options[@]}" \
            3>&1 1>&2 2>&3)" || exit 1
        mapfile -t selected_steps <<< "${choice}"
    else
        if [[ -n "${note_text}" ]]; then
            printf '%s\n\n' "${note_text}"
        fi
        echo "Available checkpoint steps under ${base_dir}:"
        printf '  %s\n' "${step_options[@]}"
        prompt_required CKPT_STEPS_STR "Checkpoint steps (space-separated)"
        read -r -a selected_steps <<< "${CKPT_STEPS_STR}"
    fi

    if [[ ${#selected_steps[@]} -eq 0 ]]; then
        echo "[ERROR] At least one checkpoint step must be selected."
        exit 1
    fi

    CKPT_STEPS_STR="${selected_steps[*]}"
}


choose_suites_interactively() {
    local note_text="${1:-}"
    local -a default_suites=()
    if [[ -n "${TASK_SUITES:-}" ]]; then
        read -r -a default_suites <<< "${TASK_SUITES}"
    else
        default_suites=("${DEFAULT_SUITES[@]}")
    fi
    if ui_can_use_whiptail; then
        local choice
        local prompt_text
        local -a dialog_options=()
        local suite
        prompt_text="$(append_note_to_prompt "Choose the suites to evaluate" "${note_text}")"
        for suite in "${DEFAULT_SUITES[@]}"; do
            local label=""
            case "${suite}" in
                libero_spatial) label="Spatial tasks" ;;
                libero_object) label="Object tasks" ;;
                libero_goal) label="Goal tasks" ;;
                libero_10) label="LIBERO-10" ;;
            esac
            local state="OFF"
            if gpu_list_contains "${suite}" "${default_suites[@]}"; then
                state="ON"
            fi
            dialog_options+=("${suite}" "${label}" "${state}")
        done
        choice="$(whiptail \
            --title "LIBERO Suites" \
            --separate-output \
            --checklist "${prompt_text}" \
            20 80 10 \
            "${dialog_options[@]}" \
            3>&1 1>&2 2>&3)" || exit 1
        mapfile -t SUITES <<< "${choice}"
    else
        local input raw_index index suite
        if [[ -n "${note_text}" ]]; then
            printf '%s\n\n' "${note_text}"
        fi
        echo "Choose suites to evaluate:"
        for index in "${!DEFAULT_SUITES[@]}"; do
            printf "  %d) %s\n" "$((index + 1))" "${DEFAULT_SUITES[$index]}"
        done
        local default_input="all"
        if [[ "${TASK_SUITES:-}" != "${DEFAULT_SUITES[*]}" && -n "${TASK_SUITES:-}" ]]; then
            default_input="${TASK_SUITES}"
        fi
        read -r -p "Enter comma-separated indices, space-separated suite names, or 'all' [${default_input}]: " input
        input="${input:-${default_input}}"

        SUITES=()
        if [[ "${input}" == "all" ]]; then
            SUITES=("${DEFAULT_SUITES[@]}")
        elif [[ "${input}" =~ libero_ ]]; then
            read -r -a SUITES <<< "${input}"
        else
            IFS=',' read -r -a raw_indices <<< "${input}"
            for raw_index in "${raw_indices[@]}"; do
                raw_index="${raw_index// /}"
                if [[ ! "${raw_index}" =~ ^[0-9]+$ ]]; then
                    echo "[ERROR] Invalid suite index: ${raw_index}"
                    exit 1
                fi
                index=$((raw_index - 1))
                if (( index < 0 || index >= ${#DEFAULT_SUITES[@]} )); then
                    echo "[ERROR] Suite index out of range: ${raw_index}"
                    exit 1
                fi
                suite="${DEFAULT_SUITES[$index]}"
                if [[ " ${SUITES[*]} " != *" ${suite} "* ]]; then
                    SUITES+=("${suite}")
                fi
            done
        fi
    fi

    if [[ ${#SUITES[@]} -eq 0 ]]; then
        echo "[ERROR] At least one suite must be selected."
        exit 1
    fi
    TASK_SUITES="${SUITES[*]}"
}


choose_mode_interactively() {
    local note_text="${1:-}"
    local prompt_text
    prompt_text="$(append_note_to_prompt "Choose execution mode" "${note_text}")"
    local default_value="${EXECUTION_MODE:-serial}"
    if ui_can_use_whiptail; then
        EXECUTION_MODE="$(whiptail \
            --title "LIBERO Eval Mode" \
            --default-item "${default_value}" \
            --menu "${prompt_text}" \
            18 80 4 \
            "serial" "Start a local custom-checkpoint server and run suites sequentially" \
            "parallel" "Run multiple suites with the GPU worker queue" \
            "benchmark" "Sweep multiple checkpoints and evaluate each one" \
            3>&1 1>&2 2>&3)" || exit 1
        return
    fi

    local input
    if [[ -n "${note_text}" ]]; then
        printf '%s\n\n' "${note_text}"
    fi
    echo "Choose execution mode:"
    echo "  1) serial   Start a local custom-checkpoint server and run suites sequentially"
    echo "  2) parallel Run multiple suites with the GPU worker queue"
    echo "  3) benchmark Sweep multiple checkpoints"
    case "${default_value}" in
        parallel) read -r -p "Enter 1, 2, or 3 [2]: " input; input="${input:-2}" ;;
        benchmark) read -r -p "Enter 1, 2, or 3 [3]: " input; input="${input:-3}" ;;
        *) read -r -p "Enter 1, 2, or 3 [1]: " input; input="${input:-1}" ;;
    esac
    case "${input}" in
        1) EXECUTION_MODE="serial" ;;
        2) EXECUTION_MODE="parallel" ;;
        3) EXECUTION_MODE="benchmark" ;;
        *)
            echo "[ERROR] Invalid mode selection: ${input}"
            exit 1
            ;;
    esac
}


edit_interactive_defaults() {
    local session_note="These edits only apply to the current wizard session. They change the prefilled values used on the next pages."
    local suites_note="Set which suites should be preselected by default."
    local mode_note="Set which execution mode should be preselected by default."
    local server_mode_note="Default source for the policy server when parallel mode asks."
    local benchmark_mode_note="Default benchmark execution mode."

    choose_suites_interactively "$(append_note_to_prompt "${suites_note}" "${session_note}")"
    choose_mode_interactively "$(append_note_to_prompt "${mode_note}" "${session_note}")"
    prompt_with_default CONFIG "$(append_note_to_prompt "Default policy config" "${session_note}")" "${CONFIG:-stage3_finetuning_libero}"
    prompt_optional CHECKPOINT_DIR "$(append_note_to_prompt "Default checkpoint directory" "${session_note}")"
    prompt_optional CKPT_BASE "$(append_note_to_prompt "Default checkpoint base directory" "${session_note}")"
    choose_parallel_server_mode_interactively "$(append_note_to_prompt "${server_mode_note}" "${session_note}")"
    choose_benchmark_mode_interactively "$(append_note_to_prompt "${benchmark_mode_note}" "${session_note}")"
    prompt_optional HOST "$(append_note_to_prompt "Default policy server host" "${session_note}")"
    prompt_with_default PORT "$(append_note_to_prompt "Default policy server port" "${session_note}")" "${PORT:-8001}"
    prompt_with_default TRIALS "$(append_note_to_prompt "Default trials per task" "${session_note}")" "${TRIALS:-50}"
    choose_record_mode_interactively "${RECORD:-failure}" "${session_note}"
    prompt_with_default SEED "$(append_note_to_prompt "Default random seed" "${session_note}")" "${SEED:-7}"
    choose_single_gpu_interactively SERVER_GPU "Choose the default policy GPU" "${SERVER_GPU:-0}" "" "${session_note}"
    choose_single_gpu_interactively CLIENT_GPU "Choose the default evaluation GPU" "${CLIENT_GPU:-1}" "${SERVER_GPU}" "$(append_note_to_prompt "${gpu_overlap_note}" "${session_note}")"
    GPU_LIST="${GPU_LIST:-$(recommended_eval_gpu_list "${#SUITES[@]}" "${SERVER_GPU}" "")}"
    choose_gpu_list_interactively GPU_LIST "Choose the default parallel evaluation GPUs" "${GPU_LIST}" "${SERVER_GPU}" "$(append_note_to_prompt "${gpu_overlap_note}" "${session_note}")"
    prompt_with_default RUN_NAME "$(append_note_to_prompt "Default parallel run name" "${session_note}")" "${RUN_NAME:-libero_eval}"
    prompt_optional TASK_IDS "$(append_note_to_prompt "Default task IDs (space-separated)" "${session_note}")"
    prompt_optional INTERACTIVE_EXTRA_ARGS_RAW "$(append_note_to_prompt "Default additional main.py args" "${session_note}")"
    if [[ -n "${CKPT_BASE:-}" ]]; then
        choose_checkpoint_steps_interactively "${CKPT_BASE}" "${session_note}"
    else
        prompt_optional CKPT_STEPS_STR "$(append_note_to_prompt "Default benchmark checkpoint steps (space-separated)" "${session_note}")"
    fi
}


run_quick_interactive_mode() {
    local recommended_gpu_list

    if [[ "${EXECUTION_MODE}" == "serial" ]]; then
        prompt_with_default CONFIG "Policy config" "${CONFIG:-stage3_finetuning_libero}"
        prompt_required CHECKPOINT_DIR "Checkpoint directory"
        SERVER_GPU="${SERVER_GPU:-0}"
        CLIENT_GPU="${CLIENT_GPU:-$(recommended_eval_gpu_list 1 "${SERVER_GPU}" "")}"
        if [[ -z "${CLIENT_GPU}" ]]; then
            echo "[ERROR] Quick mode could not find a free evaluation GPU after reserving GPU ${SERVER_GPU} for the policy server."
            echo "Use custom mode to choose different GPUs manually."
            exit 1
        fi
        run_serial_mode
        return
    fi

    if [[ "${EXECUTION_MODE}" == "parallel" ]]; then
        choose_parallel_server_mode_interactively
        if [[ "${PARALLEL_SERVER_MODE}" == "local" ]]; then
            prompt_required CHECKPOINT_DIR "Checkpoint directory"
            prompt_with_default CONFIG "Policy config" "${CONFIG:-stage3_finetuning_libero}"
            SERVER_GPU="${SERVER_GPU:-0}"
        else
            CHECKPOINT_DIR=""
            prompt_with_default HOST "Policy server host" "${HOST:-127.0.0.1}"
        fi
        GPU_LIST="${GPU_LIST:-$(recommended_eval_gpu_list "${#SUITES[@]}" "${SERVER_GPU:-}" "")}"
        if [[ -z "${GPU_LIST}" ]]; then
            echo "[ERROR] Quick mode could not find any free evaluation GPU after applying the current exclusions."
            echo "Use custom mode to choose GPUs manually."
            exit 1
        fi
        run_parallel_mode
        return
    fi

    prompt_with_default CONFIG "Policy config" "${CONFIG:-stage3_finetuning_libero}"
    prompt_required CKPT_BASE "Checkpoint base directory"
    choose_checkpoint_steps_interactively "${CKPT_BASE}"
    choose_benchmark_mode_interactively
    SERVER_GPU="${SERVER_GPU:-0}"
    if [[ "${BENCHMARK_MODE}" == "parallel" ]]; then
        GPU_LIST="${GPU_LIST:-$(recommended_eval_gpu_list "${#SUITES[@]}" "${SERVER_GPU}" "")}"
        if [[ -z "${GPU_LIST}" ]]; then
            echo "[ERROR] Quick mode could not find any free evaluation GPU after reserving GPU ${SERVER_GPU} for the policy server."
            echo "Use custom mode to choose GPUs manually."
            exit 1
        fi
    else
        CLIENT_GPU="${CLIENT_GPU:-$(recommended_eval_gpu_list 1 "${SERVER_GPU}" "")}"
        if [[ -z "${CLIENT_GPU}" ]]; then
            echo "[ERROR] Quick mode could not find a free evaluation GPU after reserving GPU ${SERVER_GPU} for the policy server."
            echo "Use custom mode to choose GPUs manually."
            exit 1
        fi
    fi
    run_benchmark_mode
}


run_custom_interactive_mode() {
    local suites_note="Each suite is a benchmark family. Keep all selected for the full benchmark, or narrow the run before you optionally filter further with task IDs."
    local mode_note="serial starts one local policy server and runs suites sequentially. parallel uses a GPU worker queue. benchmark sweeps multiple checkpoint steps."
    local config_note="PLaW-VLA config name used by the policy server when loading the checkpoint. It must match the checkpoint layout, for example stage3_finetuning_libero."
    local checkpoint_note="Directory of the checkpoint to serve locally. In serial mode this is the one checkpoint you evaluate."
    local ckpt_base_note="Parent directory that contains checkpoint step subdirectories such as 29000 or 30000."
    local steps_note="Select which checkpoint step directories under the base path should be evaluated in this benchmark sweep."
    local benchmark_mode_note="serial runs suites one by one for each checkpoint. parallel fans suites out across a GPU worker queue for each checkpoint."
    local server_mode_note="local starts the policy server from CHECKPOINT_DIR inside this script. existing connects to a server that is already running at HOST:PORT."
    local port_note="Websocket port used by the policy server. Keep 8001 unless you already have another listener on that port."
    local host_note="Host name or IP of an already running policy server."
    local trials_note="How many rollouts to run for each task. Higher values improve metric stability but take longer."
    local record_note="none disables videos, failure records only failed rollouts, and all records every rollout."
    local seed_note="Evaluation random seed. Change it only when you intentionally want a different rollout sample."
    local policy_gpu_note="GPU reserved for the policy server process."
    local eval_gpu_note="GPU used by the simulator / evaluation process. It must stay different from the policy GPU."
    local eval_gpu_list_note="Select the GPUs used by parallel evaluation workers. The policy GPU stays excluded to avoid contention."
    local run_name_note="Short label added to the parallel results directory name so runs are easier to identify later."
    local task_ids_note="Optional subset of task IDs inside the selected suites, for example '0 3 5'. Leave empty to run every task in each selected suite."
    local extra_args_note="Extra flags forwarded directly to examples/libero/main.py. Leave empty unless you are debugging a specific client-side option."
    local recommended_gpu_list

    choose_suites_interactively "${suites_note}"
    choose_mode_interactively "${mode_note}"

    if [[ "${EXECUTION_MODE}" == "serial" ]]; then
        prompt_with_default CONFIG "$(append_note_to_prompt "Policy config" "${config_note}")" "${CONFIG:-stage3_finetuning_libero}"
        prompt_required CHECKPOINT_DIR "$(append_note_to_prompt "Checkpoint directory" "${checkpoint_note}")"
        prompt_with_default PORT "$(append_note_to_prompt "Server port" "${port_note}")" "${PORT:-8001}"
        prompt_with_default TRIALS "$(append_note_to_prompt "Trials per task" "${trials_note}")" "${TRIALS:-50}"
        choose_record_mode_interactively "${RECORD:-failure}" "${record_note}"
        prompt_with_default SEED "$(append_note_to_prompt "Random seed" "${seed_note}")" "${SEED:-42}"
        choose_single_gpu_interactively SERVER_GPU "Choose the policy GPU" "${SERVER_GPU:-0}" "" "${policy_gpu_note}"
        choose_single_gpu_interactively CLIENT_GPU "Choose the evaluation GPU" "${CLIENT_GPU:-1}" "${SERVER_GPU}" "$(append_note_to_prompt "${gpu_overlap_note}" "${eval_gpu_note}")"
        prompt_optional TASK_IDS "$(append_note_to_prompt "Optional task IDs (space-separated)" "${task_ids_note}")"
        prompt_optional INTERACTIVE_EXTRA_ARGS_RAW "$(append_note_to_prompt "Additional main.py args (optional)" "${extra_args_note}")"
        if [[ -n "${INTERACTIVE_EXTRA_ARGS_RAW}" ]]; then
            read -r -a INTERACTIVE_EXTRA_ARGS <<< "${INTERACTIVE_EXTRA_ARGS_RAW}"
        else
            INTERACTIVE_EXTRA_ARGS=()
        fi
        run_serial_mode "${INTERACTIVE_EXTRA_ARGS[@]}"
        return
    fi

    if [[ "${EXECUTION_MODE}" == "parallel" ]]; then
        choose_parallel_server_mode_interactively "${server_mode_note}"
        if [[ "${PARALLEL_SERVER_MODE}" == "local" ]]; then
            prompt_required CHECKPOINT_DIR "$(append_note_to_prompt "Checkpoint directory" "${checkpoint_note}")"
            prompt_with_default CONFIG "$(append_note_to_prompt "Policy config" "${config_note}")" "${CONFIG:-stage3_finetuning_libero}"
            prompt_with_default PORT "$(append_note_to_prompt "Server port" "${port_note}")" "${PORT:-8001}"
            choose_single_gpu_interactively SERVER_GPU "Choose the policy GPU" "${SERVER_GPU:-0}" "" "${policy_gpu_note}"
        else
            CHECKPOINT_DIR=""
            prompt_with_default HOST "$(append_note_to_prompt "Policy server host" "${host_note}")" "${HOST:-127.0.0.1}"
            prompt_with_default PORT "$(append_note_to_prompt "Policy server port" "${port_note}")" "${PORT:-8001}"
        fi
        prompt_with_default TRIALS "$(append_note_to_prompt "Trials per task" "${trials_note}")" "${TRIALS:-10}"
        choose_record_mode_interactively "${RECORD:-failure}" "${record_note}"
        prompt_with_default SEED "$(append_note_to_prompt "Random seed" "${seed_note}")" "${SEED:-7}"
        recommended_gpu_list="$(recommended_eval_gpu_list "${#SUITES[@]}" "${SERVER_GPU:-}" "${GPU_LIST:-}")"
        choose_gpu_list_interactively GPU_LIST "Choose one or more evaluation GPUs" "${recommended_gpu_list}" "${SERVER_GPU:-}" "$(append_note_to_prompt "${gpu_overlap_note}" "${eval_gpu_list_note}")"
        prompt_with_default RUN_NAME "$(append_note_to_prompt "Run name" "${run_name_note}")" "${RUN_NAME:-libero_eval}"
        prompt_optional TASK_IDS "$(append_note_to_prompt "Optional task IDs (space-separated)" "${task_ids_note}")"
        prompt_optional INTERACTIVE_EXTRA_ARGS_RAW "$(append_note_to_prompt "Additional main.py args (optional)" "${extra_args_note}")"
        if [[ -n "${INTERACTIVE_EXTRA_ARGS_RAW}" ]]; then
            read -r -a INTERACTIVE_EXTRA_ARGS <<< "${INTERACTIVE_EXTRA_ARGS_RAW}"
        else
            INTERACTIVE_EXTRA_ARGS=()
        fi
        run_parallel_mode "${INTERACTIVE_EXTRA_ARGS[@]}"
        return
    fi

    prompt_with_default CONFIG "$(append_note_to_prompt "Policy config" "${config_note}")" "${CONFIG:-stage3_finetuning_libero}"
    prompt_required CKPT_BASE "$(append_note_to_prompt "Checkpoint base directory" "${ckpt_base_note}")"
    choose_checkpoint_steps_interactively "${CKPT_BASE}" "${steps_note}"
    choose_benchmark_mode_interactively "${benchmark_mode_note}"
    prompt_with_default PORT "$(append_note_to_prompt "Server port" "${port_note}")" "${PORT:-8001}"
    choose_single_gpu_interactively SERVER_GPU "Choose the policy GPU" "${SERVER_GPU:-0}" "" "${policy_gpu_note}"
    prompt_with_default TRIALS "$(append_note_to_prompt "Trials per task" "${trials_note}")" "${TRIALS:-50}"
    choose_record_mode_interactively "${RECORD:-failure}" "${record_note}"
    prompt_with_default SEED "$(append_note_to_prompt "Random seed" "${seed_note}")" "${SEED:-7}"
    if [[ "${BENCHMARK_MODE}" == "parallel" ]]; then
        recommended_gpu_list="$(recommended_eval_gpu_list "${#SUITES[@]}" "${SERVER_GPU}" "${GPU_LIST:-}")"
        choose_gpu_list_interactively GPU_LIST "Choose one or more evaluation GPUs" "${recommended_gpu_list}" "${SERVER_GPU}" "$(append_note_to_prompt "${gpu_overlap_note}" "${eval_gpu_list_note}")"
    else
        choose_single_gpu_interactively CLIENT_GPU "Choose the evaluation GPU" "${CLIENT_GPU:-1}" "${SERVER_GPU}" "$(append_note_to_prompt "${gpu_overlap_note}" "${eval_gpu_note}")"
    fi
    prompt_optional TASK_IDS "$(append_note_to_prompt "Optional task IDs (space-separated)" "${task_ids_note}")"
    prompt_optional INTERACTIVE_EXTRA_ARGS_RAW "$(append_note_to_prompt "Additional main.py args (optional)" "${extra_args_note}")"
    if [[ -n "${INTERACTIVE_EXTRA_ARGS_RAW}" ]]; then
        read -r -a INTERACTIVE_EXTRA_ARGS <<< "${INTERACTIVE_EXTRA_ARGS_RAW}"
    else
        INTERACTIVE_EXTRA_ARGS=()
    fi
    run_benchmark_mode "${INTERACTIVE_EXTRA_ARGS[@]}"
}


run_interactive_mode() {
    if ! [[ -t 0 ]]; then
        echo "[ERROR] Interactive mode requires a TTY. Use an explicit mode instead."
        usage
        exit 1
    fi

    echo "LIBERO evaluation wizard"
    echo ""

    while true; do
        choose_wizard_profile_interactively

        if [[ "${WIZARD_PROFILE}" == "defaults" ]]; then
            edit_interactive_defaults
            continue
        fi

        if is_quick_wizard; then
            choose_suites_interactively
            choose_mode_interactively
            run_quick_interactive_mode
            return
        fi

        run_custom_interactive_mode
        return
    done
}


run_client_eval() {
    local suite="$1"
    local host="$2"
    local port="$3"
    local trials="$4"
    local record="$5"
    local seed="$6"
    local video_dir="$7"
    local log_file="$8"
    local gpu="$9"
    shift 9

    local -a extra_args=("$@")
    local -a cmd=(
        uv run --project "${SCRIPT_DIR}" --frozen python "${MAIN_SCRIPT}"
        --task-suite-name "${suite}"
        --host "${host}"
        --port "${port}"
        --num-trials-per-task "${trials}"
        --record-video "${record}"
        --video-out-path "${video_dir}"
        --seed "${seed}"
    )

    if [[ ${#TASK_ID_ARGS[@]} -gt 0 ]]; then
        cmd+=("${TASK_ID_ARGS[@]}")
    fi
    if [[ ${#extra_args[@]} -gt 0 ]]; then
        cmd+=("${extra_args[@]}")
    fi

    mkdir -p "${video_dir}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    EGL_DEVICE_ID="${gpu}" \
    MUJOCO_GL="${MUJOCO_GL:-egl}" \
    PYTHONWARNINGS="ignore" \
    "${cmd[@]}" > "${log_file}"
}


start_local_server() {
    local checkpoint_dir="$1"
    local log_file="$2"
    local label="${3:-Starting local LIBERO policy server}"

    validate_checkpoint_dir "${checkpoint_dir}"
    ensure_port_available "${PORT}"

    echo "[INFO] ${label}"
    echo "  Config: ${CONFIG}"
    echo "  Checkpoint: ${checkpoint_dir}"
    echo "  Port: ${PORT}"
    echo "  Policy GPU: ${SERVER_GPU}"

    CUDA_VISIBLE_DEVICES="${SERVER_GPU}" \
    uv run "${PROJECT_ROOT}/scripts/serve_policy.py" \
        --env LIBERO \
        --port "${PORT}" \
        policy:checkpoint \
        --policy.config="${CONFIG}" \
        --policy.dir="${checkpoint_dir}" \
        > "${log_file}" 2>&1 &
    SERVER_PID=$!
    echo "[INFO] Server PID=${SERVER_PID}"

    wait_for_server "${PORT}" "${SERVER_PID}" "${log_file}"
}


print_suite_summary() {
    local summary_file="$1"
    local total_elapsed="$2"
    local overall_ok="$3"
    local exit_on_failure="${4:-true}"

    {
        echo ""
        echo -e "${BOLD}==============================================================================${RESET}"
        echo -e "${BOLD}LIBERO Evaluation Results${RESET}"
        echo -e "${BOLD}$(date +%Y-%m-%d\ %H:%M:%S)${RESET}"
        echo -e "${BOLD}------------------------------------------------------------------------------${RESET}"
        printf "%-16s | %-36s | %-8s | %-7s\n" \
            "Suite" "Task" "Result" "Rate"
        echo "-----------------+--------------------------------------+----------+--------"

        local grand_succ=0
        local grand_total=0
        local valid_suite_count=0
        local valid_rate_sum=0

        for suite in "${SUITE_ORDER[@]}"; do
            local log_file="${RESULTS_DIR}/${suite}.log"
            local task_lines
            task_lines="$(extract_task_lines "${log_file}")"
            local suite_rate="${SUITE_RESULTS[${suite}]}"
            local first=true

            if [[ -n "${task_lines}" ]]; then
                while IFS='|' read -r _tid_desc task_desc result rate; do
                    local short_desc="${task_desc}"
                    if [[ ${#short_desc} -gt 36 ]]; then
                        short_desc="${short_desc:0:33}..."
                    fi
                    local suite_col=""
                    if ${first}; then
                        suite_col="${suite}"
                        first=false
                    fi
                    printf "${CYAN}%-16s${RESET} | %-36s | %8s | %7s\n" \
                        "${suite_col}" "${short_desc}" "${result}" "${rate}"
                done <<< "${task_lines}"

                local suite_total_line
                suite_total_line="$(extract_results_block "${log_file}" | grep '^suite_total=' | head -1 | sed 's/^suite_total=//')"
                if [[ -n "${suite_total_line}" ]]; then
                    local st_result st_rate succ_count total_count
                    st_result="$(echo "${suite_total_line}" | cut -d'|' -f1)"
                    st_rate="$(echo "${suite_total_line}" | cut -d'|' -f2)"
                    succ_count="$(echo "${st_result}" | cut -d/ -f1)"
                    total_count="$(echo "${st_result}" | cut -d/ -f2)"
                    grand_succ=$((grand_succ + succ_count))
                    grand_total=$((grand_total + total_count))
                    if [[ "${st_rate}" =~ ^[0-9.]+$ ]]; then
                        valid_suite_count=$((valid_suite_count + 1))
                        valid_rate_sum="$(awk "BEGIN {printf \"%.4f\", ${valid_rate_sum} + ${st_rate}}")"
                    fi
                    printf "                 | ${BOLD}Subtotal${RESET}                             | ${BOLD}%8s${RESET} | ${BOLD}%7s${RESET}\n" \
                        "${st_result}" "${st_rate}"
                fi
            else
                printf "${CYAN}%-16s${RESET} | %-36s | %8s | %7s\n" \
                    "${suite}" "(no per-task data)" "" "${suite_rate}"
            fi
            echo "-----------------+--------------------------------------+----------+--------"
        done

        local avg="N/A"
        if [[ ${valid_suite_count} -gt 0 ]]; then
            avg="$(awk "BEGIN {printf \"%.4f\", ${valid_rate_sum} / ${valid_suite_count}}")"
        fi
        local grand_result="${grand_succ}/${grand_total}"

        printf "${YELLOW}%-16s${RESET} | ${BOLD}%-36s${RESET} | ${BOLD}%8s${RESET} | ${BOLD}%7s${RESET}\n" \
            "AVERAGE" "" "${grand_result}" "${avg}"
        echo -e "${BOLD}==============================================================================${RESET}"
        echo ""
        echo -e "  Total time: ${BOLD}$(elapsed_str "${total_elapsed}")${RESET}    Results dir: ${DIM}${RESULTS_DIR}${RESET}"
        echo ""
    } | tee "${summary_file}"

    sed 's/\x1b\[[0-9;]*m//g' "${summary_file}" > "${summary_file}.plain"
    mv "${summary_file}.plain" "${summary_file}"

    echo "Summary saved to: ${summary_file}"
    if [[ "${overall_ok}" != true && "${exit_on_failure}" == true ]]; then
        exit 1
    fi
}


evaluate_suites_sequential() {
    local client_gpu="$1"
    shift

    init_suite_tracking

    local overall_ok=true
    local total_start=$SECONDS
    local suite log_file video_dir suite_start

    for suite in "${SUITES[@]}"; do
        log_file="${RESULTS_DIR}/${suite}.log"
        video_dir="${RESULTS_DIR}/videos/${suite}"
        suite_start=$SECONDS
        echo -e "${BOLD}Running ${suite}${RESET}  ${DIM}(log: ${log_file})${RESET}"

        if run_client_eval "${suite}" "${HOST}" "${PORT}" "${TRIALS}" "${RECORD}" "${SEED}" "${video_dir}" "${log_file}" "${client_gpu}" "$@"; then
            SUITE_RESULTS["${suite}"]="$(extract_success_rate "${log_file}")"
            echo -e "  ${GREEN}OK${RESET} ${suite}  success rate: ${GREEN}${SUITE_RESULTS[${suite}]}${RESET}  ($(elapsed_str "$((SECONDS - suite_start))"))"
        else
            SUITE_RESULTS["${suite}"]="FAILED"
            echo -e "  ${RED}FAIL${RESET} ${suite}  ${RED}FAILED${RESET}  ($(elapsed_str "$((SECONDS - suite_start))"))  - see ${log_file}"
            overall_ok=false
        fi
        SUITE_TIMES["${suite}"]=$((SECONDS - suite_start))
        SUITE_ORDER+=("${suite}")
    done

    EVAL_TOTAL_ELAPSED=$((SECONDS - total_start))
    [[ "${overall_ok}" == true ]]
}


evaluate_suites_parallel() {
    local gpu_list="$1"
    shift

    read -r -a GPU_ARRAY <<< "${gpu_list}"
    if [[ ${#GPU_ARRAY[@]} -eq 0 ]]; then
        echo "[ERROR] Provide at least one GPU id via GPU_LIST."
        exit 1
    fi

    init_suite_tracking

    declare -A PID_TO_SUITE=()
    declare -A SUITE_STARTS=()
    declare -A PID_TO_GPU=()
    declare -a ACTIVE_PIDS=()
    declare -a AVAILABLE_GPUS=("${GPU_ARRAY[@]}")

    cleanup_parallel_workers() {
        local exit_code=$?
        local pid
        for pid in "${ACTIVE_PIDS[@]:-}"; do
            if kill -0 "${pid}" >/dev/null 2>&1; then
                kill "${pid}" >/dev/null 2>&1 || true
            fi
        done
        if [[ ${#ACTIVE_PIDS[@]} -gt 0 ]]; then
            wait "${ACTIVE_PIDS[@]}" >/dev/null 2>&1 || true
        fi
        exit "${exit_code}"
    }
    trap cleanup_parallel_workers INT TERM

    local total_start=$SECONDS
    local overall_ok=true
    local queue_index=0
    local suite gpu log_file video_dir pid idx finished_pid wait_status finished_suite finished_gpu

    SUITE_ORDER=("${SUITES[@]}")

    while (( queue_index < ${#SUITES[@]} || ${#ACTIVE_PIDS[@]} > 0 )); do
        while (( queue_index < ${#SUITES[@]} && ${#AVAILABLE_GPUS[@]} > 0 )); do
            suite="${SUITES[$queue_index]}"
            gpu="${AVAILABLE_GPUS[0]}"
            if (( ${#AVAILABLE_GPUS[@]} == 1 )); then
                AVAILABLE_GPUS=()
            else
                AVAILABLE_GPUS=("${AVAILABLE_GPUS[@]:1}")
            fi
            log_file="${RESULTS_DIR}/${suite}.log"
            video_dir="${RESULTS_DIR}/videos/${suite}"
            mkdir -p "${video_dir}"
            SUITE_STARTS["${suite}"]=$SECONDS

            local -a cmd=(
                uv run --project "${SCRIPT_DIR}" --frozen python -u "${MAIN_SCRIPT}"
                --host "${HOST}"
                --port "${PORT}"
                --task-suite-name "${suite}"
                --num-trials-per-task "${TRIALS}"
                --record-video "${RECORD}"
                --video-out-path "${video_dir}"
                --seed "${SEED}"
            )
            if [[ ${#TASK_ID_ARGS[@]} -gt 0 ]]; then
                cmd+=("${TASK_ID_ARGS[@]}")
            fi
            if [[ $# -gt 0 ]]; then
                cmd+=("$@")
            fi

            CUDA_VISIBLE_DEVICES="${gpu}" \
            EGL_DEVICE_ID="${gpu}" \
            MUJOCO_GL="${MUJOCO_GL:-egl}" \
            PYTHONWARNINGS="ignore" \
            "${cmd[@]}" > "${log_file}" 2>&1 &
            pid=$!
            ACTIVE_PIDS+=("${pid}")
            PID_TO_SUITE["${pid}"]="${suite}"
            PID_TO_GPU["${pid}"]="${gpu}"
            echo "  [PID ${pid}] ${suite} on GPU ${gpu} -> ${log_file}"
            queue_index=$((queue_index + 1))
        done

        if (( ${#ACTIVE_PIDS[@]} == 0 )); then
            break
        fi

        if wait -n -p finished_pid "${ACTIVE_PIDS[@]}"; then
            wait_status=0
        else
            wait_status=$?
        fi
        finished_suite="${PID_TO_SUITE[${finished_pid}]}"
        finished_gpu="${PID_TO_GPU[${finished_pid}]}"
        AVAILABLE_GPUS+=("${finished_gpu}")

        if (( wait_status == 0 )); then
            SUITE_RESULTS["${finished_suite}"]="$(extract_success_rate "${RESULTS_DIR}/${finished_suite}.log")"
            echo "  [DONE] ${finished_suite} on GPU ${finished_gpu} -> ${SUITE_RESULTS[${finished_suite}]}"
        else
            SUITE_RESULTS["${finished_suite}"]="FAILED"
            echo "  [FAIL] ${finished_suite} on GPU ${finished_gpu} - see ${RESULTS_DIR}/${finished_suite}.log"
            overall_ok=false
        fi
        SUITE_TIMES["${finished_suite}"]=$((SECONDS - SUITE_STARTS[${finished_suite}]))

        local -a remaining_pids=()
        for pid in "${ACTIVE_PIDS[@]}"; do
            if [[ "${pid}" != "${finished_pid}" ]]; then
                remaining_pids+=("${pid}")
            fi
        done
        ACTIVE_PIDS=("${remaining_pids[@]}")
    done

    trap - INT TERM
    EVAL_TOTAL_ELAPSED=$((SECONDS - total_start))
    [[ "${overall_ok}" == true ]]
}


run_suites_mode() {
    HOST="${HOST:-127.0.0.1}"
    PORT="${PORT:-8001}"
    TRIALS="${TRIALS:-50}"
    RECORD="${RECORD:-failure}"
    SEED="${SEED:-42}"
    CLIENT_GPU="${CLIENT_GPU:-0}"
    START_FROM="${START_FROM:-}"

    local timestamp
    timestamp="$(date +%Y%m%d_%H%M%S)"
    RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/results/libero/suites_${timestamp}}"
    SUMMARY_FILE="${RESULTS_DIR}/summary.txt"

    resolve_suites "${TASK_SUITES:-}"
    if [[ -n "${START_FROM}" ]]; then
        local filtered=()
        local skip=true
        local suite
        for suite in "${SUITES[@]}"; do
            if [[ "${suite}" == "${START_FROM}" ]]; then
                skip=false
            fi
            if ! ${skip}; then
                filtered+=("${suite}")
            fi
        done
        SUITES=("${filtered[@]}")
        if [[ ${#SUITES[@]} -eq 0 ]]; then
            echo "[ERROR] START_FROM='${START_FROM}' not found in suites: ${DEFAULT_SUITES[*]}"
            exit 1
        fi
    fi

    mkdir -p "${RESULTS_DIR}"
    build_task_id_args

    echo ""
    echo -e "${BOLD}==========================================================${RESET}"
    echo -e "${BOLD}LIBERO Evaluation - $(date +%Y-%m-%d\ %H:%M)${RESET}"
    echo -e "${BOLD}==========================================================${RESET}"
    echo -e "  Suites      : ${CYAN}${SUITES[*]}${RESET}"
    echo -e "  Server      : ${CYAN}${HOST}:${PORT}${RESET}"
    echo -e "  Trials/task : ${CYAN}${TRIALS}${RESET}"
    echo -e "  Eval mode   : ${CYAN}sequential${RESET}"
    echo -e "  Eval GPU    : ${CYAN}${CLIENT_GPU}${RESET}"
    echo -e "  Results dir : ${DIM}${RESULTS_DIR}${RESET}"
    echo ""

    local overall_ok=true
    if ! evaluate_suites_sequential "${CLIENT_GPU}" "$@"; then
        overall_ok=false
    fi

    print_suite_summary "${SUMMARY_FILE}" "${EVAL_TOTAL_ELAPSED}" "${overall_ok}"
}


run_parallel_mode() {
    CONFIG="${CONFIG:-stage3_finetuning_libero}"
    CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
    PORT="${PORT:-8001}"
    TRIALS="${TRIALS:-10}"
    RECORD="${RECORD:-failure}"
    SEED="${SEED:-7}"
    RUN_NAME="${RUN_NAME:-libero_eval}"
    GPU_LIST="${GPU_LIST:-0 1 2 3}"

    if [[ -n "${CHECKPOINT_DIR}" ]]; then
        HOST="127.0.0.1"
        SERVER_GPU="${SERVER_GPU:-0}"
    else
        HOST="${HOST:-127.0.0.1}"
    fi

    local timestamp
    timestamp="$(date -u +%Y%m%d_%H%M%S)"
    RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/results/libero/parallel_${RUN_NAME}_${timestamp}}"
    SUMMARY_FILE="${RESULTS_DIR}/summary.txt"

    resolve_suites "${TASK_SUITES:-}"
    mkdir -p "${RESULTS_DIR}" "${RESULTS_DIR}/videos"
    build_task_id_args

    if [[ -n "${CHECKPOINT_DIR}" ]]; then
        ensure_gpu_list_excludes "${SERVER_GPU}" "${GPU_LIST}" "GPU_LIST"
    fi

    if [[ -n "${CHECKPOINT_DIR}" ]]; then
        trap cleanup_server EXIT
        start_local_server "${CHECKPOINT_DIR}" "${RESULTS_DIR}/server.log" "Starting local LIBERO policy server for parallel evaluation"
    fi

    echo "Starting LIBERO parallel evaluation"
    if [[ -n "${CHECKPOINT_DIR}" ]]; then
        echo "  Server: local checkpoint ${CHECKPOINT_DIR}"
        echo "  Policy GPU: ${SERVER_GPU}"
    else
        echo "  Server: existing server at ${HOST}:${PORT}"
    fi
    echo "  Port: ${PORT}"
    echo "  Num Trials Per Task: ${TRIALS}"
    echo "  Task Suites: ${SUITES[*]}"
    echo "  GPUs: ${GPU_LIST}"
    echo "  Results Dir: ${RESULTS_DIR}"
    echo ""

    local overall_ok=true
    if ! evaluate_suites_parallel "${GPU_LIST}" "$@"; then
        overall_ok=false
    fi

    if [[ -n "${CHECKPOINT_DIR}" ]]; then
        cleanup_server
        trap - EXIT
    fi

    print_suite_summary "${SUMMARY_FILE}" "${EVAL_TOTAL_ELAPSED}" "${overall_ok}"
}


cleanup_server() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[INFO] Stopping policy server (PID=${SERVER_PID})..."
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
        unset SERVER_PID
    fi
}


wait_for_server() {
    local port="$1"
    local server_pid="${2:-}"
    local log_file="${3:-}"
    local max_wait=600
    local elapsed=0
    echo "[INFO] Waiting for server on port ${port} to become ready..."
    while ! python3 - "$port" <<'PY' 2>/dev/null
import base64
import os
import socket
import sys

port = int(sys.argv[1])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(2)
try:
    s.connect(("127.0.0.1", port))
    key = base64.b64encode(os.urandom(16)).decode()
    request = (
        f"GET / HTTP/1.1\r\n"
        f"Host: 127.0.0.1:{port}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        "\r\n"
    ).encode()
    s.sendall(request)
    response = s.recv(1024)
    status = response.splitlines()[0] if response else b""
    if not (status.startswith(b"HTTP/1.1 101") or status.startswith(b"HTTP/1.0 101")):
        raise RuntimeError(f"unexpected response: {status!r}")
except Exception:
    sys.exit(1)
finally:
    s.close()
sys.exit(0)
PY
    do
        if [[ -n "${server_pid}" ]] && ! kill -0 "${server_pid}" >/dev/null 2>&1; then
            echo "[ERROR] Server process exited before becoming ready."
            if [[ -n "${log_file}" && -f "${log_file}" ]]; then
                echo "[INFO] Last lines from ${log_file}:"
                tail -n 40 "${log_file}" || true
            fi
            exit 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        if [ "${elapsed}" -ge "${max_wait}" ]; then
            echo "[ERROR] Server did not start within ${max_wait}s. Aborting."
            exit 1
        fi
        echo "[INFO]   ... waited ${elapsed}s"
    done
    sleep 2
    echo "[INFO] Server is ready. (waited ~${elapsed}s)"
}


run_benchmark_mode() {
    CONFIG="${CONFIG:-stage3_finetuning_libero}"
    CKPT_BASE="${CKPT_BASE:?'Set CKPT_BASE to your checkpoint base directory'}"
    CKPT_STEPS_STR="${CKPT_STEPS_STR:-${CKPT_STEPS:-10000 20000 30000 40000 50000}}"
    BENCHMARK_MODE="${BENCHMARK_MODE:-serial}"
    PORT="${PORT:-8001}"
    TRIALS="${TRIALS:-50}"
    RECORD="${RECORD:-failure}"
    SEED="${SEED:-7}"
    SERVER_GPU="${SERVER_GPU:-0}"
    CLIENT_GPU="${CLIENT_GPU:-1}"
    GPU_LIST="${GPU_LIST:-0 1 2 3}"

    case "${BENCHMARK_MODE}" in
        serial|parallel)
            ;;
        *)
            echo "[ERROR] BENCHMARK_MODE must be 'serial' or 'parallel', got: ${BENCHMARK_MODE}"
            exit 1
            ;;
    esac

    local timestamp
    timestamp="$(date +%Y%m%d_%H%M%S)"
    local root_results_dir="${RESULTS_DIR:-${PROJECT_ROOT}/results/libero/benchmark_${timestamp}}"
    RESULTS_DIR="${root_results_dir}"
    mkdir -p "${root_results_dir}"

    SUMMARY_CSV="${root_results_dir}/summary.csv"
    SUMMARY_TXT="${root_results_dir}/summary.txt"
    resolve_suites "${TASK_SUITES:-${SUITES:-}}"
    read -r -a CKPT_STEPS_ARRAY <<< "${CKPT_STEPS_STR}"
    build_task_id_args

    if [[ "${BENCHMARK_MODE}" == "parallel" ]]; then
        ensure_gpu_list_excludes "${SERVER_GPU}" "${GPU_LIST}" "GPU_LIST"
    else
        ensure_distinct_gpu_pair "SERVER_GPU" "${SERVER_GPU}" "CLIENT_GPU" "${CLIENT_GPU}"
    fi

    trap cleanup_server EXIT

    echo "checkpoint,suite,success_rate" > "${SUMMARY_CSV}"
    declare -A RESULTS=()

    echo ""
    echo "================================================================"
    echo " LIBERO Full Benchmark"
    echo " Checkpoints : ${CKPT_STEPS_ARRAY[*]}"
    echo " Suites      : ${SUITES[*]}"
    echo " Mode        : ${BENCHMARK_MODE}"
    echo " Trials/task : ${TRIALS}"
    if [[ "${BENCHMARK_MODE}" == "parallel" ]]; then
        echo " Eval GPUs   : ${GPU_LIST}"
    else
        echo " Eval GPU    : ${CLIENT_GPU}"
    fi
    echo " Results dir : ${root_results_dir}"
    echo "================================================================"
    echo ""

    local step ckpt_dir suite rate val row_sum row_cnt row_avg col_sum col_cnt col_avg grand_sum grand_cnt grand_avg
    for step in "${CKPT_STEPS_ARRAY[@]}"; do
        ckpt_dir="${CKPT_BASE}/${step}"
        if [ ! -d "${ckpt_dir}" ]; then
            echo "[WARN] Checkpoint dir not found: ${ckpt_dir}, skipping."
            continue
        fi

        echo ""
        echo "================================================================"
        echo " Checkpoint: ${step}"
        echo "================================================================"

        cleanup_server

        local step_results_dir="${root_results_dir}/${step}"
        RESULTS_DIR="${step_results_dir}"
        mkdir -p "${step_results_dir}" "${step_results_dir}/videos"

        start_local_server "${ckpt_dir}" "${step_results_dir}/server.log" "Starting policy server for checkpoint ${step}"
        HOST="127.0.0.1"

        local step_ok=true
        if [[ "${BENCHMARK_MODE}" == "parallel" ]]; then
            echo "[INFO] Evaluating checkpoint ${step} in parallel mode."
            if ! evaluate_suites_parallel "${GPU_LIST}" "$@"; then
                step_ok=false
            fi
        else
            echo "[INFO] Evaluating checkpoint ${step} in serial mode."
            if ! evaluate_suites_sequential "${CLIENT_GPU}" "$@"; then
                step_ok=false
            fi
        fi

        print_suite_summary "${step_results_dir}/summary.txt" "${EVAL_TOTAL_ELAPSED}" "${step_ok}" false

        for suite in "${SUITE_ORDER[@]}"; do
            rate="${SUITE_RESULTS[${suite}]}"
            RESULTS["${step},${suite}"]="${rate}"
            echo "${step},${suite},${rate}" >> "${SUMMARY_CSV}"
            echo "  => ${suite}: ${rate}"
        done

        cleanup_server
    done

    cleanup_server
    RESULTS_DIR="${root_results_dir}"

    {
        echo ""
        echo "================================================================"
        echo " LIBERO Benchmark Summary - $(date)"
        echo "================================================================"
        echo ""

        printf "%-12s" "Checkpoint"
        for suite in "${SUITES[@]}"; do
            printf "  %-16s" "${suite}"
        done
        printf "  %-10s\n" "Average"

        printf '%0.s-' {1..100}
        echo ""

        for step in "${CKPT_STEPS_ARRAY[@]}"; do
            printf "%-12s" "${step}"
            row_sum=0
            row_cnt=0
            for suite in "${SUITES[@]}"; do
                val="${RESULTS[${step},${suite}]:-N/A}"
                printf "  %-16s" "${val}"
                if [[ "${val}" =~ ^[0-9.]+$ ]]; then
                    row_sum="$(echo "${row_sum} + ${val}" | bc)"
                    row_cnt=$((row_cnt + 1))
                fi
            done
            if [ "${row_cnt}" -gt 0 ]; then
                row_avg="$(echo "scale=4; ${row_sum} / ${row_cnt}" | bc)"
                printf "  %-10s" "${row_avg}"
            else
                printf "  %-10s" "N/A"
            fi
            echo ""
        done

        printf '%0.s-' {1..100}
        echo ""
        printf "%-12s" "Col Avg"
        grand_sum=0
        grand_cnt=0
        for suite in "${SUITES[@]}"; do
            col_sum=0
            col_cnt=0
            for step in "${CKPT_STEPS_ARRAY[@]}"; do
                val="${RESULTS[${step},${suite}]:-N/A}"
                if [[ "${val}" =~ ^[0-9.]+$ ]]; then
                    col_sum="$(echo "${col_sum} + ${val}" | bc)"
                    col_cnt=$((col_cnt + 1))
                fi
            done
            if [ "${col_cnt}" -gt 0 ]; then
                col_avg="$(echo "scale=4; ${col_sum} / ${col_cnt}" | bc)"
                printf "  %-16s" "${col_avg}"
                grand_sum="$(echo "${grand_sum} + ${col_sum}" | bc)"
                grand_cnt=$((grand_cnt + col_cnt))
            else
                printf "  %-16s" "N/A"
            fi
        done
        if [ "${grand_cnt}" -gt 0 ]; then
            grand_avg="$(echo "scale=4; ${grand_sum} / ${grand_cnt}" | bc)"
            printf "  %-10s" "${grand_avg}"
        else
            printf "  %-10s" "N/A"
        fi
        echo ""
        echo ""
        echo "================================================================"
        echo " CSV: ${SUMMARY_CSV}"
        echo " Logs: ${root_results_dir}/"
        echo "================================================================"
    } | tee "${SUMMARY_TXT}"
}


run_serial_mode() {
    CONFIG="${CONFIG:-stage3_finetuning_libero}"
    CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
    PORT="${PORT:-8001}"
    TRIALS="${TRIALS:-50}"
    RECORD="${RECORD:-failure}"
    SEED="${SEED:-42}"
    SERVER_GPU="${SERVER_GPU:-0}"
    CLIENT_GPU="${CLIENT_GPU:-1}"
    HOST="127.0.0.1"

    ensure_distinct_gpu_pair "SERVER_GPU" "${SERVER_GPU}" "CLIENT_GPU" "${CLIENT_GPU}"

    local timestamp
    timestamp="$(date +%Y%m%d_%H%M%S)"
    RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/results/libero/serial_${timestamp}}"
    mkdir -p "${RESULTS_DIR}"

    trap cleanup_server EXIT
    start_local_server "${CHECKPOINT_DIR}" "${RESULTS_DIR}/server.log" "Starting local LIBERO policy server"
    run_suites_mode "$@"
}


DEBUG_MODE=false
MODE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --debug)
            DEBUG_MODE=true
            shift
            ;;
        --)
            shift
            break
            ;;
        serial|suites|parallel|benchmark|interactive)
            MODE="$1"
            shift
            break
            ;;
        -*)
            echo "[ERROR] Unknown global flag: $1"
            usage
            exit 1
            ;;
        *)
            MODE="$1"
            shift
            break
            ;;
    esac
done

ensure_prereqs
apply_debug_defaults

BOLD="\033[1m"
DIM="\033[2m"
GREEN="\033[32m"
RED="\033[31m"
CYAN="\033[36m"
YELLOW="\033[33m"
RESET="\033[0m"

case "${MODE:-interactive}" in
    interactive)
        run_interactive_mode "$@"
        ;;
    serial)
        run_serial_mode "$@"
        ;;
    suites)
        run_suites_mode "$@"
        ;;
    parallel)
        run_parallel_mode "$@"
        ;;
    benchmark)
        run_benchmark_mode "$@"
        ;;
    *)
        echo "[ERROR] Unknown mode: ${MODE}"
        usage
        exit 1
        ;;
esac
