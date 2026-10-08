#!/bin/bash
# RoboTwin evaluation launcher.
#
# Capabilities:
#   * Multi-task evaluation across GPUs (one shared policy server +
#     N client renderers).
#   * Auto-spawn a local server (--server-gpu) OR talk to a remote one
#     (--server-host / --server-port).
#   * Checkpoint switching via --checkpoint (resolves POLICY_DIR / MODEL_NAME).
#   * Task selection: explicit list OR a preset (all / class1 / class2 /
#     class3, matching ``calc_stat.py:TASK_CLASS``).
#   * Video-saving policy via --video:
#       - none   (default) do not record videos.
#       - failed encode only failed episodes.
#       - all    encode every episode.
#     The client records videos separately from the simulator's recorder.
#
# Usage:
#   bash examples/robotwin/eval.sh \
#       --checkpoint /path/to/checkpoint/step \
#       --tasks all \
#       --server-gpu 0 \
#       --client-gpus 0,1
#
#   # Class-2 (bimanual) only, keep failure videos:
#   bash examples/robotwin/eval.sh \
#       --checkpoint /path/to/checkpoint/step \
#       --tasks class2 \
#       --video failed
#
#   # Specific tasks against an already-running remote server:
#   bash examples/robotwin/eval.sh \
#       --tasks "adjust_bottle,pick_dual_bottles" \
#       --server-host gpu-host --server-port 8001 \
#       --client-gpus 0,1,2,3
#
# All flags:
#   --checkpoint   PATH        Checkpoint dir (sets POLICY_DIR). Required if
#                              --server-gpu is given (we need it to spawn).
#   --ckpt-name    NAME        Label used in the save-root + server (default:
#                              basename of --checkpoint).
#   --tasks        SPEC        all | class1 | class2 | class3 | "a,b,c"
#   --client-gpus  "0,1,..."   GPU pool for client renderers.
#   --server-gpu   GPU         Spawn server on this GPU and tear it down on
#                              exit. Mutually exclusive with --server-host.
#   --server-host  HOST        Use a remote server. Default: localhost if
#                              --server-gpu is also unset.
#   --server-port  PORT        Default: 8001.
#   --task-config  demo_clean  | demo_randomized
#   --seed         INT         Default: 0.
#   --test-num     INT         Episodes per task (default: 100).
#   --video        MODE        none | failed | all (default: none).
#   --save-root    DIR         Where artifacts go (auto-named under
#                              ``results/robotwin/`` if omitted).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---------------- 50-task presets (must mirror calc_stat.py:TASK_CLASS) ------
TASKS_CLASS1="adjust_bottle,beat_block_hammer,click_alarmclock,click_bell,dump_bin_bigbin,grab_roller,lift_pot,move_can_pot,move_pillbottle_pad,move_playingcard_away,move_stapler_pad,open_laptop,open_microwave,place_a2b_left,place_a2b_right,place_bread_basket,place_container_plate,place_empty_cup,place_fan,place_mouse_pad,place_object_scale,place_object_stand,place_phone_stand,place_shoe,press_stapler,rotate_qrcode,shake_bottle,shake_bottle_horizontally,stamp_seal,turn_switch"
TASKS_CLASS2="handover_block,handover_mic,hanging_mug,pick_diverse_bottles,pick_dual_bottles,place_bread_skillet,place_burger_fries,place_can_basket,place_cans_plasticbox,place_dual_shoes,place_object_basket,put_object_cabinet,scan_object,stack_blocks_two,stack_bowls_two"
TASKS_CLASS3="blocks_ranking_rgb,blocks_ranking_size,put_bottles_dustbin,stack_blocks_three,stack_bowls_three"
TASKS_ALL="${TASKS_CLASS1},${TASKS_CLASS2},${TASKS_CLASS3}"

# ---------------- defaults --------------------------------------------------
TASKS_SPEC=""
CHECKPOINT=""
CKPT_NAME=""
CLIENT_GPUS=""
SERVER_GPU=""
SERVER_HOST=""
SERVER_PORT="${POLICY_PORT:-8001}"
TASK_CONFIG="demo_clean"
SEED="0"
TEST_NUM="100"
VIDEO_MODE="none"
SAVE_ROOT=""

# ---------------- parse args ------------------------------------------------
while [[ $# -gt 0 ]]; do
    if [[ "$1" == --* && "$1" != --help && $# -lt 2 ]]; then
        echo "[eval] missing value for $1" >&2
        exit 1
    fi
    case "$1" in
        --tasks)        TASKS_SPEC="$2"; shift 2 ;;
        --checkpoint)   CHECKPOINT="$2"; shift 2 ;;
        --ckpt-name)    CKPT_NAME="$2"; shift 2 ;;
        --client-gpus)  CLIENT_GPUS="$2"; shift 2 ;;
        --server-gpu)   SERVER_GPU="$2"; shift 2 ;;
        --server-host)  SERVER_HOST="$2"; shift 2 ;;
        --server-port)  SERVER_PORT="$2"; shift 2 ;;
        --task-config)  TASK_CONFIG="$2"; shift 2 ;;
        --seed)         SEED="$2"; shift 2 ;;
        --test-num)     TEST_NUM="$2"; shift 2 ;;
        --video)        VIDEO_MODE="$2"; shift 2 ;;
        --save-root)    SAVE_ROOT="$2"; shift 2 ;;
        -h|--help)      grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "[eval] unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ---------------- validate / normalise --------------------------------------
if [[ -z "${TASKS_SPEC}" ]]; then
    echo "[eval] --tasks is required (all | class1 | class2 | class3 | comma-list)" >&2
    exit 1
fi
if [[ -z "${CLIENT_GPUS}" ]]; then
    if [[ -n "${SERVER_GPU}" ]]; then
        CLIENT_GPUS="${SERVER_GPU}"
    else
        echo "[eval] --client-gpus is required when --server-gpu is not given" >&2
        exit 1
    fi
fi
if [[ -n "${SERVER_GPU}" && -n "${SERVER_HOST}" ]]; then
    echo "[eval] --server-gpu and --server-host are mutually exclusive" >&2
    exit 1
fi
if [[ -z "${SERVER_HOST}" ]]; then
    SERVER_HOST="localhost"
fi

case "${VIDEO_MODE}" in
    none|failed|all) ;;
    *) echo "[eval] --video must be one of: none, failed, all (got: ${VIDEO_MODE})" >&2; exit 1 ;;
esac
[[ "${SEED}" =~ ^[0-9]+$ ]] || { echo "[eval] --seed must be nonnegative" >&2; exit 1; }
[[ "${TEST_NUM}" =~ ^[0-9]+$ ]] && (( 10#${TEST_NUM} > 0 )) || {
    echo "[eval] --test-num must be a positive integer" >&2; exit 1;
}
[[ "${SERVER_PORT}" =~ ^[0-9]+$ ]] && (( 10#${SERVER_PORT} >= 1 && 10#${SERVER_PORT} <= 65535 )) || {
    echo "[eval] --server-port must be between 1 and 65535" >&2; exit 1;
}

# Resolve task spec to a concrete comma-separated list.
case "${TASKS_SPEC}" in
    all)    TASKS="${TASKS_ALL}" ;;
    class1) TASKS="${TASKS_CLASS1}" ;;
    class2) TASKS="${TASKS_CLASS2}" ;;
    class3) TASKS="${TASKS_CLASS3}" ;;
    *)      TASKS="${TASKS_SPEC}" ;;
esac

# Resolve checkpoint -> POLICY_DIR / MODEL_NAME.
if [[ -n "${CHECKPOINT}" ]]; then
    if [[ ! -d "${CHECKPOINT}" ]]; then
        echo "[eval] checkpoint directory not found: ${CHECKPOINT}" >&2
        exit 1
    fi
    CHECKPOINT="$(cd "${CHECKPOINT}" && pwd)"
    export POLICY_DIR="${CHECKPOINT}"
    if [[ -z "${CKPT_NAME}" ]]; then
        CKPT_NAME="$(basename "${CHECKPOINT}")"
    fi
    export MODEL_NAME="${CKPT_NAME}"
elif [[ -n "${SERVER_GPU}" ]]; then
    echo "[eval] --checkpoint is required when --server-gpu spawns a fresh server" >&2
    exit 1
fi

# Default save-root: ``results/robotwin/eval_<ckpt-or-run>_<timestamp>``.
if [[ -z "${SAVE_ROOT}" ]]; then
    SAVE_ROOT="${PROJECT_ROOT}/results/robotwin/eval_${CKPT_NAME:-run}_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "${SAVE_ROOT}"
SAVE_ROOT="$(cd "${SAVE_ROOT}" && pwd)"
mkdir -p "${SAVE_ROOT}/logs"

# Pre-flight: when spawning a fresh server, make sure the port is free.
if [[ -n "${SERVER_GPU}" ]] && command -v ss >/dev/null && ss -tlnp 2>/dev/null | grep -q ":${SERVER_PORT} "; then
    echo "[eval] port ${SERVER_PORT} is already in use; clean up an earlier server first:" >&2
    echo "    ps -ef | grep -E 'serve_policy_robotwin|examples/robotwin/main\\.py' | grep -v grep" >&2
    exit 1
fi

IFS=',' read -ra TASK_ARR <<< "${TASKS}"
IFS=',' read -ra GPU_ARR  <<< "${CLIENT_GPUS}"
declare -A SEEN_TASKS SEEN_GPUS
for task in "${TASK_ARR[@]}"; do
    [[ "${task}" =~ ^[a-zA-Z0-9_]+$ && -z "${SEEN_TASKS[${task}]:-}" ]] || {
        echo "[eval] task names must be nonempty, unique, and contain only letters, digits, underscores" >&2; exit 1;
    }
    SEEN_TASKS[${task}]=1
done
for gpu in "${GPU_ARR[@]}"; do
    [[ "${gpu}" =~ ^[0-9]+$ && -z "${SEEN_GPUS[${gpu}]:-}" ]] || {
        echo "[eval] client GPUs must be a nonempty list of unique numeric IDs" >&2; exit 1;
    }
    SEEN_GPUS[${gpu}]=1
done

# ---------------- export video-mode + asset-id passthrough ------------------
export ROBOTWIN_VIDEO_MODE="${VIDEO_MODE}"
# Kept exported only when explicitly set by the caller (per-task auto-routing
# lives in launch_client.sh).
if [[ -n "${ROBOTWIN_ASSET_ID+x}" ]]; then
    export ROBOTWIN_ASSET_ID
fi

echo "============================================================"
echo "[eval] tasks         = ${TASK_ARR[*]}  (${#TASK_ARR[@]} total)"
echo "[eval] client GPUs   = ${GPU_ARR[*]}   (${#GPU_ARR[@]} workers)"
if [[ -n "${SERVER_GPU}" ]]; then
    echo "[eval] server        = local GPU ${SERVER_GPU} (spawned)"
    echo "[eval] checkpoint    = ${POLICY_DIR}  (name: ${MODEL_NAME})"
else
    echo "[eval] server        = ${SERVER_HOST}:${SERVER_PORT} (external)"
fi
echo "[eval] task_config   = ${TASK_CONFIG}  seed=${SEED}  test_num=${TEST_NUM}"
echo "[eval] video mode    = ${VIDEO_MODE}"
echo "[eval] save_root     = ${SAVE_ROOT}"
echo "============================================================"

# ---------------- optional: spawn the policy server -------------------------
SERVER_PID=""
declare -A GPU_PID
stop_process_group() {
    local pid="$1"
    local attempt
    kill -TERM -- "-${pid}" 2>/dev/null || true
    # The launching PID may not have entered its new session yet.
    kill -TERM "${pid}" 2>/dev/null || true
    for attempt in {1..20}; do
        if ! kill -0 -- "-${pid}" 2>/dev/null; then
            return
        fi
        sleep 0.1
    done
    kill -KILL -- "-${pid}" 2>/dev/null || true
}
cleanup() {
    for pid in "${GPU_PID[@]}"; do
        stop_process_group "${pid}"
    done
    for pid in "${GPU_PID[@]}"; do
        wait "${pid}" 2>/dev/null || true
    done
    if [[ -n "${SERVER_PID}" ]]; then
        echo "[eval] stopping server pid=${SERVER_PID}"
        stop_process_group "${SERVER_PID}"
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ -n "${SERVER_GPU}" ]]; then
    SERVER_LOG="${SAVE_ROOT}/logs/server_gpu${SERVER_GPU}.log"
    echo "[eval] starting server on GPU ${SERVER_GPU} -> ${SERVER_LOG}"
    CUDA_VISIBLE_DEVICES="${SERVER_GPU}" \
        POLICY_PORT="${SERVER_PORT}" \
        POLICY_DIR="${POLICY_DIR}" \
        setsid bash "${SCRIPT_DIR}/launch_server.sh" \
        > "${SERVER_LOG}" 2>&1 &
    SERVER_PID=$!
    echo "[eval] server pid=${SERVER_PID}; waiting for readiness (TCP ${SERVER_HOST}:${SERVER_PORT}) ..."
    SERVER_READY=0
    for _ in $(seq 1 150); do  # 150 * 2s = 5 min budget
        if (echo > "/dev/tcp/${SERVER_HOST}/${SERVER_PORT}") 2>/dev/null; then
            echo "[eval] server is up"
            SERVER_READY=1
            break
        fi
        sleep 2
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "[eval] server died before becoming ready; see ${SERVER_LOG}" >&2
            exit 1
        fi
    done
    if (( SERVER_READY == 0 )); then
        echo "[eval] server did not become ready within five minutes; see ${SERVER_LOG}" >&2
        exit 1
    fi
fi

# ---------------- dispatch tasks across client GPUs -------------------------
dispatch() {
    local gpu="$1"
    local task="$2"
    local task_save="${SAVE_ROOT}/${task}"
    local log="${SAVE_ROOT}/logs/client_${task}_gpu${gpu}.log"
    mkdir -p "${task_save}"
    echo "[eval] -> GPU ${gpu}: ${task} (log: ${log})"
    (
        export POLICY_SERVER_HOST="${SERVER_HOST}"
        export POLICY_PORT="${SERVER_PORT}"
        export TASK_CONFIG="${TASK_CONFIG}"
        export SEED="${SEED}"
        export TEST_NUM="${TEST_NUM}"
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export ROBOTWIN_VIDEO_MODE
        export MODEL_NAME
        if [[ -n "${ROBOTWIN_ASSET_ID:-}" ]]; then
            export ROBOTWIN_ASSET_ID
        fi
        exec setsid bash "${SCRIPT_DIR}/launch_client.sh" "${task_save}" "${task}"
    ) > "${log}" 2>&1 &
    GPU_PID[${gpu}]=$!
}

# Initial fill: assign the first few tasks to all available GPUs.
TASK_IDX=0
for gpu in "${GPU_ARR[@]}"; do
    if (( TASK_IDX < ${#TASK_ARR[@]} )); then
        dispatch "${gpu}" "${TASK_ARR[$TASK_IDX]}"
        TASK_IDX=$((TASK_IDX + 1))
    fi
done

# Polling loop: as each GPU's child exits, dispatch the next pending task to it.
TOTAL_OK=0
TOTAL_FAIL=0
while (( TOTAL_OK + TOTAL_FAIL < ${#TASK_ARR[@]} )); do
    sleep 1
    for gpu in "${GPU_ARR[@]}"; do
        pid="${GPU_PID[${gpu}]:-}"
        if [[ -z "${pid}" ]]; then continue; fi
        if ! kill -0 "${pid}" 2>/dev/null; then
            if wait "${pid}"; then
                TOTAL_OK=$((TOTAL_OK + 1))
                echo "[eval] GPU ${gpu} child pid=${pid} OK"
            else
                TOTAL_FAIL=$((TOTAL_FAIL + 1))
                echo "[eval] GPU ${gpu} child pid=${pid} FAILED" >&2
            fi
            stop_process_group "${pid}"
            unset 'GPU_PID[${gpu}]'
            if (( TASK_IDX < ${#TASK_ARR[@]} )); then
                dispatch "${gpu}" "${TASK_ARR[$TASK_IDX]}"
                TASK_IDX=$((TASK_IDX + 1))
            fi
        fi
    done
done

echo "============================================================"
echo "[eval] DONE  ok=${TOTAL_OK}  fail=${TOTAL_FAIL}"
echo "[eval] aggregate success rates:"
echo "    python examples/robotwin/calc_stat.py ${SAVE_ROOT}"
echo "============================================================"
if (( TOTAL_FAIL > 0 )); then
    exit 1
fi
