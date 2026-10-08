#!/usr/bin/env bash
# Shared helpers for the three training stages. Source this file.

_TRAIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

train_python() {
  if [[ -x "${_TRAIN_ROOT}/.venv/bin/python" ]]; then
    printf '%s\n' "${_TRAIN_ROOT}/.venv/bin/python"
  else
    printf '%s\n' python3
  fi
}

train_torchrun() {
  if [[ -x "${_TRAIN_ROOT}/.venv/bin/torchrun" ]]; then
    printf '%s\n' "${_TRAIN_ROOT}/.venv/bin/torchrun"
  elif command -v torchrun >/dev/null 2>&1; then
    command -v torchrun
  else
    echo "torchrun was not found. Run the installation in README.md first." >&2
    return 1
  fi
}

train_help_requested() {
  local arg
  for arg in "$@"; do
    if [[ "${arg}" == --help || "${arg}" == -h ]]; then return 0; fi
  done
  return 1
}

train_resume_requested() {
  local arg resuming=0
  for arg in "$@"; do
    case "${arg}" in
      --resume) resuming=1 ;;
      --no-resume) resuming=0 ;;
    esac
  done
  (( resuming == 1 ))
}

reject_internal_weight_args() {
  local arg
  for arg in "$@"; do
    case "${arg}" in
      --pytorch-weight-path|--pytorch-weight-path=*|--pytorch_weight_path|--pytorch_weight_path=*|\
      --weight-load-mode|--weight-load-mode=*|--weight_load_mode|--weight_load_mode=*)
        echo "Do not pass ${arg} directly to a stage wrapper." >&2
        echo "Use STAGE1_INIT_WEIGHT, STAGE2_INIT_WEIGHT, STAGE3_INIT_WEIGHT, or STAGE3_BASE_WEIGHT." >&2
        return 1
        ;;
    esac
  done
}

require_divisible_batch() {
  local config_name="$1"
  local num_gpus="$2"
  shift 2
  local batch_size arg next_is_batch=0
  batch_size="$("$(train_python)" -c 'import plawvla.training.config as c, sys; print(c.get_config(sys.argv[1]).batch_size)' "${config_name}")"
  for arg in "$@"; do
    if (( next_is_batch )); then batch_size="${arg}"; next_is_batch=0; fi
    case "${arg}" in
      --batch-size|--batch_size) next_is_batch=1 ;;
      --batch-size=*|--batch_size=*) batch_size="${arg#*=}" ;;
    esac
  done
  (( next_is_batch == 0 )) || { echo "Missing batch size argument." >&2; return 1; }
  [[ "${num_gpus}" =~ ^[1-9][0-9]*$ && "${batch_size}" =~ ^[1-9][0-9]*$ ]] || {
    echo "NUM_GPUS and batch size must be positive integers." >&2; return 1;
  }
  if (( num_gpus < 1 )) || (( batch_size % num_gpus != 0 )); then
    echo "Config ${config_name} uses batch_size=${batch_size}, which cannot be split across ${num_gpus} GPU(s)." >&2
    echo "Set NUM_GPUS to a divisor of ${batch_size}." >&2
    return 1
  fi
}

download_stage_assets() {
  local stage="$1"
  shift
  local python_bin
  python_bin="$(train_python)"
  local extra_args=()
  if [[ "${stage}" == 1 ]]; then
    if [[ -n "${STAGE1_INIT_WEIGHT:-}" ]] || train_resume_requested "$@"; then
      extra_args+=(--skip-base-checkpoint)
    fi
  fi
  "${python_bin}" "${_TRAIN_ROOT}/scripts/setup/download_assets.py" --stage "${stage}" --checkpoint "${BASE_CHECKPOINT:-pi05_base}" "${extra_args[@]}"
}

require_checkpoint_dir() {
  [[ -f "$1/model.safetensors" ]] || {
    echo "Checkpoint step directory must contain model.safetensors: $1" >&2
    return 1
  }
}

latest_checkpoint_dir() {
  local parent="$1"
  local candidate name best="" best_step=-1
  [[ -d "${parent}" ]] || return 1
  for candidate in "${parent}"/*; do
    [[ -d "${candidate}" ]] || continue
    name="$(basename "${candidate}")"
    [[ "${name}" =~ ^[0-9]+$ ]] || continue
    [[ -f "${candidate}/model.safetensors" ]] || continue
    if (( name > best_step )); then
      best_step="${name}"
      best="${candidate}"
    fi
  done
  [[ -n "${best}" ]] || return 1
  printf '%s\n' "${best}"
}

default_num_gpus() {
  local detected
  detected="$("$(train_python)" -c 'import torch; print(torch.cuda.device_count())')"
  if (( detected < 1 )); then
    echo "No visible CUDA devices. Check CUDA_VISIBLE_DEVICES and the PyTorch installation." >&2
    return 1
  fi
  local selected="${NUM_GPUS:-${detected}}"
  [[ "${selected}" =~ ^[1-9][0-9]*$ ]] && (( selected <= detected )) || {
    echo "NUM_GPUS must be a positive integer no greater than the ${detected} visible CUDA device(s)." >&2
    return 1
  }
  printf '%s\n' "${selected}"
}

require_pretraining_data() {
  "$(train_python)" - "$1" <<'PYDATA'
import pathlib, sys
from plawvla.training.config import get_config
config = get_config(sys.argv[1])
for spec in config.data.datasets:
    roots = [spec.repo_id] if isinstance(spec.repo_id, str) else spec.repo_id
    for root in roots:
        path = pathlib.Path(root)
        if not path.is_dir() or not any(path.rglob("meta/info.json")):
            raise SystemExit(f"Missing converted {spec.dataset_type} data at {path}. Follow docs/pretraining.md before training.")
PYDATA
}
