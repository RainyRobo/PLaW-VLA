#!/usr/bin/env bash
# Shared helpers for the three training stages. Source this file.

_TRAIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

train_python() {
  if [[ -x "${_TRAIN_ROOT}/.venv/bin/python" ]]; then
    printf '%s\n' "${_TRAIN_ROOT}/.venv/bin/python"
  else
    printf '%s\n' python
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

require_divisible_batch() {
  local config_name="$1"
  local num_gpus="$2"
  local batch_size
  batch_size="$("$(train_python)" -c 'import openpi.training.config as c, sys; print(c.get_config(sys.argv[1]).batch_size)' "${config_name}")"
  if (( num_gpus < 1 )) || (( batch_size % num_gpus != 0 )); then
    echo "Config ${config_name} uses batch_size=${batch_size}, which cannot be split across ${num_gpus} GPU(s)." >&2
    echo "Set NUM_GPUS to a divisor of ${batch_size}." >&2
    return 1
  fi
}

download_stage_assets() {
  local stage="$1"
  local python_bin
  python_bin="$(train_python)"
  "${python_bin}" "${_TRAIN_ROOT}/scripts/download_assets.py" --stage "${stage}" --checkpoint "${BASE_CHECKPOINT:-pi05_base}"
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
  if [[ -n "${NUM_GPUS:-}" ]]; then
    printf '%s\n' "${NUM_GPUS}"
    return
  fi
  detected=1
  if command -v nvidia-smi >/dev/null 2>&1; then
    detected="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d '[:space:]')"
  fi
  if [[ -z "${detected}" || "${detected}" -lt 1 ]]; then
    detected=1
  fi
  printf '%s\n' "${detected}"
}
