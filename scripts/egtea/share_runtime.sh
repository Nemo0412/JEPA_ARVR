#!/bin/bash
# Shared launch environment. Source inside a compute allocation.
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PROJECT_ROOT="$(realpath "$PROJECT_ROOT")"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT}"
CODE_ROOT="$PROJECT_ROOT"
: "${VJEPA_ROOT:?Set VJEPA_ROOT to the compatible V-JEPA source tree}"
: "${PYTHON:?Set PYTHON to the interpreter in the selected environment}"
[[ -f "$PROJECT_ROOT/app/hdepic_lora_action_anticipation/eval_stream_mtp_kvcache_prune.py" ]] \
 || { echo 'ERROR: PROJECT_ROOT does not contain shared pruning code' >&2; return 2; }
[[ -d "$VJEPA_ROOT/src" ]] \
 || { echo 'ERROR: VJEPA_ROOT must contain src/ (initialize the submodule or supply an external tree)' >&2; return 2; }
B17_MANIFEST_DIR="${B17_MANIFEST_DIR:-$DATA_ROOT/runtime_manifests/b17/task-protected-jepa-oracle-l16-v1}"
SHARED_PROJECT_ROOT="$DATA_ROOT"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
export PROJECT_ROOT CODE_ROOT DATA_ROOT VJEPA_ROOT PYTHON B17_MANIFEST_DIR SHARED_PROJECT_ROOT
export HF_HOME TORCH_HOME PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT:$VJEPA_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export SHARE_GPU="${SHARE_GPU:-0}"
share_exec() (
 if [[ "$SHARE_GPU" == 1 ]]; then
  monitor_dir="${SHARE_MONITOR_DIR:-$DATA_ROOT/outputs/share_monitor}"
  mkdir -p "$monitor_dir"
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits --loop=5 > "$monitor_dir/${SLURM_JOB_ID:-local}-${BASHPID}.gpu.csv" &
  share_monitor_pid=$!
  trap 'kill "$share_monitor_pid" 2>/dev/null || true; wait "$share_monitor_pid" 2>/dev/null || true' EXIT
 fi
 if [[ -n "${SHARE_IMAGE:-}" ]]; then
  local -a options=()
  [[ "${SHARE_GPU:-1}" == 1 ]] && options+=(--nv)
  [[ -n "${SHARE_OVERLAY:-}" ]] && options+=(--overlay "${SHARE_OVERLAY}:ro")
  export SHARE_ENV_SCRIPT="${SHARE_ENV_SCRIPT:-}" SHARE_CONDA_ENV="${SHARE_CONDA_ENV:-}"
  singularity exec "${options[@]}" "$SHARE_IMAGE" /bin/bash -c '
   set -e
   if [[ -n "$SHARE_ENV_SCRIPT" ]]; then source "$SHARE_ENV_SCRIPT"; fi
   if [[ -n "$SHARE_CONDA_ENV" ]]; then conda activate "$SHARE_CONDA_ENV"; fi
   exec "$@"
  ' share-runtime "$@"
 else
  "$@"
 fi
)
