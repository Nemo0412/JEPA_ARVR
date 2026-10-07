#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__b19_scalant_official_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --time=01:45:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
CODE_ROOT=${CODE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}
PROJECT_ROOT=${PROJECT_ROOT:-${CODE_ROOT}}
export PROJECT_ROOT CODE_ROOT
source "${CODE_ROOT}/scripts/egtea/scalant_runtime.sh"
HORIZON=${1:?expected independent horizon 2 or 6}
case "$HORIZON" in 2|6) ;; *) printf 'Invalid horizon: %s\n' "$HORIZON" >&2; exit 2;; esac
EXPECTED_START_EPOCH=${2:?expected checkpoint start epoch}
EPOCHS_THIS_JOB=${3:-1}
SPLIT=${4:-1}
case "$EXPECTED_START_EPOCH" in *[!0-9]*|'') printf 'Invalid expected epoch\n' >&2; exit 2;; esac
case "$EPOCHS_THIS_JOB" in *[!0-9]*|'') printf 'Invalid epoch count\n' >&2; exit 2;; esac
case "$SPLIT" in 1|2|3) ;; *) printf 'Invalid split: %s\n' "$SPLIT" >&2; exit 2;; esac
IMAGE=${IMAGE:-/share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif}
OVERLAY=${OVERLAY:-${PROJECT_ROOT}/overlay-15GB-500K.ext3}
DEPS=${DEPS:-${PROJECT_ROOT}/outputs/b19_scalant/deps}
RULSTM_DEPS=${RULSTM_DEPS:-${PROJECT_ROOT}/rulstm_deps}
OUTPUT_DIR=${PROJECT_ROOT}/outputs/b19_scalant/train/official_s${SPLIT}_h${HORIZON}
export PROJECT_ROOT CODE_ROOT DEPS RULSTM_DEPS HORIZON SPLIT OUTPUT_DIR EXPECTED_START_EPOCH EPOCHS_THIS_JOB

scalant_exec gpu \
  /bin/bash -c '
    set -euo pipefail
    export PYTHONNOUSERSITE=1
    export OMP_NUM_THREADS=4
    export XDG_CACHE_HOME="${PROJECT_ROOT}/outputs/b19_scalant/cache/xdg"
    export TRITON_CACHE_DIR="${PROJECT_ROOT}/outputs/b19_scalant/cache/triton"
    export CUDA_CACHE_PATH="${PROJECT_ROOT}/outputs/b19_scalant/cache/cuda"
    export TORCH_EXTENSIONS_DIR="${PROJECT_ROOT}/outputs/b19_scalant/cache/torch_extensions"
    mkdir -p "${XDG_CACHE_HOME}" "${TRITON_CACHE_DIR}" "${CUDA_CACHE_PATH}" "${TORCH_EXTENSIONS_DIR}" "${OUTPUT_DIR}"
    source "${CODE_ROOT}/scripts/egtea/scalant_runtime.sh"
    scalant_activate
    cd "${CODE_ROOT}"
    export PYTHONPATH="${RULSTM_DEPS}:${DEPS}:${CODE_ROOT}:${PYTHONPATH:-}"
    MONITOR="${PROJECT_ROOT}/outputs/b19_scalant/logs/official_train_s${SPLIT}_h${HORIZON}_${SLURM_JOB_ID}.gpu.csv"
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used \
      --format=csv,noheader,nounits --loop=10 > "${MONITOR}" &
    MONITOR_PID=$!
    trap '\''kill "${MONITOR_PID}" 2>/dev/null || true; wait "${MONITOR_PID}" 2>/dev/null || true'\'' EXIT INT TERM
    python -m app.scalant_baseline.official_stream_train \
      --split "${SPLIT}" --horizon "${HORIZON}" --output-dir "${OUTPUT_DIR}" \
      --epochs-this-job "${EPOCHS_THIS_JOB}" --expected-start-epoch "${EXPECTED_START_EPOCH}"
  '
