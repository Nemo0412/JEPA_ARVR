#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__b19_scalant_long_profile
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
CODE_ROOT=${CODE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}
PROJECT_ROOT=${PROJECT_ROOT:-${CODE_ROOT}}
export PROJECT_ROOT CODE_ROOT
source "${CODE_ROOT}/scripts/egtea/scalant_runtime.sh"
IMAGE=${IMAGE:-/share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif}
OVERLAY=${OVERLAY:-${PROJECT_ROOT}/overlay-15GB-500K.ext3}
DEPS=${DEPS:-${PROJECT_ROOT}/outputs/b19_scalant/deps}
export PROJECT_ROOT CODE_ROOT DEPS

scalant_exec gpu \
  /bin/bash -c '
    set -euo pipefail
    export PYTHONNOUSERSITE=1
    export OMP_NUM_THREADS=4
    export XDG_CACHE_HOME="${PROJECT_ROOT}/outputs/b19_scalant/cache/xdg"
    export TRITON_CACHE_DIR="${PROJECT_ROOT}/outputs/b19_scalant/cache/triton"
    export CUDA_CACHE_PATH="${PROJECT_ROOT}/outputs/b19_scalant/cache/cuda"
    export TORCH_EXTENSIONS_DIR="${PROJECT_ROOT}/outputs/b19_scalant/cache/torch_extensions"
    mkdir -p "${XDG_CACHE_HOME}" "${TRITON_CACHE_DIR}" "${CUDA_CACHE_PATH}" "${TORCH_EXTENSIONS_DIR}"
    source "${CODE_ROOT}/scripts/egtea/scalant_runtime.sh"
    scalant_activate
    cd "${CODE_ROOT}"
    export PYTHONPATH="${DEPS}:${CODE_ROOT}:${PYTHONPATH:-}"
    MONITOR="${PROJECT_ROOT}/outputs/b19_scalant/logs/long_profile_${SLURM_JOB_ID}.gpu.csv"
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used \
      --format=csv,noheader,nounits --loop=10 > "${MONITOR}" &
    MONITOR_PID=$!
    trap '\''kill "${MONITOR_PID}" 2>/dev/null || true; wait "${MONITOR_PID}" 2>/dev/null || true'\'' EXIT INT TERM
    python -m app.scalant_baseline.stream_profile \
      --lengths 1600,6400,12800 --repeats 3 --minimum-seconds 45 \
      --output "${PROJECT_ROOT}/outputs/b19_scalant/implementation/stream_long_prefix_profile.json"
  '
