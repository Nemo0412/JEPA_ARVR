#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__b19_scalant_param_count
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
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
RULSTM_DEPS=${RULSTM_DEPS:-${PROJECT_ROOT}/rulstm_deps}
export PROJECT_ROOT CODE_ROOT DEPS RULSTM_DEPS

scalant_exec gpu \
  /bin/bash -c '
    set -euo pipefail
    export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2
    export XDG_CACHE_HOME="${PROJECT_ROOT}/outputs/b19_scalant/cache/xdg"
    export TRITON_CACHE_DIR="${PROJECT_ROOT}/outputs/b19_scalant/cache/triton"
    export CUDA_CACHE_PATH="${PROJECT_ROOT}/outputs/b19_scalant/cache/cuda"
    mkdir -p "${XDG_CACHE_HOME}" "${TRITON_CACHE_DIR}" "${CUDA_CACHE_PATH}"
    source "${CODE_ROOT}/scripts/egtea/scalant_runtime.sh"
    scalant_activate
    cd "${CODE_ROOT}"
    export PYTHONPATH="${RULSTM_DEPS}:${DEPS}:${CODE_ROOT}:${PYTHONPATH:-}"
    python scripts/egtea/count_scalant_rulstm_params.py
  '
