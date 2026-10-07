#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__b19_scalant_s23_selected_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --time=00:45:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
CODE_ROOT=${CODE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}
PROJECT_ROOT=${PROJECT_ROOT:-${CODE_ROOT}}
export PROJECT_ROOT CODE_ROOT
source "${CODE_ROOT}/scripts/egtea/scalant_runtime.sh"
SPLIT=${1:?expected official split 2 or 3}
HORIZON=${2:?expected horizon 2 or 6}
case "$SPLIT" in 2|3) ;; *) printf 'Invalid split: %s\n' "$SPLIT" >&2; exit 2;; esac
case "$HORIZON" in 2|6) ;; *) printf 'Invalid horizon: %s\n' "$HORIZON" >&2; exit 2;; esac
IMAGE=${IMAGE:-/share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif}
OVERLAY=${OVERLAY:-${PROJECT_ROOT}/overlay-15GB-500K.ext3}
DEPS=${DEPS:-${PROJECT_ROOT}/outputs/b19_scalant/deps}
RULSTM_DEPS=${RULSTM_DEPS:-${PROJECT_ROOT}/rulstm_deps}
CHECKPOINT_DIR=${PROJECT_ROOT}/outputs/b19_scalant/train/official_s${SPLIT}_h${HORIZON}
AUDIT=${PROJECT_ROOT}/outputs/b19_scalant/implementation/official_split${SPLIT}_stream_audit.json
OUTPUT_DIR=${PROJECT_ROOT}/outputs/b19_scalant/eval/official_s23_selected_full_heads
export PROJECT_ROOT CODE_ROOT DEPS RULSTM_DEPS SPLIT HORIZON CHECKPOINT_DIR AUDIT OUTPUT_DIR

scalant_exec gpu \
  /bin/bash -c '
    set -euo pipefail
    export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4
    export XDG_CACHE_HOME="${PROJECT_ROOT}/outputs/b19_scalant/cache/xdg"
    export TRITON_CACHE_DIR="${PROJECT_ROOT}/outputs/b19_scalant/cache/triton"
    export CUDA_CACHE_PATH="${PROJECT_ROOT}/outputs/b19_scalant/cache/cuda"
    export TORCH_EXTENSIONS_DIR="${PROJECT_ROOT}/outputs/b19_scalant/cache/torch_extensions"
    mkdir -p "${XDG_CACHE_HOME}" "${TRITON_CACHE_DIR}" "${CUDA_CACHE_PATH}" "${TORCH_EXTENSIONS_DIR}" "${OUTPUT_DIR}"
    source "${CODE_ROOT}/scripts/egtea/scalant_runtime.sh"
    scalant_activate
    cd "${CODE_ROOT}"
    export PYTHONPATH="${RULSTM_DEPS}:${DEPS}:${CODE_ROOT}:${PYTHONPATH:-}"
    python -m app.scalant_baseline.official_s23_selected_eval \
      --split "${SPLIT}" --horizon "${HORIZON}" --checkpoint-dir "${CHECKPOINT_DIR}" \
      --audit "${AUDIT}" --output-dir "${OUTPUT_DIR}"
  '
