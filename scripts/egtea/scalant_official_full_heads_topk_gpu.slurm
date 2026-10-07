#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__b19_scalant_full_heads_topk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
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
INPUT_DIR=${PROJECT_ROOT}/outputs/b19_scalant/eval/official_s1_full_heads
export PROJECT_ROOT CODE_ROOT DEPS RULSTM_DEPS INPUT_DIR

scalant_exec gpu \
  /bin/bash -c '
    set -euo pipefail
    export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2
    source "${CODE_ROOT}/scripts/egtea/scalant_runtime.sh"
    scalant_activate
    cd "${CODE_ROOT}"
    export PYTHONPATH="${RULSTM_DEPS}:${DEPS}:${CODE_ROOT}:${PYTHONPATH:-}"
    MONITOR="${PROJECT_ROOT}/outputs/b19_scalant/logs/full_heads_topk_${SLURM_JOB_ID}.gpu.csv"
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used \
      --format=csv,noheader,nounits --loop=10 > "${MONITOR}" &
    MONITOR_PID=$!
    trap '\''kill "${MONITOR_PID}" 2>/dev/null || true; wait "${MONITOR_PID}" 2>/dev/null || true'\'' EXIT INT TERM
    python -m app.scalant_baseline.official_stream_full_eval_topk_gpu --input-dir "${INPUT_DIR}"
  '
