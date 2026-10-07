#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__b19_scalant_full_heads_audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
CODE_ROOT=${CODE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}
PROJECT_ROOT=${PROJECT_ROOT:-${CODE_ROOT}}
export PROJECT_ROOT CODE_ROOT
source "${CODE_ROOT}/scripts/egtea/scalant_runtime.sh"
MODE=${1:-audit}
case "$MODE" in
  audit) MODULE=official_stream_full_eval_cpu ;;
  verify) MODULE=official_stream_full_eval_verify_cpu ;;
  *) printf 'Expected audit or verify mode: %s\n' "$MODE" >&2; exit 2 ;;
esac
IMAGE=${IMAGE:-/share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif}
OVERLAY=${OVERLAY:-${PROJECT_ROOT}/overlay-15GB-500K.ext3}
DEPS=${DEPS:-${PROJECT_ROOT}/outputs/b19_scalant/deps}
RULSTM_DEPS=${RULSTM_DEPS:-${PROJECT_ROOT}/rulstm_deps}
INPUT_DIR=${PROJECT_ROOT}/outputs/b19_scalant/eval/official_s1_full_heads
export PROJECT_ROOT CODE_ROOT DEPS RULSTM_DEPS INPUT_DIR MODULE

scalant_exec cpu \
  /bin/bash -c '
    set -euo pipefail
    export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4
    source "${CODE_ROOT}/scripts/egtea/scalant_runtime.sh"
    scalant_activate
    cd "${CODE_ROOT}"
    export PYTHONPATH="${RULSTM_DEPS}:${DEPS}:${CODE_ROOT}:${PYTHONPATH:-}"
    python -m "app.scalant_baseline.${MODULE}" --input-dir "${INPUT_DIR}"
  '
