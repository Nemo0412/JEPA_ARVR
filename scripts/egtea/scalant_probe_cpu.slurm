#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__b19_scalant_probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10GB
#SBATCH --time=00:30:00
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

scalant_exec cpu \
  /bin/bash -c '
    set -euo pipefail
    export PYTHONNOUSERSITE=1
    export OMP_NUM_THREADS=4
    source "${CODE_ROOT}/scripts/egtea/scalant_runtime.sh"
    scalant_activate
    cd "${CODE_ROOT}"
    export PYTHONPATH="${DEPS}:${CODE_ROOT}:${PYTHONPATH:-}"
    python -m app.scalant_baseline.dependency_check
    python -m app.scalant_baseline.probe --encoder GRU --tiny --split 1 --horizon 2 \
      --output "${PROJECT_ROOT}/outputs/b19_scalant/implementation/s1_h2_gru_tiny_probe.json"
  '
