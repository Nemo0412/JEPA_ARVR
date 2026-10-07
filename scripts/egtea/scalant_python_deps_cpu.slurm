#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__b19_scalant_pydeps
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4GB
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
CODE_ROOT=${CODE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}
PROJECT_ROOT=${PROJECT_ROOT:-${CODE_ROOT}}
export PROJECT_ROOT CODE_ROOT
source "${CODE_ROOT}/scripts/egtea/scalant_runtime.sh"
DEPS=${DEPS:-${PROJECT_ROOT}/outputs/b19_scalant/deps}
IMAGE=${IMAGE:-/share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif}
OVERLAY=${OVERLAY:-${PROJECT_ROOT}/overlay-15GB-500K.ext3}
export PROJECT_ROOT DEPS

scalant_exec cpu \
  /bin/bash -c '
    set -euo pipefail
    export PYTHONNOUSERSITE=1
    source "${CODE_ROOT}/scripts/egtea/scalant_runtime.sh"
    scalant_activate
    pip install --target "${DEPS}" --no-deps tabulate==0.10.0 termcolor==3.3.0
  '
