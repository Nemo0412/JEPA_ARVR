#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__b19_scalant_deps
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
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
export PROJECT_ROOT DEPS
mkdir -p "${DEPS}"

scalant_exec cpu \
  /bin/bash -c '
    set -euo pipefail
    export PYTHONNOUSERSITE=1
    export MAX_JOBS=4
    export TORCH_CUDA_ARCH_LIST="8.0;8.9"
    source "${CODE_ROOT}/scripts/egtea/scalant_runtime.sh"
    scalant_activate
    python -c "import torch; print(\"torch=\", torch.__version__, \"torch_cuda=\", torch.version.cuda)"
    nvcc --version
    pip install --target "${DEPS}" --upgrade --no-deps nestconfig==0.2.0 fvcore==0.1.5.post20221221 iopath==0.1.10 portalocker yacs==0.1.8 fla-core==0.4.1
    pip install --target "${DEPS}" --no-deps --no-build-isolation mamba-ssm==2.3.0
    cd "${CODE_ROOT}"
    export PYTHONPATH="${DEPS}:${CODE_ROOT}:${PYTHONPATH:-}"
    python -m app.scalant_baseline.dependency_check
  '
