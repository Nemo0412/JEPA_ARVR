#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__vjepa_eval_deps
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=01:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/install_eval_deps_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/install_eval_deps_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"

mkdir -p "${PROJECT_ROOT}/logs"

export PROJECT_ROOT SIF OVERLAY ENV_NAME

payload=$(cat <<'BASH'
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}

python -m pip install --upgrade pip
python -m pip install opencv-python-headless
python - <<'PY'
import cv2
print("cv2 version:", cv2.__version__)
PY
BASH
)

singularity exec --fakeroot \
    --overlay "${OVERLAY}:rw" \
    "${SIF}" \
    /bin/bash -lc "${payload}"
