#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__test_interframe_pose_matrix
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=00:30:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/test_interframe_pose_matrix_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/test_interframe_pose_matrix_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"

mkdir -p "${PROJECT_ROOT}/logs"
export PROJECT_ROOT ENV_NAME

payload=$(cat <<'BASH'
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/vjepa2:${PYTHONPATH:-}"

python "${PROJECT_ROOT}/scripts/test_interframe_pose_matrix.py"
echo "[test] interframe pose matrix unit tests passed"
BASH
)

singularity exec \
    --overlay "${OVERLAY}:ro" \
    "${SIF}" \
    /bin/bash -lc "${payload}"
