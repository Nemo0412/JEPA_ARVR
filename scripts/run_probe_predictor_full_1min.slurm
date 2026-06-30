#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__prune_pattern
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/prune_pattern_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/prune_pattern_%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
SHARED_PROJECT_ROOT="${SHARED_PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${SHARED_PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"
export PROJECT_ROOT SHARED_PROJECT_ROOT ENV_NAME

payload=""
read -r -d '' payload <<'BASH' || true
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/vjepa2:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"
python scripts/probe_predictor_full_1min.py
BASH

# CPU-only (no --nv): pure cached-tensor I/O + matplotlib, no GPU/model/video.
singularity exec --nv --overlay "${OVERLAY}:ro" "${SIF}" /bin/bash -lc "${payload}"
