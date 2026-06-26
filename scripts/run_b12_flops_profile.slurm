#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__b12_flops_profile
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=04:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/b12_flops_profile_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/b12_flops_profile_%j.err

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
export SHARED_PROJECT_ROOT="${SHARED_PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${SHARED_PROJECT_ROOT}/overlay-15GB-500K.ext3}"
export ENV_NAME="${ENV_NAME:-EgoLifeExp}"
export OUT_DIR="${OUT_DIR:-${SHARED_PROJECT_ROOT}/outputs/b12_flops_profile}"
export TARGET="${TARGET:-all}"
export WARMUP="${WARMUP:-1}"

export HF_HOME="${HF_HOME:-/path/to/.huggingface}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

payload=""
read -r -d '' payload <<'BASH' || true
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/vjepa2:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"
python scripts/profile_b12_flops.py --out-dir "${OUT_DIR}" --target "${TARGET}" --warmup "${WARMUP}"
BASH

singularity exec --nv \
  --overlay "${OVERLAY}:ro" \
  "${SIF}" \
  /bin/bash -lc "${payload}"
