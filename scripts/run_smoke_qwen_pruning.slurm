#!/bin/bash
#SBATCH --job-name=qwen_prune_smoke
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=00:25:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/qwen_prune_smoke_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/qwen_prune_smoke_%j.err

# Smoke test for the Qwen2.5-VL video-token pruner (scripts/smoke_qwen_pruning.py).
# PROJECT_ROOT points at the vlm-pruning-compare worktree so the patched modules are imported.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
SHARED_PROJECT_ROOT="${SHARED_PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${SHARED_PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"

export HF_HOME="${HF_HOME:-/path/to/.huggingface}"
export PROJECT_ROOT ENV_NAME HF_HOME

payload=""
read -r -d '' payload <<'BASH' || true
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/vjepa2:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"
python scripts/smoke_qwen_pruning.py
BASH

singularity exec --nv \
  --overlay "${OVERLAY}:ro" \
  "${SIF}" \
  /bin/bash -lc "${payload}"
