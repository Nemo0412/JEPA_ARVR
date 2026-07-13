#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__vjepa_fullctx_flops
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/vjepa_fullctx_flops_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/vjepa_fullctx_flops_%j.err

# Real per-sample FLOPs (FlopCounterMode) for the FULL-CONTEXT (no-prune) 1-min V-JEPA2 method:
# keep ALL 61440 tokens, true positions. The predictor+probe train-step here runs over 61440 ctx
# tokens (vs 4096 in the pruned variants) -> the heavy number this branch is currently training on.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
SHARED_PROJECT_ROOT="${SHARED_PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${SHARED_PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"
export PROJECT_ROOT SHARED_PROJECT_ROOT ENV_NAME
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

payload=""
read -r -d '' payload <<'BASH' || true
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/vjepa2:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"
echo "=== 1min V-JEPA2 method, FULL-CONTEXT no-prune (keep 61440, true positions) ==="
python scripts/profile_b12_vjepa_1min_flops.py --num-frames 480 --keep-count 61440 --position-mode true
BASH

singularity exec --nv --overlay "${OVERLAY}:ro" "${SIF}" /bin/bash -lc "${payload}"
