#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__pred_attn_var
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/pred_attn_var_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/pred_attn_var_%j.err

# Predictor attention variant visualization.
# MODE=4s_block0  : 4s clips, predictor block-0 (positions IN DISTRIBUTION).
# MODE=1min_block11: 1min clips, predictor block-11 (last layer, OOD positions).
# MODE=4s_prune   : 4s → K further pruning using predictor block-0 signal.
#
# Runtime: 4s_block0 ~100 samples @ 2s/sample = 3 min; 1min_block11 ~2min+pass through 12 blocks.

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP//path/to/VJEPA2-EXP}"
SHARED_PROJECT_ROOT="${SHARED_PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${SHARED_PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"
MODE="${MODE:-4s_block0}"
N_SAMPLE="${N_SAMPLE:-100}"
CHUNK_SIZE="${CHUNK_SIZE:-256}"
KEEP_K="${KEEP_K:-256}"
OUT_DIR="${OUT_DIR:-${SHARED_PROJECT_ROOT}/outputs/pred_attn_${MODE}}"
export PROJECT_ROOT SHARED_PROJECT_ROOT ENV_NAME MODE N_SAMPLE CHUNK_SIZE KEEP_K OUT_DIR

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

payload=""
read -r -d '' payload <<'BASH' || true
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/vjepa2:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"
echo "[config] MODE=${MODE}  N_SAMPLE=${N_SAMPLE}  CHUNK_SIZE=${CHUNK_SIZE}  KEEP_K=${KEEP_K}"
python scripts/analyze_pred_attn_variants.py
BASH

singularity exec --nv --overlay "${OVERLAY}:ro" "${SIF}" /bin/bash -lc "${payload}"
