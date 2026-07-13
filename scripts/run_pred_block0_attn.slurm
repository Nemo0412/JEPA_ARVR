#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__pred_block0_attn
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/pred_block0_attn_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/pred_block0_attn_%j.err

# Visualize predictor block-0 attention column-sum vs encoder block-24 (Direction A gate).
# Loads full-ctx cache (61440 tokens/sample, 126 MB/file) -> chunked importance [B,N] bf16.
# Runtime estimate: ~100 samples x (load 126MB + 240 attention chunks) ~ 5-10 min on H100.

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP//path/to/VJEPA2-EXP}"
SHARED_PROJECT_ROOT="${SHARED_PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${SHARED_PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"
N_SAMPLE="${N_SAMPLE:-100}"
CHUNK_SIZE="${CHUNK_SIZE:-256}"
export PROJECT_ROOT SHARED_PROJECT_ROOT ENV_NAME N_SAMPLE CHUNK_SIZE

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

payload=""
read -r -d '' payload <<'BASH' || true
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/vjepa2:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"
echo "[config] N_SAMPLE=${N_SAMPLE}  CHUNK_SIZE=${CHUNK_SIZE}"
echo "[cache] full-ctx $(ls ${SHARED_PROJECT_ROOT}/data/preproc_cache_vjepa/nf480_fps8.0_px256_keep61440_idx/*.pt 2>/dev/null | wc -l) files"
echo "[cache] enc-idx  $(ls ${SHARED_PROJECT_ROOT}/data/preproc_cache_vjepa/nf480_fps8.0_px256_keep4096_idx/*.pt  2>/dev/null | wc -l) files"
python scripts/analyze_pred_block0_attn.py
BASH

singularity exec --nv --overlay "${OVERLAY}:ro" "${SIF}" /bin/bash -lc "${payload}"
