#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__qwen_llm_attn
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/qwen_llm_layer0_attn_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/qwen_llm_layer0_attn_%j.err

# Direction A (Qwen variant): compare LLM layer-0 text-after-video -> video attention
# vs vision-tower block-31 col-sum, to check if LLM attention has better spatial bias.
# Sequence: 15 text + 19440 video + 17 text = 19472 total.
# Content-only importance (no M-RoPE) -- see script docstring for rationale.
# Runtime: ~50 samples x 3s/sample ~ 3 min on A100.

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP//path/to/VJEPA2-EXP}"
SHARED_PROJECT_ROOT="${SHARED_PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${SHARED_PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"
N_SAMPLE="${N_SAMPLE:-50}"
OUT_DIR="${OUT_DIR:-${SHARED_PROJECT_ROOT}/outputs/qwen_llm_layer0_attn}"
export PROJECT_ROOT SHARED_PROJECT_ROOT ENV_NAME N_SAMPLE OUT_DIR

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

payload=""
read -r -d '' payload <<'BASH' || true
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/vjepa2:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"
echo "[config] N_SAMPLE=${N_SAMPLE}  OUT_DIR=${OUT_DIR}"
echo "[cache] preproc $(ls ${SHARED_PROJECT_ROOT}/data/preproc_cache_qwen/qwen25vl_nf480_pnf480_fps8.0_px256/*.pt 2>/dev/null | wc -l) files"
python scripts/analyze_qwen_llm_layer0_attn.py
BASH

singularity exec --nv --overlay "${OVERLAY}:ro" "${SIF}" /bin/bash -lc "${payload}"
