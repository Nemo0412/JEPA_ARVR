#!/bin/bash
#SBATCH --job-name=vlesa_latency
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/ll5914/logs/vlesa_latency_%j.out
#SBATCH --error=/scratch/ll5914/logs/vlesa_latency_%j.err

set -euo pipefail

PYTHON="/scratch/ll5914/conda_envs/SVD/bin/python"
ROOT="/home/ll5914/Jepa_baseline/vlesa_latency"
OUT_DIR="/scratch/ll5914/experiments/vlesa_latency"
MODEL="${VLESA_MODEL:-hanjianghu/VLESA-Qwen3-VL-2B-Safety-Filter}"

export HF_HOME="${HF_HOME:-/scratch/ll5914/.huggingface}"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TMPDIR=/scratch/ll5914/tmp
export TMP=/scratch/ll5914/tmp
export TEMP=/scratch/ll5914/tmp

mkdir -p /scratch/ll5914/logs "${OUT_DIR}" "${HF_HOME}" /scratch/ll5914/tmp

echo "========================================================"
echo "VLESA Safety Filter Latency (local GPU component)"
echo "Job=${SLURM_JOB_ID:-manual}"
echo "Host=$(hostname)"
echo "Start: $(date)"
echo "Model: ${MODEL}"
echo "========================================================"
nvidia-smi || true

"${PYTHON}" - <<'PY'
import torch
from transformers import Qwen3VLForConditionalGeneration
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("GPU", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
print("Qwen3VL import OK")
PY

echo "[run] Latency @ max_new_tokens=1024 (paper default)..."
"${PYTHON}" "${ROOT}/bench_safety_filter_latency.py" \
  --model "${MODEL}" \
  --attn sdpa \
  --warmup 3 \
  --iters 20 \
  --max-new-tokens 1024 \
  --out "${OUT_DIR}/latency_max1024.json"

echo "[run] Latency @ max_new_tokens=256 (lower-latency profile)..."
"${PYTHON}" "${ROOT}/bench_safety_filter_latency.py" \
  --model "${MODEL}" \
  --attn sdpa \
  --warmup 3 \
  --iters 20 \
  --max-new-tokens 256 \
  --out "${OUT_DIR}/latency_max256.json"

echo "Done: $(date)"
ls -la "${OUT_DIR}"
