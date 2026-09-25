#!/bin/bash
#SBATCH --job-name=np128_vs_kv
#SBATCH --account=torch_pr_111_general
#SBATCH --partition=l40s_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ll5914/logs/nopred_128_vs_kvcache_%j.out
#SBATCH --error=/scratch/ll5914/logs/nopred_128_vs_kvcache_%j.err

set -euo pipefail
export TMPDIR=/scratch/ll5914/tmp
mkdir -p "$TMPDIR" /scratch/ll5914/logs

PYTHON="/scratch/ll5914/conda_envs/SVD/bin/python"
PROJECT_ROOT="/home/ll5914/Jepa_yifan/JEPA_ARVR"
VJEPA_ROOT="/home/ll5914/ARVR_Video/vjepa2"
export VJEPA_ROOT PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/scripts:${VJEPA_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT="/scratch/ll5914/experiments/nopred_128_vs_stream_kvcache_10clips"
mkdir -p "$OUT"

echo "Start $(date) job=${SLURM_JOB_ID:-manual} partition=l40s_public"
nvidia-smi -L || true

cd "$PROJECT_ROOT"
"$PYTHON" scripts/eval_nopred_128_vs_stream_kvcache.py \
  --out-dir "$OUT" \
  --copy-json /home/ll5914/Jepa_yifan/nopred_128_vs_stream_kvcache_10clips.json \
  --max-samples 10 \
  --unique-videos

echo "Done $(date)"
cat "$OUT/metrics.json"
