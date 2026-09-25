#!/bin/bash
#SBATCH --job-name=np_kv_sweep
#SBATCH --account=torch_pr_111_general
#SBATCH --partition=l40s_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/ll5914/logs/nopred_kvcache_sweep_%j.out
#SBATCH --error=/scratch/ll5914/logs/nopred_kvcache_sweep_%j.err

set -euo pipefail
export TMPDIR=/scratch/ll5914/tmp
mkdir -p "$TMPDIR" /scratch/ll5914/logs

PYTHON="/scratch/ll5914/conda_envs/SVD/bin/python"
PROJECT_ROOT="/home/ll5914/Jepa_yifan/JEPA_ARVR"
VJEPA_ROOT="/home/ll5914/ARVR_Video/vjepa2"
export VJEPA_ROOT PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/scripts:${VJEPA_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT="/scratch/ll5914/experiments/nopred_kvcache_size_sweep"
mkdir -p "$OUT"

echo "Start $(date) job=${SLURM_JOB_ID:-manual} partition=l40s_public"
nvidia-smi -L || true

cd "$PROJECT_ROOT"
"$PYTHON" scripts/eval_nopred_kvcache_size_sweep.py \
  --out-dir "$OUT" \
  --copy-json /home/ll5914/Jepa_yifan/nopred_kvcache_size_sweep.json \
  --copy-png /home/ll5914/Jepa_yifan/nopred_kvcache_size_sweep.png \
  --max-samples 108 \
  --clips-per-video 4 \
  --log-every 5

echo "Done $(date)"
cat "$OUT/metrics.json"
ls -lh "$OUT"/kvcache_size_vs_top5.png /home/ll5914/Jepa_yifan/nopred_kvcache_size_sweep.png
