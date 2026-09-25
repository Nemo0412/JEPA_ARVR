#!/bin/bash
#SBATCH --job-name=probe_trope
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256GB
#SBATCH --time=01:50:00
#SBATCH --output=/scratch/ll5914/logs/probe_temporal_rope_%j.out
#SBATCH --error=/scratch/ll5914/logs/probe_temporal_rope_%j.err

set -euo pipefail
export TMPDIR=/scratch/ll5914/tmp
mkdir -p "$TMPDIR" /scratch/ll5914/logs

PYTHON="/scratch/ll5914/conda_envs/SVD/bin/python"
PROJECT_ROOT="/home/ll5914/Jepa_yifan/JEPA_ARVR"
VJEPA_ROOT="/home/ll5914/ARVR_Video/vjepa2"
export VJEPA_ROOT PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/scripts:${VJEPA_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

OUT="/scratch/ll5914/experiments/probe_temporal_rope_abs_frame_128kv"
mkdir -p "$OUT"

echo "========================================================"
echo "Probe 1D temporal RoPE (abs Frame Index) vs 128 KV baseline"
echo "Job=${SLURM_JOB_ID:-manual}  $(date)  host=$(hostname)"
echo "Finetune probe+heads with RoPE; encoder frozen"
echo "========================================================"
nvidia-smi -L || true

cd "$PROJECT_ROOT"
"$PYTHON" -m py_compile \
  app/hdepic_lora_action_anticipation/stream_kvcache_attn_prune.py \
  scripts/finetune_eval_probe_temporal_rope_abs_frame.py

"$PYTHON" scripts/finetune_eval_probe_temporal_rope_abs_frame.py \
  --out-dir "$OUT" \
  --copy-json /home/ll5914/Jepa_yifan/probe_temporal_rope_abs_frame_128kv.json \
  --copy-png /home/ll5914/Jepa_yifan/probe_temporal_rope_abs_frame_128kv.png \
  --max-train 108 \
  --max-val 108 \
  --train-clips-per-video 4 \
  --val-clips-per-video 4 \
  --epochs 8 \
  --lr 1e-4

echo "Done $(date)"
"$PYTHON" - <<'PY'
import json
p="/scratch/ll5914/experiments/probe_temporal_rope_abs_frame_128kv/metrics.json"
d=json.load(open(p))
print("method", d.get("method"))
print("table", json.dumps(d["accuracy"]["table_action_top5"], indent=2))
print("delta_pp", json.dumps(d["accuracy"]["delta_pp"], indent=2))
PY
ls -lh "$OUT"/metrics.json "$OUT"/probe_temporal_rope_abs_frame_128kv.png 2>/dev/null || true
