#!/bin/bash
#SBATCH --job-name=eval64_pred0
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/ll5914/logs/eval64_pred0_prune_%j.out
#SBATCH --error=/scratch/ll5914/logs/eval64_pred0_prune_%j.err

set -euo pipefail

# Lean submit: read videos from scratch (no /dev/shm stage).

PYTHON="/scratch/ll5914/conda_envs/SVD/bin/python"
PROJECT_ROOT="/home/ll5914/Jepa_yifan/JEPA_ARVR"
VJEPA_ROOT="/home/ll5914/ARVR_Video/vjepa2"
export VJEPA_ROOT
export PYTHONPATH="${PROJECT_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"

VIDEO_ROOT="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_videos"
ANN_DIR="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/stream_half_split"
CHECKPOINT="/scratch/ll5914/models/vjepa2/vitl.pt"
VIDEO_BEST_DIR="/scratch/ll5914/experiments/p01_video_pred_joint_heads_clip/action_anticipation_frozen/p01-video-pred-joint-heads-vitl16-256-10ep"
STREAM_CKPT="/scratch/ll5914/experiments/p01_stream_mtp_2_4_6/best.pt"
OUT_DIR="/scratch/ll5914/experiments/p01_eval_64slot_pred0_prune_vs_last16"
MAX_SAMPLES="${MAX_SAMPLES:-400}"
STRIDE="${STRIDE:-8}"

mkdir -p /scratch/ll5914/logs "${OUT_DIR}"

echo "========================================================"
echo "eval 64-slot pred0 prune vs last16 (no stage)"
echo "Job=${SLURM_JOB_ID:-manual} MAX_SAMPLES=${MAX_SAMPLES} STRIDE=${STRIDE}"
echo "OUT=${OUT_DIR}"
echo "Start: $(date)"
echo "========================================================"
nvidia-smi -L || true

cd "${PROJECT_ROOT}"
"${PYTHON}" scripts/eval_64slot_pred0_prune_vs_last16.py \
  --val-csv "${ANN_DIR}/HD_EPIC_val_stream_mtp.csv" \
  --train-csv "${ANN_DIR}/HD_EPIC_train_stream_mtp.csv" \
  --video-root "${VIDEO_ROOT}" \
  --checkpoint "${CHECKPOINT}" \
  --encoder-lora "${VIDEO_BEST_DIR}/encoder_lora_best.pt" \
  --predictor-lora "${VIDEO_BEST_DIR}/predictor_lora_best.pt" \
  --stream-ckpt "${STREAM_CKPT}" \
  --out-dir "${OUT_DIR}" \
  --modes "last16,pred0_topk,enc_topk" \
  --keep-count 4096 \
  --context-sec 16 \
  --require-ctx-sec 10 \
  --horizons-sec "2,4,6" \
  --anticipation-sec 2.0 \
  --max-samples "${MAX_SAMPLES}" \
  --stride "${STRIDE}" \
  --num-workers 2 \
  --log-every 10

echo "Done: $(date)"
echo "Metrics: ${OUT_DIR}/metrics.json"
python3 - <<'PY'
import json
from pathlib import Path
p=Path("/scratch/ll5914/experiments/p01_eval_64slot_pred0_prune_vs_last16/metrics.json")
d=json.loads(p.read_text())
print("n_eval", d["config"]["n_eval"])
print("metrics_pct", json.dumps(d["metrics_pct"], indent=2))
print("deltas_pp", json.dumps(d["deltas_pp"], indent=2))
PY
