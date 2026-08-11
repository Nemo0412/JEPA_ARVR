#!/bin/bash
#SBATCH --job-name=p01_mtp_a2_vid
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256GB
#SBATCH --time=01:50:00
#SBATCH --output=/scratch/ll5914/logs/p01_stream_mtp_a2_ctx_video_%j.out
#SBATCH --error=/scratch/ll5914/logs/p01_stream_mtp_a2_ctx_video_%j.err

set -euo pipefail

# A2: val-only context-bucketed metrics for video-only stream best (25.88%).
# Writes action_top5@{2,4,6}s|ctx{4,6,8,10} into val_only_metrics.json

PYTHON="/scratch/ll5914/conda_envs/SVD/bin/python"
PROJECT_ROOT="/home/ll5914/Jepa_yifan/JEPA_ARVR"
VJEPA_ROOT="/home/ll5914/ARVR_Video/vjepa2"
export VJEPA_ROOT
export PYTHONPATH="${PROJECT_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

VIDEO_ROOT_REMOTE="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_videos"
ANN_DIR="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/stream_half_split"
CHECKPOINT="/scratch/ll5914/models/vjepa2/vitl.pt"
VIDEO_BEST_DIR="/scratch/ll5914/experiments/p01_video_pred_joint_heads_clip/action_anticipation_frozen/p01-video-pred-joint-heads-vitl16-256-10ep"
ENCODER_LORA="${VIDEO_BEST_DIR}/encoder_lora_best.pt"
PREDICTOR_LORA="${VIDEO_BEST_DIR}/predictor_lora_best.pt"
INIT_CKPT="/scratch/ll5914/experiments/p01_stream_mtp_2_4_6/best.pt"
OUT_DIR="/scratch/ll5914/experiments/p01_stream_mtp_a2_ctx_bucket_video"
mkdir -p /scratch/ll5914/logs "${OUT_DIR}"

STAGE_ROOT="${STAGE_ROOT:-/dev/shm/${USER}/hdepic_a2_vid_${SLURM_JOB_ID:-manual}}"
cleanup() { rm -rf "${STAGE_ROOT}" 2>/dev/null || true; }
trap cleanup EXIT

echo "A2 video ctx-bucket val | Job=${SLURM_JOB_ID:-manual} | $(date)"
mkdir -p "${STAGE_ROOT}/P01"
rsync -aL "${VIDEO_ROOT_REMOTE}/P01/" "${STAGE_ROOT}/P01/"

cd "${PROJECT_ROOT}"
"${PYTHON}" -m app.hdepic_lora_action_anticipation.train_stream_mtp \
  --train-csv "${ANN_DIR}/HD_EPIC_train_stream_mtp.csv" \
  --val-csv "${ANN_DIR}/HD_EPIC_val_stream_mtp.csv" \
  --video-root "${STAGE_ROOT}" \
  --checkpoint "${CHECKPOINT}" \
  --encoder-lora "${ENCODER_LORA}" \
  --predictor-lora "${PREDICTOR_LORA}" \
  --init-from-ckpt "${INIT_CKPT}" \
  --out-dir "${OUT_DIR}" \
  --horizons-sec "2,4,6" \
  --loss-weights "1.0,0.7,0.5" \
  --primary-horizon-sec 2.0 \
  --anticipation-sec 2.0 \
  --max-frames 80 \
  --fps 8 \
  --img-size 256 \
  --keep-count 4096 \
  --batch-size 1 \
  --num-workers 8 \
  --epochs 1 \
  --lr 1e-5 \
  --val-only

echo "Done A2 video: $(date)"
python3 - <<'PY'
import json
from pathlib import Path
p=Path("/scratch/ll5914/experiments/p01_stream_mtp_a2_ctx_bucket_video/val_only_metrics.json")
m=json.loads(p.read_text())
print("overall @2s", round(100*m.get("action_top5@2s",0),2))
for ctx in (4,6,8,10):
  k=f"action_top5@2s|ctx{ctx}"
  n=m.get(f"n|{k}", m.get(f"n|action_top5@2s|ctx{ctx}", "?"))
  print(f"  ctx={ctx}s @2s={100*m.get(k,float('nan')):.2f}% n={n}")
PY
