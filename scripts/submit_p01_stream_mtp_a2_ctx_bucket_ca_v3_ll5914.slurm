#!/bin/bash
#SBATCH --job-name=p01_mtp_a2_ca
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256GB
#SBATCH --time=01:50:00
#SBATCH --output=/scratch/ll5914/logs/p01_stream_mtp_a2_ctx_ca_%j.out
#SBATCH --error=/scratch/ll5914/logs/p01_stream_mtp_a2_ctx_ca_%j.err

set -euo pipefail

# A2: val-only context-bucketed metrics for concat+CA v3 best (25.08%).
# Uses v3 prune (encoder_attn) to match how that ckpt was trained.

PYTHON="/scratch/ll5914/conda_envs/SVD/bin/python"
PROJECT_ROOT="/home/ll5914/Jepa_yifan/JEPA_ARVR"
VJEPA_ROOT="/home/ll5914/ARVR_Video/vjepa2"
export VJEPA_ROOT
export PYTHONPATH="${PROJECT_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
export TRI_MODAL_GAZE_MAP_CACHE=256
export GAZE_DECODER_CACHE_READER=1
export GAZE_DECODER_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

VIDEO_ROOT_REMOTE="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_videos"
ANN_DIR="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/stream_half_split"
CHECKPOINT="/scratch/ll5914/models/vjepa2/vitl.pt"

CA_RUN="/scratch/ll5914/experiments/concat_plus_cross_attn_v2/action_anticipation_frozen/concat-plus-ca-v2-L3-keepaux-softgate-vitl16-256-12ep-1xh100"
ENCODER_LORA="${CA_RUN}/encoder_lora_best.pt"
PREDICTOR_LORA="/scratch/ll5914/experiments/p01_stream_mtp_2_4_6/predictor_lora_from_stream_best.pt"
ADAPTER_CKPT="${CA_RUN}/binary_input_adapter_best.pt"
FUSION_CKPT="${CA_RUN}/tri_modal_fusion_best.pt"
V3_BEST="/scratch/ll5914/experiments/p01_stream_mtp_concat_ca_v3_hyb_2_4_6/best.pt"

GAZE_EXTRACT_ROOT="/scratch/ll5914/datasets/HD-EPIC/_gaze_extract"
GAZE_ROOT="/scratch/ll5914/datasets/HD-EPIC/SLAM-and-Gaze"
GAZE_SYNC_ROOT="/scratch/ll5914/datasets/HD-EPIC/HD-EPIC/Videos"
POSE_SLAM_ROOT="/scratch/ll5914/datasets/HD-EPIC/SLAM-and-Gaze/P01/SLAM/multi"
POSE_MAPPING_JSON="${POSE_SLAM_ROOT}/vrs_to_multi_slam.json"

OUT_DIR="/scratch/ll5914/experiments/p01_stream_mtp_a2_ctx_bucket_ca_v3"
TICK_CACHE_DIR="/scratch/ll5914/datasets/HD-EPIC/caches/stream_mtp_concat_ca_ticks_256"
mkdir -p /scratch/ll5914/logs "${OUT_DIR}" "${TICK_CACHE_DIR}"

STAGE_ROOT="${STAGE_ROOT:-/dev/shm/${USER}/hdepic_a2_ca_${SLURM_JOB_ID:-manual}}"
cleanup() { rm -rf "${STAGE_ROOT}" 2>/dev/null || true; }
trap cleanup EXIT

echo "A2 CA v3 ctx-bucket val | Job=${SLURM_JOB_ID:-manual} | $(date)"
mkdir -p "${STAGE_ROOT}/P01"
rsync -aL "${VIDEO_ROOT_REMOTE}/P01/" "${STAGE_ROOT}/P01/"

cd "${PROJECT_ROOT}"
"${PYTHON}" -m app.hdepic_lora_action_anticipation.train_stream_mtp_concat_ca \
  --train-csv "${ANN_DIR}/HD_EPIC_train_stream_mtp.csv" \
  --val-csv "${ANN_DIR}/HD_EPIC_val_stream_mtp.csv" \
  --video-root "${STAGE_ROOT}" \
  --checkpoint "${CHECKPOINT}" \
  --encoder-lora "${ENCODER_LORA}" \
  --predictor-lora "${PREDICTOR_LORA}" \
  --adapter-ckpt "${ADAPTER_CKPT}" \
  --fusion-ckpt "${FUSION_CKPT}" \
  --init-from-ckpt "${V3_BEST}" \
  --out-dir "${OUT_DIR}" \
  --gaze-root "${GAZE_ROOT}" \
  --gaze-extract-root "${GAZE_EXTRACT_ROOT}" \
  --gaze-sync-root "${GAZE_SYNC_ROOT}" \
  --pose-slam-root "${POSE_SLAM_ROOT}" \
  --pose-mapping-json "${POSE_MAPPING_JSON}" \
  --horizons-sec "2,4,6" \
  --loss-weights "1.0,0.7,0.5" \
  --primary-horizon-sec 2.0 \
  --anticipation-sec 2.0 \
  --max-frames 80 \
  --fps 8 \
  --img-size 256 \
  --keep-count 4096 \
  --prune-mode encoder_attn \
  --batch-size 1 \
  --batch-size-by-frames "32:4,48:2,64:2,80:1" \
  --num-workers 8 \
  --epochs 1 \
  --lr 1e-5 \
  --freeze-adapter \
  --freeze-fusion \
  --tick-cache-dir "${TICK_CACHE_DIR}" \
  --save-every 200 \
  --val-only

echo "Done A2 CA: $(date)"
python3 - <<'PY'
import json
from pathlib import Path
p=Path("/scratch/ll5914/experiments/p01_stream_mtp_a2_ctx_bucket_ca_v3/val_only_metrics.json")
m=json.loads(p.read_text())
print("overall @2s", round(100*m.get("action_top5@2s",0),2))
for ctx in (4,6,8,10):
  k=f"action_top5@2s|ctx{ctx}"
  n=m.get(f"n|{k}", "?")
  print(f"  ctx={ctx}s @2s={100*m.get(k,float('nan')):.2f}% n={n}")
PY
