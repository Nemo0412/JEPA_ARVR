#!/bin/bash
#SBATCH --job-name=rulstm_vjepa_feat
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ll5914/logs/rulstm_vjepa_feat_%j.out
#SBATCH --error=/scratch/ll5914/logs/rulstm_vjepa_feat_%j.err

set -euo pipefail

# Stronger-features RU-LSTM baseline:
#   1) Extract frozen V-JEPA ViT-L (+ HD-EPIC encoder LoRA) features @4Hz, 1024-d
#   2) Train original Streaming RU-LSTM (~18M) with the same recipe as TSN baseline

PYTHON="/scratch/ll5914/conda_envs/SVD/bin/python"
ROOT="/home/ll5914/Jepa_baseline/rulstm_hdepic"
FEAT_DIR="/scratch/ll5914/datasets/HD-EPIC/rulstm_features/vjepa_vitl_p01"
OUT_DIR="/scratch/ll5914/experiments/rulstm_hdepic_p01_stream_vjepa_vitl_feat"
ANN_DIR="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/stream_half_split"
VIDEO_ROOT="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_videos/P01"
CHECKPOINT="/scratch/ll5914/models/vjepa2/vitl.pt"
ENCODER_LORA="/scratch/ll5914/experiments/p01_video_pred_joint_heads_clip/action_anticipation_frozen/p01-video-pred-joint-heads-vitl16-256-10ep/encoder_lora_best.pt"

export JEPA_ARVR_ROOT="/home/ll5914/Jepa_yifan/JEPA_ARVR"
export VJEPA_ROOT="/home/ll5914/ARVR_Video/vjepa2"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

mkdir -p /scratch/ll5914/logs "${FEAT_DIR}" "${OUT_DIR}"

echo "========================================================"
echo "RU-LSTM on V-JEPA ViT-L features (original ~18M head)"
echo "Job=${SLURM_JOB_ID:-manual} host=$(hostname)"
echo "Start: $(date)"
echo "========================================================"
nvidia-smi -L || true

echo "[1/2] Extract V-JEPA ViT-L features @8fps / tubelet=2 → 4Hz ..."
"${PYTHON}" -u "${ROOT}/extract_vjepa_features.py" \
  --video-root "${VIDEO_ROOT}" \
  --out "${FEAT_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --encoder-lora "${ENCODER_LORA}" \
  --fps 8 \
  --img-size 256 \
  --chunk-frames 16 \
  --batch-chunks 4

echo "[2/2] Train original Streaming RU-LSTM (~18M) on ViT-L features ..."
"${PYTHON}" -u "${ROOT}/train_stream_rulstm.py" \
  --train-csv "${ANN_DIR}/HD_EPIC_train_stream_mtp.csv" \
  --val-csv "${ANN_DIR}/HD_EPIC_val_stream_mtp.csv" \
  --feat-dir "${FEAT_DIR}" \
  --out-dir "${OUT_DIR}" \
  --horizons-sec "2,4,6" \
  --loss-weights "1.0,0.7,0.5" \
  --alpha 0.25 \
  --max-context-sec 10.0 \
  --batch-size 64 \
  --num-workers 4 \
  --epochs 40 \
  --lr 0.01

echo "Done: $(date)"
echo "param_count:"; cat "${OUT_DIR}/param_count.json" 2>/dev/null || true
echo "val_metrics:"; cat "${OUT_DIR}/val_metrics.json"
