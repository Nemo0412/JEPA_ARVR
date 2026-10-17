#!/bin/bash
#SBATCH --job-name=lstm_p01_4s2s
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/ll5914/logs/lstm_p01_stream_4s2s_%j.out
#SBATCH --error=/scratch/ll5914/logs/lstm_p01_stream_4s2s_%j.err

set -euo pipefail

# Streaming LSTM: last 4s of TSN-RGB @4fps, predict +2s.
PYTHON="/scratch/ll5914/conda_envs/SVD/bin/python"
ROOT="/home/ll5914/Jepa_baseline/lstm_hdepic"
FEAT_DIR="/scratch/ll5914/datasets/HD-EPIC/rulstm_features/rgb_p01"
OUT_DIR="/scratch/ll5914/experiments/lstm_hdepic_p01_stream_4s2s"
ANN_DIR="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/stream_half_split"
EXTRACT="/home/ll5914/Jepa_baseline/rulstm_hdepic/extract_rgb_features.py"
TSN="/scratch/ll5914/models/rulstm/TSN-rgb.pth.tar"

mkdir -p /scratch/ll5914/logs "${FEAT_DIR}" "${OUT_DIR}"

echo "========================================================"
echo "LSTM HD-EPIC P01 Streaming (4s input / +2s)"
echo "Job=${SLURM_JOB_ID:-manual}"
echo "Start: $(date)"
echo "========================================================"
nvidia-smi || true

echo "[1/2] Ensure TSN-RGB features @4fps ..."
"${PYTHON}" "${EXTRACT}" \
  --video-root /scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_videos/P01 \
  --out "${FEAT_DIR}" \
  --tsn-ckpt "${TSN}" \
  --alpha 0.25 \
  --batch-size 128

echo "[2/2] Train / validate Streaming LSTM ..."
"${PYTHON}" "${ROOT}/train_stream_lstm.py" \
  --train-csv "${ANN_DIR}/HD_EPIC_train_stream_mtp.csv" \
  --val-csv "${ANN_DIR}/HD_EPIC_val_stream_mtp.csv" \
  --feat-dir "${FEAT_DIR}" \
  --out-dir "${OUT_DIR}" \
  --max-context-sec 4.0 \
  --horizon-sec 2.0 \
  --batch-size 64 \
  --num-workers 4 \
  --epochs 40 \
  --lr 1e-3 \
  --amp

echo "Done: $(date)"
if [[ -f "${OUT_DIR}/val_metrics.json" ]]; then
  cat "${OUT_DIR}/val_metrics.json"
fi
