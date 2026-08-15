#!/bin/bash
#SBATCH --job-name=aga_p01_stream
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/ll5914/logs/aga_p01_stream_%j.out
#SBATCH --error=/scratch/ll5914/logs/aga_p01_stream_%j.err

set -euo pipefail

# AGA (Action-Guided Attention) on HD-EPIC P01 Streaming Video (+2/+4/+6s).
# Same TSN-RGB features and stream CSVs as RU-LSTM.

PYTHON="/scratch/ll5914/conda_envs/SVD/bin/python"
ROOT="/home/ll5914/Jepa_baseline/aga_hdepic"
FEAT_DIR="/scratch/ll5914/datasets/HD-EPIC/rulstm_features/rgb_p01"
OUT_DIR="/scratch/ll5914/experiments/aga_hdepic_p01_stream"
ANN_DIR="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/stream_half_split"
TSN="/scratch/ll5914/models/rulstm/TSN-rgb.pth.tar"
EXTRACT="/home/ll5914/Jepa_baseline/rulstm_hdepic/extract_rgb_features.py"

mkdir -p /scratch/ll5914/logs "${FEAT_DIR}" "${OUT_DIR}"

echo "========================================================"
echo "AGA HD-EPIC P01 Streaming (+2/+4/+6s)"
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

echo "[2/2] Train / validate Streaming AGA ..."
"${PYTHON}" "${ROOT}/train_stream_aga.py" \
  --train-csv "${ANN_DIR}/HD_EPIC_train_stream_mtp.csv" \
  --val-csv "${ANN_DIR}/HD_EPIC_val_stream_mtp.csv" \
  --feat-dir "${FEAT_DIR}" \
  --out-dir "${OUT_DIR}" \
  --horizons-sec "2,4,6" \
  --loss-weights "1.0,0.7,0.5" \
  --time-step 1.0 \
  --feat-alpha 0.25 \
  --max-context-sec 10.0 \
  --batch-size 128 \
  --num-workers 4 \
  --epochs 50 \
  --lr 2e-4 \
  --weight-decay 0.01 \
  --amp

echo "Done: $(date)"
echo "Metrics: ${OUT_DIR}/val_metrics.json"
if [[ -f "${OUT_DIR}/val_metrics.json" ]]; then
  cat "${OUT_DIR}/val_metrics.json"
fi
"${PYTHON}" "${ROOT}/compare_to_rulstm.py" --aga-dir "${OUT_DIR}" || true
