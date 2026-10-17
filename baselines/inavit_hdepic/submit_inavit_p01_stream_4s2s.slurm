#!/bin/bash
#SBATCH --job-name=inavit_p01_4s2s
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ll5914/logs/inavit_p01_stream_4s2s_%j.out
#SBATCH --error=/scratch/ll5914/logs/inavit_p01_stream_4s2s_%j.err

set -euo pipefail

# Streaming InAViT: last 4s RGB (16 frames) → predict +2s.
PYTHON="/scratch/ll5914/conda_envs/SVD/bin/python"
ROOT="/home/ll5914/Jepa_baseline/inavit_hdepic"
INAVIT="/home/ll5914/Jepa_baseline/InAViT"
OUT_DIR="/scratch/ll5914/experiments/inavit_hdepic_p01_stream_4s2s"
ANN_DIR="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/stream_half_split"

mkdir -p /scratch/ll5914/logs "${OUT_DIR}"
cd "${INAVIT}"
export PYTHONPATH="${INAVIT}:${PYTHONPATH:-}"

echo "========================================================"
echo "InAViT HD-EPIC P01 Streaming (4s input / +2s)"
echo "Job=${SLURM_JOB_ID:-manual}"
echo "Start: $(date)"
echo "========================================================"
nvidia-smi || true

echo "[0/1] Ensure InAViT Python deps ..."
"${PYTHON}" -m pip install --quiet fvcore yacs simplejson timm opencv-python-headless einops iopath imgaug filterpy

echo "[1/1] Train / validate Streaming InAViT ..."
"${PYTHON}" "${ROOT}/train_stream_inavit.py" \
  --cfg "${INAVIT}/configs/INAVIT/HDEPIC_STREAM_4s2s.yaml" \
  --train-csv "${ANN_DIR}/HD_EPIC_train_stream_mtp.csv" \
  --val-csv "${ANN_DIR}/HD_EPIC_val_stream_mtp.csv" \
  --video-root /scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_videos/P01 \
  --out-dir "${OUT_DIR}" \
  --obs-sec 4.0 \
  --num-frames 16 \
  --batch-size 4 \
  --num-workers 4 \
  --epochs 10 \
  --lr 1e-4

echo "Done: $(date)"
if [[ -f "${OUT_DIR}/val_metrics.json" ]]; then
  cat "${OUT_DIR}/val_metrics.json"
fi
