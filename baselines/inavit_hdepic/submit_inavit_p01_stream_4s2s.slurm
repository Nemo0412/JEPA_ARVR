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

# Original InAViT (MotionFormer+HoIViT). Only the 4s RGB window / +2s label
# change; backbone architecture and param count stay the same as EK InAViT.
PYTHON="/scratch/ll5914/conda_envs/SVD/bin/python"
ROOT="/home/ll5914/Jepa_baseline/inavit_hdepic"
INAVIT="/home/ll5914/Jepa_baseline/InAViT"
RULSTM="/home/ll5914/Jepa_baseline/rulstm_hdepic"
OUT_DIR="/scratch/ll5914/experiments/inavit_hdepic_stream_4s2s"
ANN_DIR="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/stream_4s2s_all"
VIDEO_ROOT="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_videos"
CLIPS="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/clip_split_all"

mkdir -p /scratch/ll5914/logs "${OUT_DIR}" "${ANN_DIR}"
cd /home/ll5914/Jepa/JEPA_ARVR
export PYTHONPATH="/home/ll5914/Jepa/JEPA_ARVR:/home/ll5914/Jepa/JEPA_ARVR/scripts:${INAVIT}:${PYTHONPATH:-}"

echo "========================================================"
echo "Original InAViT · HD-EPIC entire-scene · 4s input / +2s"
echo "Job=${SLURM_JOB_ID:-manual}"
echo "Start: $(date)"
echo "========================================================"
nvidia-smi || true

echo "[0/2] Entire-scene 4s/+2s index (Jepa sampler, P01–P09) ..."
"${PYTHON}" "${RULSTM}/build_stream_4s2s_index.py" \
  --clips "${CLIPS}" \
  --video-root "${VIDEO_ROOT}" \
  --out "${ANN_DIR}" \
  --obs-sec 4.0 \
  --horizon-sec 2.0 \
  --participant all

echo "[1/2] InAViT Python deps ..."
"${PYTHON}" -m pip install --quiet fvcore yacs simplejson timm opencv-python-headless einops iopath imgaug filterpy

echo "[2/2] Train / validate original InAViT ..."
cd "${INAVIT}"
"${PYTHON}" "${ROOT}/train_stream_inavit.py" \
  --cfg "${INAVIT}/configs/INAVIT/HDEPIC_STREAM_4s2s.yaml" \
  --train-csv "${ANN_DIR}/HD_EPIC_train_stream_mtp.csv" \
  --val-csv "${ANN_DIR}/HD_EPIC_val_stream_mtp.csv" \
  --video-root "${VIDEO_ROOT}" \
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
