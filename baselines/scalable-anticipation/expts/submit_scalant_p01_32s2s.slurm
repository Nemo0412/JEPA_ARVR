#!/bin/bash
#SBATCH --job-name=scalant_p01_32s2s
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/ll5914/logs/scalant_p01_32s2s_%j.out
#SBATCH --error=/scratch/ll5914/logs/scalant_p01_32s2s_%j.err

set -euo pipefail

# ScalAnt v2 on HD-EPIC P01 clip split (see README_V2.md):
#   32s history (28s long + 4s working), MAMBA encoder, predict the action 2s ahead.
# Requires mamba-ssm in the Python environment.
# Same TSN-RGB features @4fps as RU-LSTM / AGA.

PYTHON="/scratch/ll5914/conda_envs/SVD/bin/python"
ROOT="/home/ll5914/Jepa_baseline/scalable-anticipation"
FEAT_DIR="/scratch/ll5914/datasets/HD-EPIC/rulstm_features/rgb_p01"
OUT_DIR="/scratch/ll5914/experiments/scalant_hdepic_p01_32s2s_mamba"
EXTRACT="/home/ll5914/Jepa_baseline/rulstm_hdepic/extract_rgb_features.py"
TSN="/scratch/ll5914/models/rulstm/TSN-rgb.pth.tar"

mkdir -p /scratch/ll5914/logs "${FEAT_DIR}" "${OUT_DIR}"

echo "========================================================"
echo "ScalAnt v2 HD-EPIC P01 clip 32s history / +2s anticipation"
echo "Job=${SLURM_JOB_ID:-manual}"
echo "Start: $(date)"
echo "========================================================"
nvidia-smi || true

echo "[0/3] Lightweight Python deps (nestconfig / fvcore) ..."
"${PYTHON}" -m pip install --quiet nestconfig yacs fvcore

echo "[1/3] Ensure TSN-RGB features @4fps ..."
"${PYTHON}" "${EXTRACT}" \
  --video-root /scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_videos/P01 \
  --out "${FEAT_DIR}" \
  --tsn-ckpt "${TSN}" \
  --alpha 0.25 \
  --batch-size 128

echo "[2/3] Train / validate ScalAnt ..."
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES=0
export SCALANT_CKPT_DIR="${OUT_DIR}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

"${PYTHON}" main.py \
  --cfg configs/hdepic/clip_32s_2s.yaml \
  --opts \
  DATA.DATA_ROOT_PATH /scratch/ll5914/datasets/HD-EPIC \
  DATA.FEAT_DIR rulstm_features/rgb_p01 \
  USE_WANDB False

echo "Done: $(date)"
echo "Checkpoints: ${OUT_DIR}"
ls -lh "${OUT_DIR}" || true
