#!/bin/bash
#SBATCH --job-name=rulstm_vitl_align
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/ll5914/logs/rulstm_vitl_aligned_%j.out
#SBATCH --error=/scratch/ll5914/logs/rulstm_vitl_aligned_%j.err

set -euo pipefail

# Capacity-matched RU-LSTM ≈ V-JEPA ViT-L (~304M):
#   hidden=2048, depth=4, input_proj, mlp_head  → ~290M
# Same Streaming protocol as jepa_yifan (half-split, +2/+4/+6s).
# Reuses precomputed TSN-RGB features (no re-extract).

PYTHON="/scratch/ll5914/conda_envs/SVD/bin/python"
ROOT="/home/ll5914/Jepa_baseline/rulstm_hdepic"
FEAT_DIR="/scratch/ll5914/datasets/HD-EPIC/rulstm_features/rgb_p01"
OUT_DIR="/scratch/ll5914/experiments/rulstm_hdepic_p01_stream_vitl_aligned"
ANN_DIR="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/stream_half_split"

mkdir -p /scratch/ll5914/logs "${OUT_DIR}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "========================================================"
echo "RU-LSTM ViT-L-aligned (~290M) Streaming +2/+4/+6s"
echo "Job=${SLURM_JOB_ID:-manual} host=$(hostname)"
echo "Start: $(date)"
echo "========================================================"
nvidia-smi -L || true

# Ensure features exist (skip if already extracted)
n_npy=$(ls "${FEAT_DIR}"/*.npy 2>/dev/null | wc -l)
echo "feature npy count=${n_npy}"
if [[ "${n_npy}" -lt 27 ]]; then
  echo "Missing features; extracting..."
  "${PYTHON}" "${ROOT}/extract_rgb_features.py" \
    --video-root /scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_videos/P01 \
    --out "${FEAT_DIR}" \
    --tsn-ckpt /scratch/ll5914/models/rulstm/TSN-rgb.pth.tar \
    --alpha 0.25 --batch-size 128
fi

"${PYTHON}" "${ROOT}/train_stream_rulstm.py" \
  --train-csv "${ANN_DIR}/HD_EPIC_train_stream_mtp.csv" \
  --val-csv "${ANN_DIR}/HD_EPIC_val_stream_mtp.csv" \
  --feat-dir "${FEAT_DIR}" \
  --out-dir "${OUT_DIR}" \
  --horizons-sec "2,4,6" \
  --loss-weights "1.0,0.7,0.5" \
  --alpha 0.25 \
  --max-context-sec 10.0 \
  --hidden 2048 \
  --depth 4 \
  --input-proj \
  --mlp-head \
  --dropout 0.5 \
  --optimizer adamw \
  --lr 3e-4 \
  --weight-decay 0.01 \
  --lr-milestones "12,20" \
  --batch-size 16 \
  --num-workers 4 \
  --epochs 30

echo "Done: $(date)"
echo "param_count:"; cat "${OUT_DIR}/param_count.json"
echo "val_metrics:"; cat "${OUT_DIR}/val_metrics.json"
