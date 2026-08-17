#!/bin/bash
#SBATCH --job-name=rulstm_p01_cpu
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100GB
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/ll5914/logs/rulstm_p01_cpu_%j.out
#SBATCH --error=/scratch/ll5914/logs/rulstm_p01_cpu_%j.err

set -euo pipefail

PYTHON="/scratch/ll5914/conda_envs/SVD/bin/python"
ROOT="/home/ll5914/Jepa_baseline/rulstm_hdepic"
FEAT_DIR="/scratch/ll5914/datasets/HD-EPIC/rulstm_features/rgb_p01"
OUT_DIR="/scratch/ll5914/experiments/rulstm_hdepic_p01_stream"
ANN_DIR="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/stream_half_split"
TSN="/scratch/ll5914/models/rulstm/TSN-rgb.pth.tar"

mkdir -p /scratch/ll5914/logs "${FEAT_DIR}" "${OUT_DIR}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "========================================================"
echo "RU-LSTM HD-EPIC P01 Streaming (+2/+4/+6s) [CPU]"
echo "Job=${SLURM_JOB_ID:-manual} host=$(hostname)"
echo "Start: $(date)"
echo "========================================================"

"${PYTHON}" "${ROOT}/extract_rgb_features.py" \
  --video-root /scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_videos/P01 \
  --out "${FEAT_DIR}" \
  --tsn-ckpt "${TSN}" \
  --alpha 0.25 \
  --batch-size 8

"${PYTHON}" "${ROOT}/train_stream_rulstm.py" \
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
cat "${OUT_DIR}/val_metrics.json"
