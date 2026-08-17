#!/bin/bash
#SBATCH --job-name=rulstm_vitl_v3
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/ll5914/logs/rulstm_vitl_v3_%j.out
#SBATCH --error=/scratch/ll5914/logs/rulstm_vitl_v3_%j.err

set -euo pipefail

# Baseline-preserving param-matched RU-LSTM:
#   frozen original RU-LSTM (~18.1M) + zero-init residual RU-LSTM (~285.6M)
#   = ~303.7M total, close to V-JEPA ViT-L (~304M).
# Epoch 0 is exactly the original checkpoint; later epochs can only replace it
# when validation action Top-5 @2s improves.

PYTHON="/scratch/ll5914/conda_envs/SVD/bin/python"
ROOT="/home/ll5914/Jepa_baseline/rulstm_hdepic"
FEAT_DIR="/scratch/ll5914/datasets/HD-EPIC/rulstm_features/rgb_p01"
OUT_DIR="/scratch/ll5914/experiments/rulstm_hdepic_p01_stream_vitl_aligned_v3_residual"
ANN_DIR="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/stream_half_split"
BASELINE="/scratch/ll5914/experiments/rulstm_hdepic_p01_stream/rulstm_stream_best.pt"

mkdir -p /scratch/ll5914/logs "${OUT_DIR}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "========================================================"
echo "RU-LSTM ViT-L-aligned v3 (exact baseline + residual)"
echo "Job=${SLURM_JOB_ID:-manual} host=$(hostname)"
echo "Start: $(date)"
echo "========================================================"
nvidia-smi -L || true

"${PYTHON}" "${ROOT}/train_stream_rulstm.py" \
  --train-csv "${ANN_DIR}/HD_EPIC_train_stream_mtp.csv" \
  --val-csv "${ANN_DIR}/HD_EPIC_val_stream_mtp.csv" \
  --feat-dir "${FEAT_DIR}" \
  --out-dir "${OUT_DIR}" \
  --horizons-sec "2,4,6" \
  --loss-weights "1.0,0.7,0.5" \
  --alpha 0.25 \
  --max-context-sec 10.0 \
  --hidden 3816 \
  --depth 1 \
  --input-proj \
  --mlp-head \
  --trunk-layers 0 \
  --dropout 0.8 \
  --optimizer adamw \
  --lr 3e-4 \
  --weight-decay 0.05 \
  --lr-schedule cosine \
  --warmup-epochs 3 \
  --grad-clip 1.0 \
  --amp \
  --early-stop-patience 12 \
  --preserve-baseline-from "${BASELINE}" \
  --batch-size 8 \
  --num-workers 4 \
  --epochs 40

echo "Done: $(date)"
echo "param_count:"; cat "${OUT_DIR}/param_count.json"
echo "val_metrics:"; cat "${OUT_DIR}/val_metrics.json"
