#!/bin/bash
#SBATCH --job-name=rulstm_vitl_v2
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/ll5914/logs/rulstm_vitl_v2_%j.out
#SBATCH --error=/scratch/ll5914/logs/rulstm_vitl_v2_%j.err

set -euo pipefail

# Optimized ViT-L-aligned RU-LSTM (~328M):
#   - shallow-wide: hidden=4096, depth=1 (v1 depth=4 failed to train)
#   - LayerNorm + orthogonal LSTM init + forget bias=1
#   - warm-start overlapping weights from 18M baseline
#   - AdamW + warmup/cosine, label smoothing, feature noise, AMP, early stop

PYTHON="/scratch/ll5914/conda_envs/SVD/bin/python"
ROOT="/home/ll5914/Jepa_baseline/rulstm_hdepic"
FEAT_DIR="/scratch/ll5914/datasets/HD-EPIC/rulstm_features/rgb_p01"
OUT_DIR="/scratch/ll5914/experiments/rulstm_hdepic_p01_stream_vitl_aligned_v2"
ANN_DIR="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/stream_half_split"
INIT_FROM="/scratch/ll5914/experiments/rulstm_hdepic_p01_stream/rulstm_stream_best.pt"

mkdir -p /scratch/ll5914/logs "${OUT_DIR}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "========================================================"
echo "RU-LSTM ViT-L-aligned v2 (shallow-wide + train fixes)"
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
  --hidden 4096 \
  --depth 1 \
  --input-proj \
  --mlp-head \
  --trunk-layers 0 \
  --dropout 0.5 \
  --optimizer adamw \
  --lr 1e-3 \
  --weight-decay 0.05 \
  --lr-schedule cosine \
  --warmup-epochs 3 \
  --label-smoothing 0.1 \
  --feat-noise 0.05 \
  --grad-clip 1.0 \
  --amp \
  --early-stop-patience 12 \
  --init-from "${INIT_FROM}" \
  --batch-size 8 \
  --num-workers 4 \
  --epochs 40

echo "Done: $(date)"
echo "param_count:"; cat "${OUT_DIR}/param_count.json"
echo "val_metrics:"; cat "${OUT_DIR}/val_metrics.json"
