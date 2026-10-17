#!/bin/bash
#SBATCH --job-name=rulstm_hdepic_4s2s
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/ll5914/logs/rulstm_hdepic_stream_4s2s_%j.out
#SBATCH --error=/scratch/ll5914/logs/rulstm_hdepic_stream_4s2s_%j.err

set -euo pipefail

# Original RU-LSTM (RGB branch, hidden=1024, depth=1). Only the input window
# changes: last 4s TSN-RGB @4fps (16 steps), unroll +2s. Params unchanged vs
# the official Rolling-Unrolling LSTM. Train+val on HD-EPIC entire-scene ticks
# (P01–P09), same sampler as /home/ll5914/Jepa/JEPA_ARVR/scripts/make_hdepic_stream_half_split.py.
PYTHON="/scratch/ll5914/conda_envs/SVD/bin/python"
ROOT="/home/ll5914/Jepa_baseline/rulstm_hdepic"
FEAT_DIR="/scratch/ll5914/datasets/HD-EPIC/rulstm_features/rgb_all"
OUT_DIR="/scratch/ll5914/experiments/rulstm_hdepic_stream_4s2s"
ANN_DIR="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/stream_4s2s_all"
TSN="/scratch/ll5914/models/rulstm/TSN-rgb.pth.tar"
VIDEO_ROOT="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_videos"
CLIPS="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/clip_split_all"

mkdir -p /scratch/ll5914/logs "${FEAT_DIR}" "${OUT_DIR}" "${ANN_DIR}"
cd /home/ll5914/Jepa/JEPA_ARVR
export PYTHONPATH="/home/ll5914/Jepa/JEPA_ARVR:/home/ll5914/Jepa/JEPA_ARVR/scripts:${PYTHONPATH:-}"

echo "========================================================"
echo "Original RU-LSTM · HD-EPIC entire-scene · 4s input / +2s"
echo "Job=${SLURM_JOB_ID:-manual}"
echo "Start: $(date)"
echo "========================================================"
nvidia-smi || true

echo "[1/3] Build entire-scene 4s/+2s index (Jepa sampler, P01–P09) ..."
"${PYTHON}" "${ROOT}/build_stream_4s2s_index.py" \
  --clips "${CLIPS}" \
  --video-root "${VIDEO_ROOT}" \
  --out "${ANN_DIR}" \
  --obs-sec 4.0 \
  --horizon-sec 2.0 \
  --participant all

echo "[2/3] TSN-RGB features @4fps for all HD-EPIC videos ..."
"${PYTHON}" "${ROOT}/extract_rgb_features.py" \
  --video-root "${VIDEO_ROOT}" \
  --out "${FEAT_DIR}" \
  --tsn-ckpt "${TSN}" \
  --alpha 0.25 \
  --batch-size 128

echo "[3/3] Train / validate original RU-LSTM (hidden=1024, no extra capacity) ..."
"${PYTHON}" "${ROOT}/train_stream_rulstm.py" \
  --train-csv "${ANN_DIR}/HD_EPIC_train_stream_mtp.csv" \
  --val-csv "${ANN_DIR}/HD_EPIC_val_stream_mtp.csv" \
  --feat-dir "${FEAT_DIR}" \
  --out-dir "${OUT_DIR}" \
  --horizons-sec "2" \
  --loss-weights "1.0" \
  --alpha 0.25 \
  --max-context-sec 4.0 \
  --hidden 1024 \
  --depth 1 \
  --no-layernorm \
  --dropout 0.8 \
  --batch-size 64 \
  --num-workers 4 \
  --epochs 40 \
  --lr 0.01 \
  --optimizer sgd

echo "Done: $(date)"
if [[ -f "${OUT_DIR}/val_metrics.json" ]]; then
  cat "${OUT_DIR}/val_metrics.json"
fi
if [[ -f "${OUT_DIR}/param_count.json" ]]; then
  echo "param_count:"
  cat "${OUT_DIR}/param_count.json"
fi
