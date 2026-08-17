#!/bin/bash
# EGTEA path-only port of baseline@6ce4290 V-JEPA-feature RU-LSTM.

#SBATCH --job-name=egtea_rulstm_vjfeat
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ll5914/logs/egtea_rulstm_vjfeat_%j.out
#SBATCH --error=/scratch/ll5914/logs/egtea_rulstm_vjfeat_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${PYTHON:-/scratch/ll5914/conda_envs/SVD/bin/python}"
VJEPA_ROOT="${VJEPA_ROOT:-/home/ll5914/ARVR_Video/vjepa2}"
EGTEA_ROOT="${EGTEA_ROOT:-/scratch/ll5914/datasets/EGTEA}"
ANN_DIR="${ANN_DIR:-${PROJECT_ROOT}/data/egtea/vjepa_annotations/stream_half_split/split1}"
VIDEO_ROOT="${VIDEO_ROOT:-${EGTEA_ROOT}/session_videos}"
CHECKPOINT="${CHECKPOINT:-/scratch/ll5914/models/vjepa2/vitl.pt}"
ENCODER_LORA="${ENCODER_LORA:?Set the EGTEA video encoder-LoRA sidecar}"
FEAT_DIR="${FEAT_DIR:-${EGTEA_ROOT}/rulstm_features/vjepa_vitl_split1}"
OUT_DIR="${OUT_DIR:-/scratch/ll5914/experiments/egtea_rulstm_vjepa_vitl_feat}"
ROOT="${PROJECT_ROOT}/baselines/rulstm_hdepic"

"${PROJECT_ROOT}/scripts/egtea/verify_stream_split.sh"
for path in "${CHECKPOINT}" "${ENCODER_LORA}"; do
  [[ -f "${path}" ]] || { echo "ERROR: missing ${path}" >&2; exit 2; }
done
mkdir -p "${FEAT_DIR}" "${OUT_DIR}" /scratch/ll5914/logs
export JEPA_ARVR_ROOT="${PROJECT_ROOT}" VJEPA_ROOT
export PYTHONPATH="${PROJECT_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

"${PYTHON}" -u "${ROOT}/extract_vjepa_features.py" \
  --video-root "${VIDEO_ROOT}" --out "${FEAT_DIR}" \
  --checkpoint "${CHECKPOINT}" --encoder-lora "${ENCODER_LORA}" \
  --fps 8 --img-size 256 --chunk-frames 16 --batch-chunks 4

[[ "$(find "${FEAT_DIR}" -maxdepth 1 -name '*.npy' -type f | wc -l)" == "86" ]]
[[ "$(find "${FEAT_DIR}" -maxdepth 1 -name '*.json' -type f | wc -l)" == "86" ]]

exec "${PYTHON}" -u "${ROOT}/train_stream_rulstm.py" \
  --train-csv "${ANN_DIR}/EGTEA_train_stream_mtp.csv" \
  --val-csv "${ANN_DIR}/EGTEA_val_stream_mtp.csv" \
  --feat-dir "${FEAT_DIR}" --out-dir "${OUT_DIR}" \
  --horizons-sec "2,4,6" --loss-weights "1.0,0.7,0.5" \
  --alpha 0.25 --max-context-sec 10.0 \
  --batch-size 64 --num-workers 4 --epochs 40 --lr 0.01
