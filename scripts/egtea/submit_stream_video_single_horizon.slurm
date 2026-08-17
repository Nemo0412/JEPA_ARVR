#!/bin/bash
# EGTEA path port of opencl@22743f4 video-only Vanilla V-JEPA.
# One independent training job per HORIZON=2,4,6; no communicating MTP head.

#SBATCH --job-name=egtea_vanilla_1h
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=512GB
#SBATCH --time=01:50:00
#SBATCH --output=/scratch/ll5914/logs/egtea_vanilla_1h_%j.out
#SBATCH --error=/scratch/ll5914/logs/egtea_vanilla_1h_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${PYTHON:-/scratch/ll5914/conda_envs/SVD/bin/python}"
VJEPA_ROOT="${VJEPA_ROOT:-/home/ll5914/ARVR_Video/vjepa2}"
EGTEA_ROOT="${EGTEA_ROOT:-/scratch/ll5914/datasets/EGTEA}"
ANN_DIR="${ANN_DIR:-${PROJECT_ROOT}/data/egtea/vjepa_annotations/stream_half_split/split1}"
VIDEO_ROOT="${VIDEO_ROOT:-${EGTEA_ROOT}/stream_session_videos}"
CHECKPOINT="${CHECKPOINT:-/scratch/ll5914/models/vjepa2/vitl.pt}"
ENCODER_LORA="${ENCODER_LORA:?Set the EGTEA video encoder-LoRA sidecar}"
PREDICTOR_LORA="${PREDICTOR_LORA:?Set the EGTEA video predictor-LoRA sidecar}"
HORIZON="${HORIZON:?Set HORIZON to 2, 4, or 6}"
OUT_DIR="${OUT_DIR:-/scratch/ll5914/experiments/egtea_vanilla_video_h${HORIZON}}"
DATA_DIR="${DATA_DIR:-${OUT_DIR}/selected_horizon_csv}"

case "${HORIZON}" in 2|4|6) ;; *) echo "HORIZON must be 2, 4, or 6" >&2; exit 2 ;; esac
"${PROJECT_ROOT}/scripts/egtea/verify_stream_split.sh"
for path in "${CHECKPOINT}" "${ENCODER_LORA}" "${PREDICTOR_LORA}"; do
  [[ -f "${path}" ]] || { echo "ERROR: missing ${path}" >&2; exit 2; }
done
"${PROJECT_ROOT}/scripts/egtea/verify_stream_video_tree.sh" "${VIDEO_ROOT}" "${ANN_DIR}"
mkdir -p "${OUT_DIR}" "${DATA_DIR}" /scratch/ll5914/logs
export VJEPA_ROOT PYTHONPATH="${PROJECT_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "${PROJECT_ROOT}"
"${PYTHON}" scripts/egtea/prepare_vanilla_single_horizon_csv.py \
  --train-csv "${ANN_DIR}/EGTEA_train_stream_mtp.csv" \
  --val-csv "${ANN_DIR}/EGTEA_val_stream_mtp.csv" \
  --out-dir "${DATA_DIR}" --horizon "${HORIZON}" --mode full

exec "${PYTHON}" -m app.hdepic_lora_action_anticipation.train_stream_video \
  --train-csv "${DATA_DIR}/train.csv" \
  --val-csv "${DATA_DIR}/val.csv" \
  --video-root "${VIDEO_ROOT}" \
  --checkpoint "${CHECKPOINT}" \
  --encoder-lora "${ENCODER_LORA}" \
  --predictor-lora "${PREDICTOR_LORA}" \
  --out-dir "${OUT_DIR}" \
  --horizons-sec "${HORIZON}" --loss-weights 1.0 \
  --primary-horizon-sec "${HORIZON}" \
  --max-frames 80 --fps 8 --img-size 256 --keep-count 4096 \
  --batch-size 1 --num-workers 4 --epochs 8 --lr 2e-5 --save-every 200
