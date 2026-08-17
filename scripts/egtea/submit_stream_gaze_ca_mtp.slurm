#!/bin/bash
# EGTEA data-port of ll's opencl@29063fe gaze-only CA + communicating MTP.
# Set USE_FFN=1 for the matched +FFN arm; that adds only --use-ffn --ffn-mult 4.

#SBATCH --job-name=egtea_gaze_ca_mtp
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=512GB
#SBATCH --time=01:50:00
#SBATCH --signal=B:USR1@90
#SBATCH --output=/scratch/ll5914/logs/egtea_stream_gaze_ca_%j.out
#SBATCH --error=/scratch/ll5914/logs/egtea_stream_gaze_ca_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${PYTHON:-/scratch/ll5914/conda_envs/SVD/bin/python}"
VJEPA_ROOT="${VJEPA_ROOT:-/home/ll5914/ARVR_Video/vjepa2}"
EGTEA_ROOT="${EGTEA_ROOT:-/scratch/ll5914/datasets/EGTEA}"
ANN_DIR="${ANN_DIR:-${PROJECT_ROOT}/data/egtea/vjepa_annotations/stream_half_split/split1}"
VIDEO_ROOT="${VIDEO_ROOT:-${EGTEA_ROOT}/stream_session_videos}"
GAZE_ROOT="${GAZE_ROOT:?Set GAZE_ROOT to the directory containing <session>.txt BeGaze exports}"
CHECKPOINT="${CHECKPOINT:-/scratch/ll5914/models/vjepa2/vitl.pt}"
ENCODER_LORA="${ENCODER_LORA:?Set ENCODER_LORA to the intended EGTEA gaze encoder-LoRA sidecar}"
PREDICTOR_LORA="${PREDICTOR_LORA:?Set PREDICTOR_LORA to the intended EGTEA stream predictor-LoRA sidecar}"
USE_FFN="${USE_FFN:-0}"
OUT_DIR="${OUT_DIR:-/scratch/ll5914/experiments/egtea_stream_gaze_ca_mtp_ffn${USE_FFN}}"
TICK_CACHE_DIR="${TICK_CACHE_DIR:-${EGTEA_ROOT}/caches/stream_gaze_ca_ticks_256}"
GAZE_COVERAGE="${GAZE_COVERAGE:-${ANN_DIR}/frozen_gaze_coverage.json}"

case "${USE_FFN}" in 0|1) ;; *) echo "USE_FFN must be 0 or 1" >&2; exit 2 ;; esac
"${PROJECT_ROOT}/scripts/egtea/verify_stream_split.sh"
for path in "${CHECKPOINT}" "${ENCODER_LORA}" "${PREDICTOR_LORA}" "${GAZE_COVERAGE}"; do
  [[ -f "${path}" ]] || { echo "ERROR: missing ${path}" >&2; exit 2; }
done
[[ -d "${VIDEO_ROOT}" ]] || { echo "ERROR: missing ${VIDEO_ROOT}" >&2; exit 2; }
[[ -d "${GAZE_ROOT}" ]] || { echo "ERROR: missing ${GAZE_ROOT}" >&2; exit 2; }
mkdir -p "${OUT_DIR}" "${TICK_CACHE_DIR}" /scratch/ll5914/logs

export VJEPA_ROOT PYTHONPATH="${PROJECT_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TRI_MODAL_GAZE_MAP_CACHE=256
export GAZE_DECODER_CACHE_READER=1 GAZE_DECODER_NUM_THREADS=1

ARGS=(
  --train-csv "${ANN_DIR}/EGTEA_train_stream_mtp.csv"
  --val-csv "${ANN_DIR}/EGTEA_val_stream_mtp.csv"
  --video-root "${VIDEO_ROOT}"
  --checkpoint "${CHECKPOINT}"
  --encoder-lora "${ENCODER_LORA}"
  --predictor-lora "${PREDICTOR_LORA}"
  --adapter-ckpt "/tmp/intentional_nonexistent_egtea_gaze_adapter.pt"
  --out-dir "${OUT_DIR}"
  --gaze-root "${GAZE_ROOT}"
  --gaze-extract-root "${GAZE_ROOT}"
  --gaze-format egtea
  --egtea-gaze-dir "${GAZE_ROOT}"
  --gaze-coverage-json "${GAZE_COVERAGE}"
  --ca-aux gaze
  --horizons-sec "2,4,6"
  --loss-weights "1.0,0.7,0.5"
  --primary-horizon-sec 2.0
  --anticipation-sec 2.0
  --max-frames 80 --fps 8 --img-size 256 --keep-count 4096
  --cap-total-to-keep --max-aux-tokens 1600
  --prune-mode postfuse_recency
  --batch-size 1 --batch-size-by-frames "32:4,48:2,64:2,80:1"
  --num-workers 4 --epochs 8 --lr 2e-5
  --fusion-lr-mult 2.0 --adapter-lr-mult 1.0
  --tick-cache-dir "${TICK_CACHE_DIR}" --save-every 200
)
[[ "${USE_FFN}" == "1" ]] && ARGS+=(--use-ffn --ffn-mult 4)

printf 'EGTEA_ARG\t%s\n' "${ARGS[@]}"
cd "${PROJECT_ROOT}"
exec "${PYTHON}" -m app.hdepic_lora_action_anticipation.train_stream_mtp_concat_ca "${ARGS[@]}"
