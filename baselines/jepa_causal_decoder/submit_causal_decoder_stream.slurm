#!/bin/bash
#SBATCH --job-name=p01_causal_dec
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=512GB
#SBATCH --time=01:50:00
#SBATCH --signal=B:USR1@90
#SBATCH --output=/scratch/ll5914/logs/p01_causal_dec_%j.out
#SBATCH --error=/scratch/ll5914/logs/p01_causal_dec_%j.err

set -euo pipefail

# Experiment A — causal decoder on frozen V-JEPA ViT-L latents (stream MTP).
# Same protocol / encoder / MTP heads as p01_stream_mtp_2_4_6; only swap
# anticipative JEPA predictor → param-matched causal Transformer decoder
# (d=384, depth=12, heads=12 ≈29M).

PYTHON="/scratch/ll5914/conda_envs/SVD/bin/python"
PROJECT_ROOT="/home/ll5914/Jepa_yifan/JEPA_ARVR"
VJEPA_ROOT="/home/ll5914/ARVR_Video/vjepa2"
ROOT="${PROJECT_ROOT}/baselines/jepa_causal_decoder"
SCRIPT_PATH="${ROOT}/submit_causal_decoder_stream.slurm"
export VJEPA_ROOT
export PYTHONPATH="${PROJECT_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"

VIDEO_ROOT_REMOTE="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_videos"
ANN_DIR="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/stream_half_split"
CHECKPOINT="/scratch/ll5914/models/vjepa2/vitl.pt"

VIDEO_BEST_DIR="/scratch/ll5914/experiments/p01_video_pred_joint_heads_clip/action_anticipation_frozen/p01-video-pred-joint-heads-vitl16-256-10ep"
ENCODER_LORA="${VIDEO_BEST_DIR}/encoder_lora_best.pt"

OUT_DIR="/scratch/ll5914/experiments/p01_stream_causal_decoder_2_4_6"
DONE_FLAG="${OUT_DIR}/TRAINING_DONE"
mkdir -p /scratch/ll5914/logs "${OUT_DIR}" "${ANN_DIR}"

VAL_ONLY="${VAL_ONLY:-0}"

if [[ "${VAL_ONLY}" != "1" && -f "${DONE_FLAG}" ]]; then
  echo "Already done (${DONE_FLAG}); exiting."
  exit 0
fi

STAGE_ROOT="${STAGE_ROOT:-/dev/shm/${USER}/hdepic_p01_causal_${SLURM_JOB_ID:-manual}}"
VIDEO_ROOT="${STAGE_ROOT}"
_RESUBMITTED=0
TRAIN_PID=""

request_resubmit() {
  if [[ "${VAL_ONLY}" == "1" || -f "${DONE_FLAG}" || "${_RESUBMITTED}" == "1" ]]; then
    return 0
  fi
  _RESUBMITTED=1
  echo "Auto-resubmitting next chunk..."
  sbatch "${SCRIPT_PATH}" || echo "WARN: sbatch resubmit failed" >&2
}

cleanup_stage() {
  if [[ -n "${STAGE_ROOT:-}" && -d "${STAGE_ROOT}" ]]; then
    rm -rf "${STAGE_ROOT}" || true
  fi
}

on_signal() {
  echo "Caught signal — save/resubmit @ $(date)"
  request_resubmit
  if [[ -n "${TRAIN_PID}" ]] && kill -0 "${TRAIN_PID}" 2>/dev/null; then
    kill -TERM "${TRAIN_PID}" 2>/dev/null || true
  fi
}

trap cleanup_stage EXIT
trap on_signal USR1 TERM

echo "========================================================"
echo "P01 streaming causal-decoder ablation (exp A)"
echo "Job=${SLURM_JOB_ID:-manual} VAL_ONLY=${VAL_ONLY}"
echo "ANN=${ANN_DIR}"
echo "OUT=${OUT_DIR}"
echo "Stage: ${STAGE_ROOT}"
echo "Start: $(date)"
echo "========================================================"

if [[ ! -f "${ANN_DIR}/HD_EPIC_train_stream_mtp.csv" ]]; then
  echo "Building stream half-split index..."
  "${PYTHON}" "${PROJECT_ROOT}/scripts/make_hdepic_stream_half_split.py" \
    --out "${ANN_DIR}"
fi

mkdir -p "${STAGE_ROOT}/P01"
echo "Staging P01 videos -> ${STAGE_ROOT}/P01 ..."
rsync -aL --info=stats2 "${VIDEO_ROOT_REMOTE}/P01/" "${STAGE_ROOT}/P01/"
echo "Stage done: $(du -sh "${STAGE_ROOT}" | awk '{print $1}') @ $(date)"

ARGS=(
  --train-csv "${ANN_DIR}/HD_EPIC_train_stream_mtp.csv"
  --val-csv "${ANN_DIR}/HD_EPIC_val_stream_mtp.csv"
  --video-root "${VIDEO_ROOT}"
  --checkpoint "${CHECKPOINT}"
  --encoder-lora "${ENCODER_LORA}"
  --out-dir "${OUT_DIR}"
  --horizons-sec "2,4,6"
  --loss-weights "1.0,0.7,0.5"
  --primary-horizon-sec 2.0
  --anticipation-sec 2.0
  --max-frames 80
  --fps 8
  --img-size 256
  --keep-count 4096
  --batch-size 1
  --num-workers 2
  --epochs 8
  --lr 2e-5
  --save-every 200
  --decoder-depth 12
  --decoder-dim 384
  --decoder-heads 12
)

if [[ "${VAL_ONLY}" == "1" ]]; then
  ARGS+=(--val-only)
fi

cd "${PROJECT_ROOT}"
set +e
"${PYTHON}" "${ROOT}/train_stream_causal_decoder.py" "${ARGS[@]}" &
TRAIN_PID=$!
wait "${TRAIN_PID}"
train_rc=$?
TRAIN_PID=""
set -e

echo "Training exit_code=${train_rc} at $(date)"

if [[ "${VAL_ONLY}" == "1" ]]; then
  exit ${train_rc}
fi

if [[ -f "${DONE_FLAG}" ]]; then
  echo "Causal decoder stream complete: $(date)"
  exit 0
fi

request_resubmit
if [[ "${_RESUBMITTED}" == "1" ]]; then
  exit 0
fi
exit ${train_rc}
