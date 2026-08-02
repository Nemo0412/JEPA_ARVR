#!/bin/bash
#SBATCH --job-name=p01_jepa_18m
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=512GB
#SBATCH --time=01:50:00
#SBATCH --signal=B:USR1@90
#SBATCH --output=/scratch/ll5914/logs/p01_jepa_tiny18m_%j.out
#SBATCH --error=/scratch/ll5914/logs/p01_jepa_tiny18m_%j.err

set -euo pipefail

# Capacity-matched tiny V-JEPA vs original RU-LSTM (~18M):
#   encoder=vit_tiny (~5.7M) + predictor 320d×10 (~12M) ≈ 18M backbone
#   Same stream half-split protocol: +2/+4/+6s, context 4→10s, P01
#   From-scratch (no Meta small ckpt) → train encoder end-to-end

PYTHON="/scratch/ll5914/conda_envs/SVD/bin/python"
PROJECT_ROOT="/home/ll5914/Jepa_yifan/JEPA_ARVR"
VJEPA_ROOT="/home/ll5914/ARVR_Video/vjepa2"
BASELINE_ROOT="/home/ll5914/Jepa_baseline/jepa_tiny_18m"
SCRIPT_PATH="${BASELINE_ROOT}/submit_stream_tiny_18m.slurm"
export VJEPA_ROOT
export PYTHONPATH="${PROJECT_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
export TMPDIR=/scratch/ll5914/tmp
mkdir -p /scratch/ll5914/tmp /scratch/ll5914/logs

VIDEO_ROOT_REMOTE="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_videos"
ANN_DIR="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/stream_half_split"
OUT_DIR="/scratch/ll5914/experiments/p01_stream_mtp_tiny18m_2_4_6"
CKPT="${OUT_DIR}/vitti_pred320x10_init.pt"
DONE_FLAG="${OUT_DIR}/TRAINING_DONE"
STOP_FLAG="${OUT_DIR}/STOP_REQUESTED"
mkdir -p "${OUT_DIR}"

VAL_ONLY="${VAL_ONLY:-0}"
if [[ -f "${STOP_FLAG}" ]]; then
  echo "STOP_REQUESTED present (${STOP_FLAG}); exiting without train/resubmit."
  exit 0
fi
if [[ "${VAL_ONLY}" != "1" && -f "${DONE_FLAG}" ]]; then
  echo "Already done (${DONE_FLAG}); exiting."
  exit 0
fi

STAGE_ROOT="${STAGE_ROOT:-/dev/shm/${USER}/hdepic_p01_tiny18m_${SLURM_JOB_ID:-manual}}"
VIDEO_ROOT="${STAGE_ROOT}"
_RESUBMITTED=0
TRAIN_PID=""

request_resubmit() {
  if [[ "${VAL_ONLY}" == "1" || -f "${DONE_FLAG}" || -f "${STOP_FLAG}" || "${_RESUBMITTED}" == "1" ]]; then
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
echo "P01 stream MTP tiny≈18M (vit_tiny + pred320×10) @ 2/4/6s"
echo "Job=${SLURM_JOB_ID:-manual} VAL_ONLY=${VAL_ONLY}"
echo "OUT=${OUT_DIR}"
echo "Start: $(date)"
nvidia-smi || true
echo "========================================================"

if [[ ! -f "${ANN_DIR}/HD_EPIC_train_stream_mtp.csv" ]]; then
  "${PYTHON}" "${PROJECT_ROOT}/scripts/make_hdepic_stream_half_split.py" --out "${ANN_DIR}"
fi

if [[ ! -f "${CKPT}" ]]; then
  echo "[init] Writing random-init tiny checkpoint..."
  "${PYTHON}" "${BASELINE_ROOT}/make_init_ckpt.py" \
    --out "${CKPT}" \
    --encoder-model vit_tiny \
    --predictor-embed-dim 320 \
    --predictor-depth 10 \
    --predictor-heads 8 \
    --num-frames 80 \
    --img-size 256
fi

mkdir -p "${STAGE_ROOT}/P01"
echo "Staging P01 videos -> ${STAGE_ROOT}/P01 ..."
rsync -aL --info=stats2 "${VIDEO_ROOT_REMOTE}/P01/" "${STAGE_ROOT}/P01/"
echo "Stage done: $(du -sh "${STAGE_ROOT}" | awk '{print $1}') @ $(date)"

ARGS=(
  --train-csv "${ANN_DIR}/HD_EPIC_train_stream_mtp.csv"
  --val-csv "${ANN_DIR}/HD_EPIC_val_stream_mtp.csv"
  --video-root "${VIDEO_ROOT}"
  --checkpoint "${CKPT}"
  --out-dir "${OUT_DIR}"
  --encoder-model vit_tiny
  --predictor-embed-dim 320
  --predictor-depth 10
  --predictor-heads 8
  --classifier-depth 2
  --classifier-heads 8
  --train-encoder
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
  --epochs 12
  --lr 1e-4
  --save-every 200
)

if [[ "${VAL_ONLY}" == "1" ]]; then
  ARGS+=(--val-only)
fi

cd "${PROJECT_ROOT}"
set +e
"${PYTHON}" -m app.hdepic_lora_action_anticipation.train_stream_mtp "${ARGS[@]}" &
TRAIN_PID=$!
wait "${TRAIN_PID}"
train_rc=$?
TRAIN_PID=""
set -e

echo "Training exit_code=${train_rc} at $(date)"

# Write comparison table vs RU-LSTM when val metrics exist
"${PYTHON}" "${BASELINE_ROOT}/compare_to_rulstm.py" --jepa-dir "${OUT_DIR}" || true

if [[ "${VAL_ONLY}" == "1" ]]; then
  exit ${train_rc}
fi

if [[ -f "${DONE_FLAG}" ]]; then
  echo "Tiny-18M stream MTP complete: $(date)"
  exit 0
fi

request_resubmit
if [[ "${_RESUBMITTED}" == "1" ]]; then
  exit 0
fi
exit ${train_rc}
