#!/bin/bash
#SBATCH --job-name=p01_qwen2vl2b
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256GB
#SBATCH --time=01:50:00
#SBATCH --signal=B:USR1@90
#SBATCH --output=/scratch/ll5914/logs/p01_qwen2vl2b_%j.out
#SBATCH --error=/scratch/ll5914/logs/p01_qwen2vl2b_%j.err

set -euo pipefail

# Qwen2-VL-2B LoRA probe on P01 stream MTP (+2/+4/+6s)
# Official smallest Qwen-VL (~2B); still ~5x V-JEPA vanilla (~400M).
# Compare against: /scratch/ll5914/experiments/p01_stream_mtp_2_4_6

PYTHON="/scratch/ll5914/conda_envs/SVD/bin/python"
ROOT="/home/ll5914/Jepa_baseline/qwen_vl_stream"
SCRIPT_PATH="${ROOT}/submit_qwen_stream.slurm"
ANN_DIR="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/stream_half_split"
VIDEO_ROOT_REMOTE="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_videos"
OUT_DIR="/scratch/ll5914/experiments/p01_stream_qwen2vl2b_2_4_6"
DEFAULT_SNAP=$(ls -d /scratch/ll5914/hf_cache/models--Qwen--Qwen2-VL-2B-Instruct/snapshots/*/ 2>/dev/null | head -1 || true)
if [[ -d /scratch/ll5914/hf_cache/Qwen2-VL-2B-Instruct ]]; then
  DEFAULT_LOCAL="/scratch/ll5914/hf_cache/Qwen2-VL-2B-Instruct"
else
  DEFAULT_LOCAL=""
fi
MODEL_ID="${QWEN_MODEL_ID:-${DEFAULT_LOCAL:-${DEFAULT_SNAP:-Qwen/Qwen2-VL-2B-Instruct}}}"
DONE_FLAG="${OUT_DIR}/TRAINING_DONE"

export HF_HOME="${HF_HOME:-/scratch/ll5914/hf_cache}"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}"
export TMPDIR=/scratch/ll5914/tmp
mkdir -p /scratch/ll5914/logs "${OUT_DIR}" /scratch/ll5914/tmp

# Offline only when using a local snapshot path
LOCAL_ONLY_FLAG=()
if [[ -d "${MODEL_ID}" ]]; then
  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_OFFLINE=1
  LOCAL_ONLY_FLAG=(--local-files-only)
fi

VAL_ONLY="${VAL_ONLY:-0}"
if [[ "${VAL_ONLY}" != "1" && -f "${DONE_FLAG}" ]]; then
  echo "Already done (${DONE_FLAG}); exiting."
  exit 0
fi

STAGE_ROOT="${STAGE_ROOT:-/dev/shm/${USER}/hdepic_p01_qwen2b_${SLURM_JOB_ID:-manual}}"
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
echo "P01 stream Qwen2-VL-2B LoRA probe @ 2/4/6s"
echo "Job=${SLURM_JOB_ID:-manual} VAL_ONLY=${VAL_ONLY}"
echo "MODEL=${MODEL_ID}"
echo "OUT=${OUT_DIR}"
echo "Start: $(date)"
nvidia-smi || true
echo "========================================================"

mkdir -p "${STAGE_ROOT}/P01"
echo "Staging P01 videos -> ${STAGE_ROOT}/P01 ..."
rsync -aL --info=stats2 "${VIDEO_ROOT_REMOTE}/P01/" "${STAGE_ROOT}/P01/"
echo "Stage done: $(du -sh "${STAGE_ROOT}" | awk '{print $1}') @ $(date)"

ARGS=(
  --train-csv "${ANN_DIR}/HD_EPIC_train_stream_mtp.csv"
  --val-csv "${ANN_DIR}/HD_EPIC_val_stream_mtp.csv"
  --video-root "${VIDEO_ROOT}"
  --model-id "${MODEL_ID}"
  --out-dir "${OUT_DIR}"
  --horizons-sec "2,4,6"
  --loss-weights "1.0,0.7,0.5"
  --primary-horizon-sec 2.0
  --probe-num-frames 8
  --frame-size 256
  --batch-size 1
  --grad-accum 4
  --epochs 6
  --lr 2e-4
  --lora-rank 16
  --lora-alpha 32.0
  --num-workers 2
  --save-every 100
  "${LOCAL_ONLY_FLAG[@]}"
)

if [[ "${VAL_ONLY}" == "1" ]]; then
  ARGS+=(--val-only)
fi

cd "${ROOT}"
set +e
"${PYTHON}" "${ROOT}/train_stream_qwen_probe.py" "${ARGS[@]}" &
TRAIN_PID=$!
wait "${TRAIN_PID}"
train_rc=$?
TRAIN_PID=""
set -e

echo "Training exit_code=${train_rc} at $(date)"
"${PYTHON}" "${ROOT}/compare_to_jepa_vanilla.py" --qwen-dir "${OUT_DIR}" || true

if [[ "${VAL_ONLY}" == "1" ]]; then
  exit ${train_rc}
fi
if [[ -f "${DONE_FLAG}" ]]; then
  echo "Qwen2-VL-2B stream probe complete: $(date)"
  exit 0
fi
# Only auto-resubmit after clean signal/time-limit path (train_rc==0 or already flagged).
# Do NOT resubmit on hard crashes — that burns the queue in a failure loop.
if [[ "${train_rc}" -eq 0 || "${_RESUBMITTED}" == "1" ]]; then
  request_resubmit
  if [[ "${_RESUBMITTED}" == "1" ]]; then
    exit 0
  fi
else
  echo "Training crashed (rc=${train_rc}); NOT auto-resubmitting." >&2
fi
exit ${train_rc}
