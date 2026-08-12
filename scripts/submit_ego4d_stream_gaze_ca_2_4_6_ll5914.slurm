#!/bin/bash
#SBATCH --job-name=ego4d_gaze_ca
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=512GB
#SBATCH --time=01:50:00
#SBATCH --signal=B:USR1@90
#SBATCH --output=/scratch/ll5914/logs/ego4d_stream_gaze_ca_%j.out
#SBATCH --error=/scratch/ll5914/logs/ego4d_stream_gaze_ca_%j.err

set -euo pipefail

# Ego4D streaming CA+MTP: Gaze as K/V only (no IMU concat).
# Requires:
#   1) Ego4D AWS credentials + downloaded annotations/gaze/clips
#   2) Stream half-split CSVs under ANN_DIR (make once data is ready)
#   3) Warm video encoder/predictor LoRA + optional gaze adapter

PYTHON="/scratch/ll5914/conda_envs/SVD/bin/python"
PROJECT_ROOT="/home/ll5914/Jepa_yifan/JEPA_ARVR"
VJEPA_ROOT="/home/ll5914/ARVR_Video/vjepa2"
SCRIPT_PATH="${PROJECT_ROOT}/scripts/submit_ego4d_stream_gaze_ca_2_4_6_ll5914.slurm"
export VJEPA_ROOT
export PYTHONPATH="${PROJECT_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TRI_MODAL_GAZE_MAP_CACHE=256
export GAZE_DECODER_CACHE_READER=1
export GAZE_DECODER_NUM_THREADS=1

EGO4D_ROOT="${EGO4D_ROOT:-/scratch/ll5914/datasets/Ego4D}"
ANN_DIR="${ANN_DIR:-${EGO4D_ROOT}/stream_half_split}"
VIDEO_ROOT="${VIDEO_ROOT:-${EGO4D_ROOT}/v2/clips}"
GAZE_ROOT="${GAZE_ROOT:-${EGO4D_ROOT}/v2/gaze}"
GAZE_EXTRACT_ROOT="${GAZE_EXTRACT_ROOT:-${EGO4D_ROOT}/gaze_extract}"
CHECKPOINT="/scratch/ll5914/models/vjepa2/vitl.pt"

# Warm from HD-EPIC video stream / clip LoRA until Ego4D-specific ckpts exist.
VIDEO_BEST_DIR="${VIDEO_BEST_DIR:-/scratch/ll5914/experiments/p01_video_pred_joint_heads_clip/action_anticipation_frozen/p01-video-pred-joint-heads-vitl16-256-10ep}"
ENCODER_LORA="${ENCODER_LORA:-${VIDEO_BEST_DIR}/encoder_lora_best.pt}"
PREDICTOR_LORA="${PREDICTOR_LORA:-${VIDEO_BEST_DIR}/predictor_lora_best.pt}"
# Gaze-only adapter (4ch). Prefer an Ego4D/EGTEA gaze adapter if present; else identity init.
ADAPTER_CKPT="${ADAPTER_CKPT:-}"
FUSION_CKPT="${FUSION_CKPT:-}"  # ignored for ca_aux=gaze (trained from scratch)

OUT_DIR="${OUT_DIR:-/scratch/ll5914/experiments/ego4d_stream_gaze_ca_2_4_6}"
DONE_FLAG="${OUT_DIR}/TRAINING_DONE"
mkdir -p /scratch/ll5914/logs "${OUT_DIR}"

VAL_ONLY="${VAL_ONLY:-0}"
if [[ "${VAL_ONLY}" != "1" && -f "${DONE_FLAG}" ]]; then
  echo "Already done (${DONE_FLAG}); exiting."
  exit 0
fi

_RESUBMITTED=0
TRAIN_PID=""
request_resubmit() {
  if [[ "${VAL_ONLY}" == "1" || -f "${DONE_FLAG}" || "${_RESUBMITTED}" == "1" ]]; then return 0; fi
  _RESUBMITTED=1
  sbatch "${SCRIPT_PATH}" || echo "WARN: resubmit failed" >&2
}
on_signal() {
  echo "Caught signal — save/resubmit @ $(date)"
  request_resubmit
  if [[ -n "${TRAIN_PID}" ]] && kill -0 "${TRAIN_PID}" 2>/dev/null; then
    kill -TERM "${TRAIN_PID}" 2>/dev/null || true
  fi
}
trap on_signal USR1 TERM

echo "========================================================"
echo "Ego4D streaming CA+MTP | ca_aux=gaze (KV=gaze, no IMU)"
echo "Job=${SLURM_JOB_ID:-manual} VAL_ONLY=${VAL_ONLY}"
echo "EGO4D_ROOT=${EGO4D_ROOT}"
echo "ANN=${ANN_DIR} VIDEO=${VIDEO_ROOT}"
echo "OUT=${OUT_DIR}"
echo "Start: $(date)"
echo "========================================================"

for f in "${CHECKPOINT}" "${ENCODER_LORA}" "${PREDICTOR_LORA}"; do
  [[ -f "${f}" ]] || { echo "ERROR: missing ${f}" >&2; exit 1; }
done

if [[ ! -f "${ANN_DIR}/Ego4D_train_stream_mtp.csv" ]]; then
  cat >&2 <<EOF
ERROR: missing ${ANN_DIR}/Ego4D_train_stream_mtp.csv
1) Get AWS credentials from https://ego4d.dev/request/ego4d
2) sbatch scripts/download_ego4d_gaze_sta.slurm
3) Build stream half-split CSVs into ${ANN_DIR}
EOF
  exit 1
fi

ARGS=(
  --train-csv "${ANN_DIR}/Ego4D_train_stream_mtp.csv"
  --val-csv "${ANN_DIR}/Ego4D_val_stream_mtp.csv"
  --video-root "${VIDEO_ROOT}"
  --checkpoint "${CHECKPOINT}"
  --encoder-lora "${ENCODER_LORA}"
  --predictor-lora "${PREDICTOR_LORA}"
  --adapter-ckpt "${ADAPTER_CKPT:-/dev/null}"
  --out-dir "${OUT_DIR}"
  --gaze-root "${GAZE_ROOT}"
  --gaze-extract-root "${GAZE_EXTRACT_ROOT}"
  --ca-aux gaze
  --horizons-sec "2,4,6"
  --loss-weights "1.0,0.7,0.5"
  --primary-horizon-sec 2.0
  --anticipation-sec 2.0
  --max-frames 80
  --fps 8
  --img-size 256
  --keep-count 4096
  --prune-mode postfuse_recency
  --batch-size 1
  --num-workers 4
  --epochs 8
  --lr 2e-5
  --fusion-lr-mult 2.0
  --adapter-lr-mult 0.25
  --save-every 200
)

if [[ -n "${ADAPTER_CKPT}" && -f "${ADAPTER_CKPT}" ]]; then
  :
else
  # Placeholder path → build_concat_ca_model identity-inits adapter.
  ARGS=( "${ARGS[@]/ --adapter-ckpt "/tmp/no_adapter.pt" )
fi

if [[ "${VAL_ONLY}" == "1" ]]; then
  [[ -f "${OUT_DIR}/best.pt" ]] && ARGS+=(--init-from-ckpt "${OUT_DIR}/best.pt")
  ARGS+=(--val-only)
fi

cd "${PROJECT_ROOT}"
set +e
"${PYTHON}" -m app.hdepic_lora_action_anticipation.train_stream_mtp_concat_ca "${ARGS[@]}" &
TRAIN_PID=$!
wait "${TRAIN_PID}"
train_rc=$?
TRAIN_PID=""
set -e

echo "exit_code=${train_rc} at $(date)"
if [[ "${VAL_ONLY}" == "1" ]]; then exit ${train_rc}; fi
if [[ -f "${DONE_FLAG}" ]]; then echo "Ego4D gaze CA done"; exit 0; fi
request_resubmit
[[ "${_RESUBMITTED}" == "1" ]] && exit 0
exit ${train_rc}
