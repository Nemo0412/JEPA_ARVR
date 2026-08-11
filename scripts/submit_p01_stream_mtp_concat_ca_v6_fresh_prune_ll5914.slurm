#!/bin/bash
#SBATCH --job-name=p01_mtp_ca_v6f
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=512GB
#SBATCH --time=01:50:00
#SBATCH --signal=B:USR1@90
#SBATCH --output=/scratch/ll5914/logs/p01_stream_mtp_concat_ca_v6_fresh_prune_%j.out
#SBATCH --error=/scratch/ll5914/logs/p01_stream_mtp_concat_ca_v6_fresh_prune_%j.err

set -euo pipefail

# Streaming MTP + concat+CA v6 — fresh train with corrected prune
#
# Why v5 failed: init from v3 best (trained under encoder_attn prune) then
# switched to postfuse_recency → token distribution mismatch; val fell to ~23.9%.
#
# v6 = v3 winning recipe FROM SCRATCH, but prune aligned from epoch 0:
#   - freeze adapter+fusion (CA 43.92% features)
#   - warm predictor LoRA + MTP heads from video-only stream (25.88%)
#   - --prune-mode postfuse_recency + --recent-keep-sec 2.0
#     (hard-keep last 2s; score-select older tokens into remaining budget)
# Target: beat video-only 25.88%@2s (v3 was 25.08%).

PYTHON="/scratch/ll5914/conda_envs/SVD/bin/python"
PROJECT_ROOT="/home/ll5914/Jepa_yifan/JEPA_ARVR"
VJEPA_ROOT="/home/ll5914/ARVR_Video/vjepa2"
SCRIPT_PATH="${PROJECT_ROOT}/scripts/submit_p01_stream_mtp_concat_ca_v6_fresh_prune_ll5914.slurm"
export VJEPA_ROOT
export PYTHONPATH="${PROJECT_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
export TRI_MODAL_GAZE_MAP_CACHE=256
export GAZE_DECODER_CACHE_READER=1
export GAZE_DECODER_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

VIDEO_ROOT_REMOTE="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_videos"
ANN_DIR="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/stream_half_split"
CHECKPOINT="/scratch/ll5914/models/vjepa2/vitl.pt"

CA_RUN="/scratch/ll5914/experiments/concat_plus_cross_attn_v2/action_anticipation_frozen/concat-plus-ca-v2-L3-keepaux-softgate-vitl16-256-12ep-1xh100"
ENCODER_LORA="${CA_RUN}/encoder_lora_best.pt"
PREDICTOR_LORA="/scratch/ll5914/experiments/p01_stream_mtp_2_4_6/predictor_lora_from_stream_best.pt"
MTP_WARM_CKPT="/scratch/ll5914/experiments/p01_stream_mtp_2_4_6/mtp_classifier_best.pt"
ADAPTER_CKPT="${CA_RUN}/binary_input_adapter_best.pt"
FUSION_CKPT="${CA_RUN}/tri_modal_fusion_best.pt"

GAZE_EXTRACT_ROOT="/scratch/ll5914/datasets/HD-EPIC/_gaze_extract"
GAZE_ROOT="/scratch/ll5914/datasets/HD-EPIC/SLAM-and-Gaze"
GAZE_SYNC_ROOT="/scratch/ll5914/datasets/HD-EPIC/HD-EPIC/Videos"
POSE_SLAM_ROOT="/scratch/ll5914/datasets/HD-EPIC/SLAM-and-Gaze/P01/SLAM/multi"
POSE_MAPPING_JSON="${POSE_SLAM_ROOT}/vrs_to_multi_slam.json"

OUT_DIR="/scratch/ll5914/experiments/p01_stream_mtp_concat_ca_v6_fresh_prune_2_4_6"
TICK_CACHE_DIR="/scratch/ll5914/datasets/HD-EPIC/caches/stream_mtp_concat_ca_ticks_256"
DONE_FLAG="${OUT_DIR}/TRAINING_DONE"
mkdir -p /scratch/ll5914/logs "${OUT_DIR}" "${ANN_DIR}" "${TICK_CACHE_DIR}"

VAL_ONLY="${VAL_ONLY:-0}"
if [[ "${VAL_ONLY}" != "1" && -f "${DONE_FLAG}" ]]; then
  echo "Already done (${DONE_FLAG}); exiting."
  exit 0
fi

STAGE_ROOT="${STAGE_ROOT:-/dev/shm/${USER}/hdepic_p01_mtp_ca_v6f_${SLURM_JOB_ID:-manual}}"
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
echo "P01 stream MTP concat+CA v6 fresh prune (postfuse+recent2s)"
echo "Job=${SLURM_JOB_ID:-manual} VAL_ONLY=${VAL_ONLY}"
echo "v3 recipe from scratch + aligned prune (NOT init from v3 best)"
echo "Target: beat video-only 25.88%@2s (v3=25.08%, v5 failed)"
echo "OUT=${OUT_DIR}"
echo "Start: $(date)"
echo "========================================================"

for f in "${CHECKPOINT}" "${ENCODER_LORA}" "${PREDICTOR_LORA}" "${MTP_WARM_CKPT}" "${ADAPTER_CKPT}" "${FUSION_CKPT}" "${POSE_MAPPING_JSON}"; do
  if [[ ! -f "${f}" ]]; then
    echo "ERROR: missing ${f}" >&2
    exit 1
  fi
done

if [[ ! -f "${ANN_DIR}/HD_EPIC_train_stream_mtp.csv" ]]; then
  "${PYTHON}" "${PROJECT_ROOT}/scripts/make_hdepic_stream_half_split.py" --out "${ANN_DIR}"
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
  --predictor-lora "${PREDICTOR_LORA}"
  --adapter-ckpt "${ADAPTER_CKPT}"
  --fusion-ckpt "${FUSION_CKPT}"
  --mtp-warm-ckpt "${MTP_WARM_CKPT}"
  --out-dir "${OUT_DIR}"
  --gaze-root "${GAZE_ROOT}"
  --gaze-extract-root "${GAZE_EXTRACT_ROOT}"
  --gaze-sync-root "${GAZE_SYNC_ROOT}"
  --pose-slam-root "${POSE_SLAM_ROOT}"
  --pose-mapping-json "${POSE_MAPPING_JSON}"
  --horizons-sec "2,4,6"
  --loss-weights "1.0,0.7,0.5"
  --primary-horizon-sec 2.0
  --anticipation-sec 2.0
  --max-frames 80
  --fps 8
  --img-size 256
  --keep-count 4096
  --prune-mode postfuse_recency
  --recency-strength 1.0
  --recent-keep-sec 2.0
  --batch-size 1
  --batch-size-by-frames "32:4,48:2,64:2,80:1"
  --num-workers 8
  --epochs 8
  --lr 1e-5
  --head-lr-mult 0.5
  --weight-decay 1e-3
  --fusion-lr-mult 1.0
  --adapter-lr-mult 0.25
  --save-every 200
  --freeze-adapter
  --freeze-fusion
  --tick-cache-dir "${TICK_CACHE_DIR}"
  --early-stop-patience 3
)

if [[ "${VAL_ONLY}" == "1" ]]; then
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

echo "Training exit_code=${train_rc} at $(date)"

if [[ "${VAL_ONLY}" == "1" ]]; then
  exit ${train_rc}
fi

if [[ -f "${DONE_FLAG}" ]]; then
  echo "Streaming MTP concat+CA v6 fresh prune complete: $(date)"
  exit 0
fi

request_resubmit
if [[ "${_RESUBMITTED}" == "1" ]]; then
  exit 0
fi
exit ${train_rc}
