#!/bin/bash
#SBATCH --job-name=p01_gaze_ca
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=512GB
#SBATCH --time=01:50:00
#SBATCH --signal=B:USR1@90
#SBATCH --output=/scratch/ll5914/logs/p01_stream_gaze_ca_%j.out
#SBATCH --error=/scratch/ll5914/logs/p01_stream_gaze_ca_%j.err

set -euo pipefail

# HD-EPIC P01 ablation: same streaming CA+MTP protocol as concat+CA, but
# --ca-aux gaze (KV=gaze only, no IMU concat). Useful before Ego4D data arrives.

PYTHON="/scratch/ll5914/conda_envs/SVD/bin/python"
PROJECT_ROOT="/home/ll5914/Jepa_yifan/JEPA_ARVR"
VJEPA_ROOT="/home/ll5914/ARVR_Video/vjepa2"
SCRIPT_PATH="${PROJECT_ROOT}/scripts/submit_p01_stream_gaze_ca_2_4_6_ll5914.slurm"
export VJEPA_ROOT
export PYTHONPATH="${PROJECT_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TRI_MODAL_GAZE_MAP_CACHE=256
export GAZE_DECODER_CACHE_READER=1
export GAZE_DECODER_NUM_THREADS=1

VIDEO_ROOT_REMOTE="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_videos"
ANN_DIR="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/stream_half_split"
CHECKPOINT="/scratch/ll5914/models/vjepa2/vitl.pt"

CA_RUN="/scratch/ll5914/experiments/concat_plus_cross_attn_v2/action_anticipation_frozen/concat-plus-ca-v2-L3-keepaux-softgate-vitl16-256-12ep-1xh100"
ENCODER_LORA="${CA_RUN}/encoder_lora_best.pt"
PREDICTOR_LORA="/scratch/ll5914/experiments/p01_stream_mtp_2_4_6/predictor_lora_from_stream_best.pt"
# 5ch adapter is incompatible with gaze-only 4ch — leave empty → identity init.
ADAPTER_CKPT=""

GAZE_EXTRACT_ROOT="/scratch/ll5914/datasets/HD-EPIC/_gaze_extract"
GAZE_ROOT="/scratch/ll5914/datasets/HD-EPIC/SLAM-and-Gaze"
GAZE_SYNC_ROOT="/scratch/ll5914/datasets/HD-EPIC/HD-EPIC/Videos"

OUT_DIR="/scratch/ll5914/experiments/p01_stream_gaze_ca_2_4_6"
TICK_CACHE_DIR="/scratch/ll5914/datasets/HD-EPIC/caches/stream_gaze_ca_ticks_256"
DONE_FLAG="${OUT_DIR}/TRAINING_DONE"
mkdir -p /scratch/ll5914/logs "${OUT_DIR}" "${TICK_CACHE_DIR}"

VAL_ONLY="${VAL_ONLY:-0}"
if [[ "${VAL_ONLY}" != "1" && -f "${DONE_FLAG}" ]]; then
  echo "Already done; exiting."; exit 0
fi

STAGE_ROOT="${STAGE_ROOT:-/dev/shm/${USER}/hdepic_p01_gaze_ca_${SLURM_JOB_ID:-manual}}"
VIDEO_ROOT="${STAGE_ROOT}"
_RESUBMITTED=0
TRAIN_PID=""
request_resubmit() {
  if [[ "${VAL_ONLY}" == "1" || -f "${DONE_FLAG}" || "${_RESUBMITTED}" == "1" ]]; then return 0; fi
  _RESUBMITTED=1
  sbatch "${SCRIPT_PATH}" || echo "WARN: resubmit failed" >&2
}
cleanup_stage() { rm -rf "${STAGE_ROOT}" 2>/dev/null || true; }
on_signal() {
  echo "Caught signal — save/resubmit @ $(date)"
  request_resubmit
  if [[ -n "${TRAIN_PID}" ]] && kill -0 "${TRAIN_PID}" 2>/dev/null; then kill -TERM "${TRAIN_PID}" 2>/dev/null || true; fi
}
trap cleanup_stage EXIT
trap on_signal USR1 TERM

echo "P01 gaze-only CA (KV=gaze, no IMU) | Job=${SLURM_JOB_ID:-manual} | $(date)"
for f in "${CHECKPOINT}" "${ENCODER_LORA}" "${PREDICTOR_LORA}"; do
  [[ -f "$f" ]] || { echo "missing $f"; exit 1; }
done
[[ -f "${ANN_DIR}/HD_EPIC_train_stream_mtp.csv" ]] || \
  "${PYTHON}" "${PROJECT_ROOT}/scripts/make_hdepic_stream_half_split.py" --out "${ANN_DIR}"

mkdir -p "${STAGE_ROOT}/P01"
rsync -aL "${VIDEO_ROOT_REMOTE}/P01/" "${STAGE_ROOT}/P01/"

ARGS=(
  --train-csv "${ANN_DIR}/HD_EPIC_train_stream_mtp.csv"
  --val-csv "${ANN_DIR}/HD_EPIC_val_stream_mtp.csv"
  --video-root "${VIDEO_ROOT}"
  --checkpoint "${CHECKPOINT}"
  --encoder-lora "${ENCODER_LORA}"
  --predictor-lora "${PREDICTOR_LORA}"
  --adapter-ckpt "/tmp/no_gaze_adapter.pt"
  --out-dir "${OUT_DIR}"
  --gaze-root "${GAZE_ROOT}"
  --gaze-extract-root "${GAZE_EXTRACT_ROOT}"
  --gaze-sync-root "${GAZE_SYNC_ROOT}"
  --ca-aux gaze
  --horizons-sec "2,4,6"
  --loss-weights "1.0,0.7,0.5"
  --primary-horizon-sec 2.0
  --anticipation-sec 2.0
  --max-frames 80 --fps 8 --img-size 256 --keep-count 4096
  --prune-mode postfuse_recency
  --batch-size 1
  --batch-size-by-frames "32:4,48:2,64:2,80:1"
  --num-workers 4
  --epochs 8
  --lr 2e-5
  --fusion-lr-mult 2.0
  --adapter-lr-mult 1.0
  --tick-cache-dir "${TICK_CACHE_DIR}"
  --save-every 200
)
[[ "${VAL_ONLY}" == "1" ]] && ARGS+=(--val-only --init-from-ckpt "${OUT_DIR}/best.pt")

cd "${PROJECT_ROOT}"
set +e
"${PYTHON}" -m app.hdepic_lora_action_anticipation.train_stream_mtp_concat_ca "${ARGS[@]}" &
TRAIN_PID=$!
wait "${TRAIN_PID}"
rc=$?
TRAIN_PID=""
set -e
echo "exit=${rc} @ $(date)"
[[ "${VAL_ONLY}" == "1" ]] && exit ${rc}
[[ -f "${DONE_FLAG}" ]] && exit 0
request_resubmit
[[ "${_RESUBMITTED}" == "1" ]] && exit 0
exit ${rc}
