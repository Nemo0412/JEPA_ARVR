#!/bin/bash
#SBATCH --job-name=p01_mtp_a1_vid
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=512GB
#SBATCH --time=01:50:00
#SBATCH --signal=B:USR1@90
#SBATCH --output=/scratch/ll5914/logs/p01_stream_mtp_a1_samehead_video_%j.out
#SBATCH --error=/scratch/ll5914/logs/p01_stream_mtp_a1_samehead_video_%j.err

set -euo pipefail

# A1 same-head contrast (video arm):
#   freeze video stream-best backbone; train MTP heads from scratch.
# Pair with CA arm; success if CA@2s − video@2s ≥ 0.3pp under identical head recipe.

PYTHON="/scratch/ll5914/conda_envs/SVD/bin/python"
PROJECT_ROOT="/home/ll5914/Jepa_yifan/JEPA_ARVR"
VJEPA_ROOT="/home/ll5914/ARVR_Video/vjepa2"
SCRIPT_PATH="${PROJECT_ROOT}/scripts/submit_p01_stream_mtp_a1_samehead_video_ll5914.slurm"
export VJEPA_ROOT
export PYTHONPATH="${PROJECT_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

VIDEO_ROOT_REMOTE="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_videos"
ANN_DIR="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/stream_half_split"
CHECKPOINT="/scratch/ll5914/models/vjepa2/vitl.pt"
VIDEO_BEST_DIR="/scratch/ll5914/experiments/p01_video_pred_joint_heads_clip/action_anticipation_frozen/p01-video-pred-joint-heads-vitl16-256-10ep"
ENCODER_LORA="${VIDEO_BEST_DIR}/encoder_lora_best.pt"
PREDICTOR_LORA="${VIDEO_BEST_DIR}/predictor_lora_best.pt"
INIT_CKPT="/scratch/ll5914/experiments/p01_stream_mtp_2_4_6/best.pt"

OUT_DIR="/scratch/ll5914/experiments/p01_stream_mtp_a1_samehead_video"
DONE_FLAG="${OUT_DIR}/TRAINING_DONE"
mkdir -p /scratch/ll5914/logs "${OUT_DIR}"

VAL_ONLY="${VAL_ONLY:-0}"
if [[ "${VAL_ONLY}" != "1" && -f "${DONE_FLAG}" ]]; then
  echo "Already done; exiting."; exit 0
fi

STAGE_ROOT="${STAGE_ROOT:-/dev/shm/${USER}/hdepic_a1_vid_${SLURM_JOB_ID:-manual}}"
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

echo "A1 video same-head | Job=${SLURM_JOB_ID:-manual} | $(date)"
for f in "${CHECKPOINT}" "${ENCODER_LORA}" "${PREDICTOR_LORA}" "${INIT_CKPT}"; do
  [[ -f "$f" ]] || { echo "missing $f"; exit 1; }
done
mkdir -p "${STAGE_ROOT}/P01"
rsync -aL "${VIDEO_ROOT_REMOTE}/P01/" "${STAGE_ROOT}/P01/"

ARGS=(
  --train-csv "${ANN_DIR}/HD_EPIC_train_stream_mtp.csv"
  --val-csv "${ANN_DIR}/HD_EPIC_val_stream_mtp.csv"
  --video-root "${VIDEO_ROOT}"
  --checkpoint "${CHECKPOINT}"
  --encoder-lora "${ENCODER_LORA}"
  --predictor-lora "${PREDICTOR_LORA}"
  --init-from-ckpt "${INIT_CKPT}"
  --heads-from-scratch
  --freeze-backbone
  --out-dir "${OUT_DIR}"
  --horizons-sec "2,4,6"
  --loss-weights "1.0,0.7,0.5"
  --primary-horizon-sec 2.0
  --anticipation-sec 2.0
  --max-frames 80 --fps 8 --img-size 256 --keep-count 4096
  --batch-size 1
  --num-workers 8
  --epochs 6
  --lr 2e-5
  --save-every 200
  --seed 0
)
[[ "${VAL_ONLY}" == "1" ]] && ARGS+=(--val-only)

cd "${PROJECT_ROOT}"
set +e
"${PYTHON}" -m app.hdepic_lora_action_anticipation.train_stream_mtp "${ARGS[@]}" &
TRAIN_PID=$!
wait "${TRAIN_PID}"; train_rc=$?; TRAIN_PID=""; set -e
echo "exit=${train_rc} @ $(date)"
[[ "${VAL_ONLY}" == "1" ]] && exit ${train_rc}
[[ -f "${DONE_FLAG}" ]] && exit 0
request_resubmit
[[ "${_RESUBMITTED}" == "1" ]] && exit 0
exit ${train_rc}
