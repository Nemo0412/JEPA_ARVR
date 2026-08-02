#!/bin/bash
#SBATCH --job-name=p01_qwen_style
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=512GB
#SBATCH --time=01:50:00
#SBATCH --signal=B:USR1@90
#SBATCH --output=/scratch/ll5914/logs/p01_qwen_style_%j.out
#SBATCH --error=/scratch/ll5914/logs/p01_qwen_style_%j.err

set -euo pipefail

# V-JEPA ViT-L (frozen) + Qwen-style decoder-only AR with KV cache on val.
# Same stream MTP protocol as vanilla JEPA / Exp A.

PYTHON="/scratch/ll5914/conda_envs/SVD/bin/python"
PROJECT_ROOT="/home/ll5914/Jepa_yifan/JEPA_ARVR"
VJEPA_ROOT="/home/ll5914/ARVR_Video/vjepa2"
ROOT="${PROJECT_ROOT}/baselines/jepa_qwen_style_decoder"
SCRIPT_PATH="${ROOT}/submit_qwen_style_decoder.slurm"
export VJEPA_ROOT
export PYTHONPATH="${PROJECT_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"

VIDEO_ROOT_REMOTE="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_videos"
ANN_DIR="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/stream_half_split"
CHECKPOINT="/scratch/ll5914/models/vjepa2/vitl.pt"
ENCODER_LORA="/scratch/ll5914/experiments/p01_video_pred_joint_heads_clip/action_anticipation_frozen/p01-video-pred-joint-heads-vitl16-256-10ep/encoder_lora_best.pt"
OUT_DIR="/scratch/ll5914/experiments/p01_stream_qwen_style_decoder_2_4_6"
DONE_FLAG="${OUT_DIR}/TRAINING_DONE"
mkdir -p /scratch/ll5914/logs "${OUT_DIR}"

VAL_ONLY="${VAL_ONLY:-0}"
if [[ "${VAL_ONLY}" != "1" && -f "${DONE_FLAG}" ]]; then echo "done"; exit 0; fi

STAGE_ROOT="${STAGE_ROOT:-/dev/shm/${USER}/hdepic_p01_qstyle_${SLURM_JOB_ID:-manual}}"
VIDEO_ROOT="${STAGE_ROOT}"
_RESUBMITTED=0
TRAIN_PID=""
request_resubmit() {
  if [[ "${VAL_ONLY}" == "1" || -f "${DONE_FLAG}" || "${_RESUBMITTED}" == "1" ]]; then return 0; fi
  _RESUBMITTED=1; sbatch "${SCRIPT_PATH}" || true
}
cleanup_stage() { [[ -n "${STAGE_ROOT:-}" && -d "${STAGE_ROOT}" ]] && rm -rf "${STAGE_ROOT}" || true; }
on_signal() {
  request_resubmit
  [[ -n "${TRAIN_PID}" ]] && kill -0 "${TRAIN_PID}" 2>/dev/null && kill -TERM "${TRAIN_PID}" 2>/dev/null || true
}
trap cleanup_stage EXIT
trap on_signal USR1 TERM

echo "=== qwen-style decoder Job=${SLURM_JOB_ID:-manual} $(date) ==="
mkdir -p "${STAGE_ROOT}/P01"
rsync -aL "${VIDEO_ROOT_REMOTE}/P01/" "${STAGE_ROOT}/P01/"

ARGS=(
  --train-csv "${ANN_DIR}/HD_EPIC_train_stream_mtp.csv"
  --val-csv "${ANN_DIR}/HD_EPIC_val_stream_mtp.csv"
  --video-root "${VIDEO_ROOT}"
  --checkpoint "${CHECKPOINT}"
  --encoder-lora "${ENCODER_LORA}"
  --out-dir "${OUT_DIR}"
  --horizons-sec "2,4,6" --loss-weights "1.0,0.7,0.5" --primary-horizon-sec 2.0
  --anticipation-sec 2.0 --max-frames 80 --fps 8 --img-size 256 --keep-count 4096
  --batch-size 1 --num-workers 2 --epochs 8 --lr 2e-5 --save-every 200
  --decoder-depth 12 --decoder-dim 384 --decoder-heads 12
  --num-future-tokens 16
  --train-decode-mode teacher
  --val-decode-mode ar_kv
  --aux-ar-weight 0.1
)
[[ "${VAL_ONLY}" == "1" ]] && ARGS+=(--val-only)

cd "${PROJECT_ROOT}"
set +e
"${PYTHON}" "${ROOT}/train_stream_qwen_style.py" "${ARGS[@]}" &
TRAIN_PID=$!; wait "${TRAIN_PID}"; train_rc=$?; TRAIN_PID=""; set -e
[[ -f "${DONE_FLAG}" ]] && exit 0
request_resubmit
[[ "${_RESUBMITTED}" == "1" ]] && exit 0
exit ${train_rc}
