#!/bin/bash
#SBATCH --job-name=p01_mtp_ca_b1g
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=512GB
#SBATCH --time=01:50:00
#SBATCH --signal=B:USR1@90
#SBATCH --output=/scratch/ll5914/logs/p01_stream_mtp_concat_ca_b1_gateonly_%j.out
#SBATCH --error=/scratch/ll5914/logs/p01_stream_mtp_concat_ca_b1_gateonly_%j.err

set -euo pipefail

# B1: gate-only FT from v3 best (25.08%) — light multimodal adapt without full fusion.
# Requires --train-gates-only support in train_stream_mtp_concat_ca.py.
# Target: beat video 25.88%@2s.

PYTHON="/scratch/ll5914/conda_envs/SVD/bin/python"
PROJECT_ROOT="/home/ll5914/Jepa_yifan/JEPA_ARVR"
VJEPA_ROOT="/home/ll5914/ARVR_Video/vjepa2"
SCRIPT_PATH="${PROJECT_ROOT}/scripts/submit_p01_stream_mtp_concat_ca_b1_gateonly_ll5914.slurm"
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
ADAPTER_CKPT="${CA_RUN}/binary_input_adapter_best.pt"
FUSION_CKPT="${CA_RUN}/tri_modal_fusion_best.pt"
V3_BEST="/scratch/ll5914/experiments/p01_stream_mtp_concat_ca_v3_hyb_2_4_6/best.pt"

GAZE_EXTRACT_ROOT="/scratch/ll5914/datasets/HD-EPIC/_gaze_extract"
GAZE_ROOT="/scratch/ll5914/datasets/HD-EPIC/SLAM-and-Gaze"
GAZE_SYNC_ROOT="/scratch/ll5914/datasets/HD-EPIC/HD-EPIC/Videos"
POSE_SLAM_ROOT="/scratch/ll5914/datasets/HD-EPIC/SLAM-and-Gaze/P01/SLAM/multi"
POSE_MAPPING_JSON="${POSE_SLAM_ROOT}/vrs_to_multi_slam.json"

OUT_DIR="/scratch/ll5914/experiments/p01_stream_mtp_concat_ca_b1_gateonly_2_4_6"
TICK_CACHE_DIR="/scratch/ll5914/datasets/HD-EPIC/caches/stream_mtp_concat_ca_ticks_256"
DONE_FLAG="${OUT_DIR}/TRAINING_DONE"
mkdir -p /scratch/ll5914/logs "${OUT_DIR}" "${TICK_CACHE_DIR}"

VAL_ONLY="${VAL_ONLY:-0}"
if [[ "${VAL_ONLY}" != "1" && -f "${DONE_FLAG}" ]]; then
  echo "Already done; exiting."; exit 0
fi

STAGE_ROOT="${STAGE_ROOT:-/dev/shm/${USER}/hdepic_p01_mtp_ca_b1g_${SLURM_JOB_ID:-manual}}"
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

echo "B1 gate-only FT from v3 | Job=${SLURM_JOB_ID:-manual} | $(date)"
for f in "${CHECKPOINT}" "${ENCODER_LORA}" "${PREDICTOR_LORA}" "${ADAPTER_CKPT}" "${FUSION_CKPT}" "${V3_BEST}" "${POSE_MAPPING_JSON}"; do
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
  --adapter-ckpt "${ADAPTER_CKPT}"
  --fusion-ckpt "${FUSION_CKPT}"
  --init-from-ckpt "${V3_BEST}"
  --carry-best
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
  --max-frames 80 --fps 8 --img-size 256 --keep-count 4096
  --prune-mode encoder_attn
  --batch-size 1
  --batch-size-by-frames "32:4,48:2,64:2,80:1"
  --num-workers 8
  --epochs 6
  --lr 5e-6
  --head-lr-mult 0.25
  --weight-decay 1e-3
  --fusion-lr-mult 0.05
  --save-every 200
  --freeze-adapter
  --train-gates-only
  --tick-cache-dir "${TICK_CACHE_DIR}"
  --early-stop-patience 3
)
[[ "${VAL_ONLY}" == "1" ]] && ARGS+=(--val-only)

cd "${PROJECT_ROOT}"
set +e
"${PYTHON}" -m app.hdepic_lora_action_anticipation.train_stream_mtp_concat_ca "${ARGS[@]}" &
TRAIN_PID=$!
wait "${TRAIN_PID}"; train_rc=$?; TRAIN_PID=""; set -e
echo "exit=${train_rc} @ $(date)"
[[ "${VAL_ONLY}" == "1" ]] && exit ${train_rc}
[[ -f "${DONE_FLAG}" ]] && exit 0
request_resubmit
[[ "${_RESUBMITTED}" == "1" ]] && exit 0
exit ${train_rc}
