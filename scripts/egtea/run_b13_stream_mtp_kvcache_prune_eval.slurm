#!/bin/bash
# B13 frozen EGTEA streaming-MTP KV-cache-prune eval.
# Loads the video-only MTP parent-best and runs native Action Top-5 @ +2/+4/+6 s
# under one post-encoder memory-prune STRATEGY. No finetune (val-only).
#
# Code defaults to the integrated main project tree; CODE_ROOT remains
# overridable for isolated reproduction. Data/checkpoints/overlay are read-only.
#
# Env knobs: STRATEGY={attention|recent|none|loss_aware} KEEP_COUNT FPS
# MAX_FRAMES PRUNE_CONFIG MODE={smoke|full}.
#SBATCH --job-name=VJEPA2-EXP__b13_kvcache_prune_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=384G
#SBATCH --time=01:30:00

set -euo pipefail
SHARE_GPU="${SHARE_GPU:-1}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"

DATA_ROOT="${DATA_ROOT:-${DATA_ROOT}}"
CODE_ROOT="${CODE_ROOT:-${DATA_ROOT}}"
export VJEPA_ROOT

MODE="${MODE:-smoke}"
STRATEGY="${STRATEGY:-attention}"
KEEP_COUNT="${KEEP_COUNT:-4096}"
FPS="${FPS:-8}"
MAX_FRAMES="${MAX_FRAMES:-80}"
PRUNE_CONFIG="${PRUNE_CONFIG:-}"
HYBRID_WEIGHT="${HYBRID_WEIGHT:-0.5}"
HYBRID_PRIOR_LAYER="${HYBRID_PRIOR_LAYER:--1}"
COVERAGE_SEED="${COVERAGE_SEED:-recent}"
COVERAGE_OBJECTIVE="${COVERAGE_OBJECTIVE:-kcenter}"
JITTER_DROP_FRAC="${JITTER_DROP_FRAC:-0.05}"
JITTER_SEED="${JITTER_SEED:-0}"
POSITION_MODE="${POSITION_MODE:-rebase}"
VAL_SUBSET_N="${VAL_SUBSET_N:-0}"
VAL_SUBSET_SEED="${VAL_SUBSET_SEED:-0}"
ONLY_CONTEXT_SEC="${ONLY_CONTEXT_SEC:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"

ANN_DIR="${ANN_DIR:-${DATA_ROOT}/data/egtea/vjepa_annotations/stream_half_split/split1}"
VIDEO_ROOT="${VIDEO_ROOT:-${DATA_ROOT}/data/egtea/stream_session_videos}"
CHECKPOINT="${CHECKPOINT:-${DATA_ROOT}/checkpoints/vitl.pt}"
ENCODER_LORA="${ENCODER_LORA:-${DATA_ROOT}/outputs/egtea_split1_video_enc_ll_exact/action_anticipation_frozen/egtea-split1-video-enc-fixed-vitl16-256-10ep/encoder_lora_best.pt}"
PREDICTOR_LORA="${PREDICTOR_LORA:-${DATA_ROOT}/outputs/egtea_split1_video_pred_joint_heads_ll_exact/action_anticipation_frozen/egtea-split1-video-pred-joint-heads-vitl16-256-10ep/predictor_lora_best.pt}"
PARENT_CKPT="${PARENT_CKPT:-${DATA_ROOT}/outputs/egtea_stream_mtp/action_anticipation_frozen/egtea-split1-stream-mtp-2-4-6-vitl256-llwarm/best.pt}"
OUT_ROOT="${OUT_ROOT:-${DATA_ROOT}/outputs/egtea_stream_mtp_kvcache_prune}"

# Overridable so Axis 4 can evaluate a regenerated window-specific validation
# split while preserving the original train vocabulary.
TRAIN_CSV="${TRAIN_CSV:-${ANN_DIR}/EGTEA_train_stream_mtp.csv}"
VAL_CSV="${VAL_CSV:-${ANN_DIR}/EGTEA_val_stream_mtp.csv}"

mkdir -p "${OUT_ROOT}"
for path in "${CHECKPOINT}" "${ENCODER_LORA}" "${PREDICTOR_LORA}" "${PARENT_CKPT}" \
  "${TRAIN_CSV}" "${VAL_CSV}"; do
  [[ -f "${path}" ]] || { echo "ERROR: missing required file: ${path}" >&2; exit 2; }
done
[[ -d "${VIDEO_ROOT}" ]] || { echo "ERROR: missing video root: ${VIDEO_ROOT}" >&2; exit 2; }
[[ -f "${CODE_ROOT}/app/hdepic_lora_action_anticipation/eval_stream_mtp_kvcache_prune.py" ]] \
  || { echo "ERROR: eval harness missing in CODE_ROOT" >&2; exit 2; }
if [[ "${STRATEGY}" == "loss_aware" || "${STRATEGY}" == "hybrid" ]]; then
  [[ -n "${PRUNE_CONFIG}" && -f "${PRUNE_CONFIG}" ]] \
    || { echo "ERROR: ${STRATEGY} requires PRUNE_CONFIG (calibrated JSON)" >&2; exit 2; }
fi

MAX_VAL_BATCHES=0
WORKERS=4
TAG="${STRATEGY}-fps${FPS}-keep${KEEP_COUNT}${WINDOW_TAG:+-${WINDOW_TAG}}"
if [[ "${STRATEGY}" == "hybrid" ]]; then TAG="${STRATEGY}-w${HYBRID_WEIGHT}-fps${FPS}-keep${KEEP_COUNT}${WINDOW_TAG:+-${WINDOW_TAG}}"; fi
if [[ "${STRATEGY}" == "coverage" ]]; then TAG="${STRATEGY}-${COVERAGE_OBJECTIVE}-${COVERAGE_SEED}-fps${FPS}-keep${KEEP_COUNT}${WINDOW_TAG:+-${WINDOW_TAG}}"; fi
if [[ "${STRATEGY}" == "recent_jitter" ]]; then TAG="${STRATEGY}-d${JITTER_DROP_FRAC}-s${JITTER_SEED}-fps${FPS}-keep${KEEP_COUNT}${WINDOW_TAG:+-${WINDOW_TAG}}"; fi
if [[ "${POSITION_MODE}" != "rebase" ]]; then TAG="${TAG}-pos_${POSITION_MODE}"; fi
if [[ "${VAL_SUBSET_N}" != "0" ]]; then TAG="${TAG}-sub${VAL_SUBSET_N}s${VAL_SUBSET_SEED}"; fi
if [[ "${MODE}" == "smoke" ]]; then
  MAX_VAL_BATCHES=2
  WORKERS=0
  TAG="${TAG}-smoke"
elif [[ "${MODE}" != "full" ]]; then
  echo "ERROR: MODE must be smoke or full, got ${MODE}" >&2; exit 2
fi
OUT_JSON="${OUT_ROOT}/${TAG}.json"

export DATA_ROOT CODE_ROOT VJEPA_ROOT PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TRAIN_CSV VAL_CSV VIDEO_ROOT CHECKPOINT ENCODER_LORA PREDICTOR_LORA PARENT_CKPT
export STRATEGY KEEP_COUNT FPS MAX_FRAMES PRUNE_CONFIG MAX_VAL_BATCHES WORKERS OUT_JSON ONLY_CONTEXT_SEC BATCH_SIZE
export HYBRID_WEIGHT HYBRID_PRIOR_LAYER COVERAGE_SEED COVERAGE_OBJECTIVE JITTER_DROP_FRAC JITTER_SEED POSITION_MODE VAL_SUBSET_N VAL_SUBSET_SEED

source "${CODE_ROOT}/scripts/egtea/gpu_monitor.sh"
start_gpu_monitor "${DATA_ROOT}/logs/b13_kvcache_prune_eval_${SLURM_JOB_ID}.gpu.csv" 30
trap stop_gpu_monitor EXIT INT TERM

share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "$CODE_ROOT"
export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
EXTRA=()
if [[ -n "${PRUNE_CONFIG}" ]]; then EXTRA+=(--prune-config "${PRUNE_CONFIG}"); fi
if [[ "${STRATEGY}" == "hybrid" ]]; then EXTRA+=(--hybrid-weight "${HYBRID_WEIGHT}" --hybrid-prior-layer "${HYBRID_PRIOR_LAYER}"); fi
if [[ "${STRATEGY}" == "coverage" ]]; then EXTRA+=(--coverage-seed "${COVERAGE_SEED}" --coverage-objective "${COVERAGE_OBJECTIVE}"); fi
if [[ "${STRATEGY}" == "recent_jitter" ]]; then EXTRA+=(--jitter-drop-frac "${JITTER_DROP_FRAC}" --jitter-seed "${JITTER_SEED}"); fi
if [[ "${POSITION_MODE}" != "rebase" ]]; then EXTRA+=(--position-mode "${POSITION_MODE}"); fi
if [[ "${VAL_SUBSET_N}" != "0" ]]; then EXTRA+=(--val-subset-n "${VAL_SUBSET_N}" --val-subset-seed "${VAL_SUBSET_SEED}"); fi
python -m app.hdepic_lora_action_anticipation.eval_stream_mtp_kvcache_prune \
  --train-csv "$TRAIN_CSV" --val-csv "$VAL_CSV" --video-root "$VIDEO_ROOT" \
  --checkpoint "$CHECKPOINT" --init-from-ckpt "$PARENT_CKPT" \
  --encoder-lora "$ENCODER_LORA" --predictor-lora "$PREDICTOR_LORA" \
  --prune-strategy "$STRATEGY" --keep-count "$KEEP_COUNT" \
  --fps "$FPS" --max-frames "$MAX_FRAMES" --img-size 256 \
  --horizons-sec 2,4,6 --primary-horizon-sec 2 --anticipation-sec 2 \
  --batch-size "$BATCH_SIZE" --num-workers "$WORKERS" --max-val-batches "$MAX_VAL_BATCHES" \
  --only-context-sec "$ONLY_CONTEXT_SEC" \
  --out-json "$OUT_JSON" "${EXTRA[@]}"
EOF

stop_gpu_monitor
trap - EXIT INT TERM
export GPU_MONITOR_INTERVAL
report_gpu_monitor

echo "B13 kvcache-prune eval ${MODE} completed: ${OUT_JSON}"
