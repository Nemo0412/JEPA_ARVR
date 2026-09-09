#!/bin/bash
# [ATTN-CORNER-SINK] B18 follow-up: what does dropping ONLY the corner-sink tokens
# cost downstream? Frozen EGTEA streaming-MTP, restricted to 4 s context clips.
# A/Bs three strategies on the SAME clips through the SAME pruning pipeline:
#   * none               -- full 4 s memory (upper bound)
#   * corner             -- drop the 4 spatial-corner cells (ring x ring) per slot
#   * corner_random_ctrl -- drop the SAME count of random cells per slot (control)
# corner-vs-random isolates whether the sink tokens are specifically informative;
# {corner,random}-vs-none shows the combined count+position-rebasing cost.
#
# Env knobs: MODE={smoke|full} RING (corner ring width) SEED (random ctrl).
#SBATCH --job-name=VJEPA2-EXP__b18_corner_prune_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=384G
#SBATCH --time=03:00:00

set -euo pipefail
SHARE_GPU="${SHARE_GPU:-1}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"

DATA_ROOT="${DATA_ROOT:-${DATA_ROOT}}"
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}}"
export VJEPA_ROOT

MODE="${MODE:-smoke}"
SEED="${SEED:-0}"
FPS="${FPS:-8}"
MAX_FRAMES="${MAX_FRAMES:-80}"
KEEP_COUNT="${KEEP_COUNT:-4096}"     # unused by corner/none; kept for harness parity
ONLY_CONTEXT_SEC="${ONLY_CONTEXT_SEC:-4}"    # 4 => the small 4s diagnostic subset; 10 => streaming set
CONTEXT_CROP_SEC="${CONTEXT_CROP_SEC:-0}"    # >0 => crop each clip to newest N s (e.g. 4 for streaming->4s)
CORNER_MASK_BLOCKS="${CORNER_MASK_BLOCKS:-all}"   # corner_keymask: which blocks mask sink key-cols (all|last|idx list)
SINK_THRESH_MULT="${SINK_THRESH_MULT:-4.0}"       # attention_exclude_sink: exclude imp > mult*uniform (offline-calibrated)
BATCH_SIZE="${BATCH_SIZE:-1}"
# space-separated strategy:ring pairs. ring ignored for none/recent.
RUNS="${RUNS:-none:0 corner:1 corner_random_ctrl:1}"

ANN_DIR="${ANN_DIR:-${DATA_ROOT}/data/egtea/vjepa_annotations/stream_half_split/split1}"
VIDEO_ROOT="${VIDEO_ROOT:-${DATA_ROOT}/data/egtea/stream_session_videos}"
CHECKPOINT="${CHECKPOINT:-${DATA_ROOT}/checkpoints/vitl.pt}"
ENCODER_LORA="${ENCODER_LORA:-${DATA_ROOT}/outputs/egtea_split1_video_enc_ll_exact/action_anticipation_frozen/egtea-split1-video-enc-fixed-vitl16-256-10ep/encoder_lora_best.pt}"
PREDICTOR_LORA="${PREDICTOR_LORA:-${DATA_ROOT}/outputs/egtea_split1_video_pred_joint_heads_ll_exact/action_anticipation_frozen/egtea-split1-video-pred-joint-heads-vitl16-256-10ep/predictor_lora_best.pt}"
PARENT_CKPT="${PARENT_CKPT:-${DATA_ROOT}/outputs/egtea_stream_mtp/action_anticipation_frozen/egtea-split1-stream-mtp-2-4-6-vitl256-llwarm/best.pt}"
OUT_ROOT="${OUT_ROOT:-${DATA_ROOT}/outputs/attn_corner_sink/corner_prune}"

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

MAX_VAL_BATCHES=0
WORKERS=4
SUFFIX=""
if [[ "${MODE}" == "smoke" ]]; then
  MAX_VAL_BATCHES=2
  WORKERS=0
  SUFFIX="-smoke"
elif [[ "${MODE}" != "full" ]]; then
  echo "ERROR: MODE must be smoke or full, got ${MODE}" >&2; exit 2
fi

export DATA_ROOT CODE_ROOT VJEPA_ROOT PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TRAIN_CSV VAL_CSV VIDEO_ROOT CHECKPOINT ENCODER_LORA PREDICTOR_LORA PARENT_CKPT OUT_ROOT
export FPS MAX_FRAMES KEEP_COUNT MAX_VAL_BATCHES WORKERS ONLY_CONTEXT_SEC CONTEXT_CROP_SEC BATCH_SIZE SEED RUNS SUFFIX CORNER_MASK_BLOCKS SINK_THRESH_MULT

source "${CODE_ROOT}/scripts/egtea/gpu_monitor.sh"
start_gpu_monitor "${DATA_ROOT}/logs/b18_corner_prune_eval_${SLURM_JOB_ID}.gpu.csv" 30
trap stop_gpu_monitor EXIT INT TERM

share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "$CODE_ROOT"
export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
CROP_TAG=""
if [[ "${CONTEXT_CROP_SEC}" != "0" && "${CONTEXT_CROP_SEC}" != "0.0" ]]; then
  CROP_TAG="-crop${CONTEXT_CROP_SEC%.*}s"
fi
for PAIR in ${RUNS}; do
  STRATEGY="${PAIR%%:*}"; RING="${PAIR##*:}"; [[ "$RING" == "$STRATEGY" ]] && RING=0
  BLK_TAG=""; [[ "$STRATEGY" == "corner_keymask" ]] && BLK_TAG="-${CORNER_MASK_BLOCKS}"
  [[ "$STRATEGY" == "attention_exclude_sink" ]] && BLK_TAG="-m${SINK_THRESH_MULT%.*}"
  OUT_JSON="${OUT_ROOT}/${STRATEGY}-ring${RING}${BLK_TAG}-ctx${ONLY_CONTEXT_SEC%.*}s${CROP_TAG}${SUFFIX}.json"
  echo "=== strategy=${STRATEGY} ring=${RING} blocks=${CORNER_MASK_BLOCKS} -> ${OUT_JSON} ==="
  python -m app.hdepic_lora_action_anticipation.eval_stream_mtp_kvcache_prune \
    --train-csv "$TRAIN_CSV" --val-csv "$VAL_CSV" --video-root "$VIDEO_ROOT" \
    --checkpoint "$CHECKPOINT" --init-from-ckpt "$PARENT_CKPT" \
    --encoder-lora "$ENCODER_LORA" --predictor-lora "$PREDICTOR_LORA" \
    --prune-strategy "$STRATEGY" --keep-count "$KEEP_COUNT" \
    --corner-ring "$RING" --corner-seed "$SEED" --corner-mask-blocks "$CORNER_MASK_BLOCKS" \
    --sink-thresh-mult "$SINK_THRESH_MULT" \
    --fps "$FPS" --max-frames "$MAX_FRAMES" --img-size 256 \
    --horizons-sec 2,4,6 --primary-horizon-sec 2 --anticipation-sec 2 \
    --batch-size "$BATCH_SIZE" --num-workers "$WORKERS" --max-val-batches "$MAX_VAL_BATCHES" \
    --only-context-sec "$ONLY_CONTEXT_SEC" --context-crop-sec "$CONTEXT_CROP_SEC" \
    --out-json "$OUT_JSON"
done
EOF

stop_gpu_monitor
trap - EXIT INT TERM
export GPU_MONITOR_INTERVAL
report_gpu_monitor

echo "B18 corner-prune eval ${MODE} completed: ${OUT_ROOT}"
