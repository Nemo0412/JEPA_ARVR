#!/bin/bash
# =============================================================================
# [B13-ATTN-MIDLAYER]  ADD-ON smoke/full driver -- easy to delete.
# -----------------------------------------------------------------------------
# Runs the intermediate-layer received-attention top-K prune eval
# (app/.../attn_midlayer_pruning.py) at one or more encoder blocks L*, frozen,
# native Action Top-5 @ +2/+4/+6 s. Same signal as the last-layer `attention`
# strategy, but pruned INSIDE the encoder (true-pos threaded, no rebase); the
# predictor still receives arange(K) -> fair vs attention/recent.
#
# This file touches NO existing script. To REMOVE: `git rm` this file + the
# python add-on module. To MERGE: fold PRUNE_LAYER into the main eval slurm once
# a `--prune-strategy attention_midlayer` branch lands.
#
# Env knobs: MODE={smoke|full}  PRUNE_LAYERS="8 16"  KEEP_COUNT FPS MAX_FRAMES.
# Default is a smoke over layers 8 and 16 (2 val batches each).
#SBATCH --job-name=VJEPA2-EXP__b13_attn_midlayer_eval
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
PRUNE_LAYERS="${PRUNE_LAYERS:-8 16}"
KEEP_COUNT="${KEEP_COUNT:-4096}"
FPS="${FPS:-8}"
MAX_FRAMES="${MAX_FRAMES:-80}"
BATCH_SIZE="${BATCH_SIZE:-1}"

ANN_DIR="${ANN_DIR:-${DATA_ROOT}/data/egtea/vjepa_annotations/stream_half_split/split1}"
VIDEO_ROOT="${VIDEO_ROOT:-${DATA_ROOT}/data/egtea/stream_session_videos}"
CHECKPOINT="${CHECKPOINT:-${DATA_ROOT}/checkpoints/vitl.pt}"
ENCODER_LORA="${ENCODER_LORA:-${DATA_ROOT}/outputs/egtea_split1_video_enc_ll_exact/action_anticipation_frozen/egtea-split1-video-enc-fixed-vitl16-256-10ep/encoder_lora_best.pt}"
PREDICTOR_LORA="${PREDICTOR_LORA:-${DATA_ROOT}/outputs/egtea_split1_video_pred_joint_heads_ll_exact/action_anticipation_frozen/egtea-split1-video-pred-joint-heads-vitl16-256-10ep/predictor_lora_best.pt}"
PARENT_CKPT="${PARENT_CKPT:-${DATA_ROOT}/outputs/egtea_stream_mtp/action_anticipation_frozen/egtea-split1-stream-mtp-2-4-6-vitl256-llwarm/best.pt}"
OUT_ROOT="${OUT_ROOT:-${DATA_ROOT}/outputs/egtea_stream_mtp_attn_midlayer}"

TRAIN_CSV="${TRAIN_CSV:-${ANN_DIR}/EGTEA_train_stream_mtp.csv}"
VAL_CSV="${VAL_CSV:-${ANN_DIR}/EGTEA_val_stream_mtp.csv}"

mkdir -p "${OUT_ROOT}"
for path in "${CHECKPOINT}" "${ENCODER_LORA}" "${PREDICTOR_LORA}" "${PARENT_CKPT}" \
  "${TRAIN_CSV}" "${VAL_CSV}"; do
  [[ -f "${path}" ]] || { echo "ERROR: missing required file: ${path}" >&2; exit 2; }
done
[[ -d "${VIDEO_ROOT}" ]] || { echo "ERROR: missing video root: ${VIDEO_ROOT}" >&2; exit 2; }
[[ -f "${CODE_ROOT}/app/hdepic_lora_action_anticipation/attn_midlayer_pruning.py" ]] \
  || { echo "ERROR: attn_midlayer add-on missing in CODE_ROOT" >&2; exit 2; }

MAX_VAL_BATCHES=0
WORKERS=4
SMOKE_TAG=""
if [[ "${MODE}" == "smoke" ]]; then
  MAX_VAL_BATCHES=2
  WORKERS=0
  SMOKE_TAG="-smoke"
elif [[ "${MODE}" != "full" ]]; then
  echo "ERROR: MODE must be smoke or full, got ${MODE}" >&2; exit 2
fi

export DATA_ROOT CODE_ROOT VJEPA_ROOT PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TRAIN_CSV VAL_CSV VIDEO_ROOT CHECKPOINT ENCODER_LORA PREDICTOR_LORA PARENT_CKPT
export KEEP_COUNT FPS MAX_FRAMES MAX_VAL_BATCHES WORKERS BATCH_SIZE OUT_ROOT SMOKE_TAG
export PRUNE_LAYERS

source "${CODE_ROOT}/scripts/egtea/gpu_monitor.sh"
start_gpu_monitor "${DATA_ROOT}/logs/b13_attn_midlayer_eval_${SLURM_JOB_ID}.gpu.csv" 30
trap stop_gpu_monitor EXIT INT TERM

share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "$CODE_ROOT"
export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
for L in ${PRUNE_LAYERS}; do
  OUT_JSON="${OUT_ROOT}/attn_midlayer-L${L}-fps${FPS}-keep${KEEP_COUNT}${SMOKE_TAG}.json"
  echo "=== [B13-ATTN-MIDLAYER] prune_layer=${L} -> ${OUT_JSON} ==="
  python -m app.hdepic_lora_action_anticipation.attn_midlayer_pruning \
    --train-csv "$TRAIN_CSV" --val-csv "$VAL_CSV" --video-root "$VIDEO_ROOT" \
    --checkpoint "$CHECKPOINT" --init-from-ckpt "$PARENT_CKPT" \
    --encoder-lora "$ENCODER_LORA" --predictor-lora "$PREDICTOR_LORA" \
    --prune-layer "$L" --keep-count "$KEEP_COUNT" \
    --fps "$FPS" --max-frames "$MAX_FRAMES" --img-size 256 \
    --horizons-sec 2,4,6 --primary-horizon-sec 2 --anticipation-sec 2 \
    --batch-size "$BATCH_SIZE" --num-workers "$WORKERS" --max-val-batches "$MAX_VAL_BATCHES" \
    --out-json "$OUT_JSON"
done
EOF

stop_gpu_monitor
trap - EXIT INT TERM
export GPU_MONITOR_INTERVAL
report_gpu_monitor

echo "B13 attn-midlayer eval ${MODE} completed for layers: ${PRUNE_LAYERS}"
