#!/bin/bash
# [ATTN-CORNER-SINK] B18 · Predictor-score pruning experiment on the 16 s / 64-slot index.
# Score site = PREDICTOR block-0 received attention (all queries, head-summed), computed online
# on the full 64-slot context (pass 1), then keep the 16-slot (4096-token) budget by HIGH or LOW
# score; recency + none + encoder-attention are references. Two-pass predictor for pred_* only.
#   * pred_attention_high : keep the highest-score (sink) tokens
#   * pred_attention_low  : keep the lowest-score (content) tokens
#   * recent              : keep the most-recent 16 slots
#   * none                : full 64-slot (upper bound / count-OOD ref)
#   * attention           : ENCODER last-block score keep-high (score-site ablation)
# Env: MODE={smoke|full} ONLY_CONTEXT_SEC(16=genuine 64-slot focus; 0=native all) BATCH_SIZE
#      RUNS (space list of strategies) PRED_SCORE_BLOCK(0)
#SBATCH --job-name=VJEPA2-EXP__b18_pred_score_prune
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=384G
#SBATCH --time=06:00:00

set -euo pipefail
SHARE_GPU="${SHARE_GPU:-1}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"
DATA_ROOT="${DATA_ROOT:-${DATA_ROOT}}"
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}}"
export VJEPA_ROOT

MODE="${MODE:-smoke}"
FPS="${FPS:-8}"; MAX_FRAMES="${MAX_FRAMES:-128}"; KEEP_COUNT="${KEEP_COUNT:-4096}"
ONLY_CONTEXT_SEC="${ONLY_CONTEXT_SEC:-16}"; CONTEXT_CROP_SEC="${CONTEXT_CROP_SEC:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"; PRED_SCORE_BLOCK="${PRED_SCORE_BLOCK:-0}"
PRED_CALIB_PATH="${PRED_CALIB_PATH:-${DATA_ROOT}/outputs/attn_corner_sink/pred_offline_calib/calib_predblk0_map_64x256.npy}"
RUNS="${RUNS:-none recent pred_attention_high pred_attention_low attention}"

ANN_DIR="${ANN_DIR:-${DATA_ROOT}/data/egtea/vjepa_annotations/stream_16s_split/split1}"
VIDEO_ROOT="${VIDEO_ROOT:-${DATA_ROOT}/data/egtea/stream_session_videos}"
CHECKPOINT="${CHECKPOINT:-${DATA_ROOT}/checkpoints/vitl.pt}"
ENCODER_LORA="${ENCODER_LORA:-${DATA_ROOT}/outputs/egtea_split1_video_enc_ll_exact/action_anticipation_frozen/egtea-split1-video-enc-fixed-vitl16-256-10ep/encoder_lora_best.pt}"
PREDICTOR_LORA="${PREDICTOR_LORA:-${DATA_ROOT}/outputs/egtea_split1_video_pred_joint_heads_ll_exact/action_anticipation_frozen/egtea-split1-video-pred-joint-heads-vitl16-256-10ep/predictor_lora_best.pt}"
PARENT_CKPT="${PARENT_CKPT:-${DATA_ROOT}/outputs/egtea_stream_mtp/action_anticipation_frozen/egtea-split1-stream-mtp-2-4-6-vitl256-llwarm/best.pt}"
OUT_ROOT="${OUT_ROOT:-${DATA_ROOT}/outputs/attn_corner_sink/pred_score_prune}"
TRAIN_CSV="${TRAIN_CSV:-${ANN_DIR}/EGTEA_train_stream_mtp.csv}"
VAL_CSV="${VAL_CSV:-${ANN_DIR}/EGTEA_val_stream_mtp.csv}"

mkdir -p "${OUT_ROOT}"
for path in "${CHECKPOINT}" "${ENCODER_LORA}" "${PREDICTOR_LORA}" "${PARENT_CKPT}" "${TRAIN_CSV}" "${VAL_CSV}"; do
  [[ -f "${path}" ]] || { echo "ERROR: missing ${path}" >&2; exit 2; }
done
[[ -d "${VIDEO_ROOT}" ]] || { echo "ERROR: missing video root ${VIDEO_ROOT}" >&2; exit 2; }

MAX_VAL_BATCHES=0; WORKERS=4; SUFFIX=""
if [[ "${MODE}" == "smoke" ]]; then MAX_VAL_BATCHES=2; WORKERS=0; SUFFIX="-smoke";
elif [[ "${MODE}" != "full" ]]; then echo "ERROR: MODE must be smoke|full" >&2; exit 2; fi

export DATA_ROOT CODE_ROOT VJEPA_ROOT PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TRAIN_CSV VAL_CSV VIDEO_ROOT CHECKPOINT ENCODER_LORA PREDICTOR_LORA PARENT_CKPT OUT_ROOT
export FPS MAX_FRAMES KEEP_COUNT MAX_VAL_BATCHES WORKERS ONLY_CONTEXT_SEC CONTEXT_CROP_SEC BATCH_SIZE SUFFIX RUNS PRED_SCORE_BLOCK PRED_CALIB_PATH

share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "$CODE_ROOT"
export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
for STRATEGY in ${RUNS}; do
  OUT_JSON="${OUT_ROOT}/${STRATEGY}-ctx${ONLY_CONTEXT_SEC%.*}s-keep${KEEP_COUNT}-mf${MAX_FRAMES}${SUFFIX}.json"
  echo "=== strategy=${STRATEGY} ctx=${ONLY_CONTEXT_SEC}s keep=${KEEP_COUNT} mf=${MAX_FRAMES} -> ${OUT_JSON} ==="
  python -m app.hdepic_lora_action_anticipation.eval_stream_mtp_kvcache_prune \
    --train-csv "$TRAIN_CSV" --val-csv "$VAL_CSV" --video-root "$VIDEO_ROOT" \
    --checkpoint "$CHECKPOINT" --init-from-ckpt "$PARENT_CKPT" \
    --encoder-lora "$ENCODER_LORA" --predictor-lora "$PREDICTOR_LORA" \
    --prune-strategy "$STRATEGY" --keep-count "$KEEP_COUNT" --pred-score-block "$PRED_SCORE_BLOCK" \
    --pred-calib-path "$PRED_CALIB_PATH" \
    --fps "$FPS" --max-frames "$MAX_FRAMES" --img-size 256 \
    --horizons-sec 2,4,6 --primary-horizon-sec 2 --anticipation-sec 2 \
    --batch-size "$BATCH_SIZE" --num-workers "$WORKERS" --max-val-batches "$MAX_VAL_BATCHES" \
    --only-context-sec "$ONLY_CONTEXT_SEC" --context-crop-sec "$CONTEXT_CROP_SEC" \
    --out-json "$OUT_JSON"
done
EOF
echo "B18 pred-score-prune eval ${MODE} done: ${OUT_ROOT}"
