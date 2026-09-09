#!/bin/bash
# [ATTN-CORNER-SINK] B18 · Shared-encoder multi-strategy eval: ONE 64-slot encoder forward per
# batch, all prune strategies evaluated on it. Env: MODE{smoke|full} STRATEGIES ONLY_CONTEXT_SEC
# BATCH_SIZE PRED_CALIB_PATH.
#SBATCH --job-name=VJEPA2-EXP__b18_multi_strategy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=384G
#SBATCH --time=01:50:00

set -euo pipefail
SHARE_GPU="${SHARE_GPU:-1}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"
DATA_ROOT="${DATA_ROOT:-${DATA_ROOT}}"
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}}"
export VJEPA_ROOT

MODE="${MODE:-smoke}"
STRATEGIES="${STRATEGIES:-none recent attention pred_attention_high pred_attention_low pred_offline_high pred_offline_low}"
ONLY_CONTEXT_SEC="${ONLY_CONTEXT_SEC:-16}"; BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_FRAMES="${MAX_FRAMES:-128}"; KEEP_COUNT="${KEEP_COUNT:-4096}"; FPS="${FPS:-8}"; SRC_FPS="${SRC_FPS:-8}"
PRED_CALIB_PATH="${PRED_CALIB_PATH:-${DATA_ROOT}/outputs/attn_corner_sink/pred_offline_calib/calib_predblk0_map_64x256.npy}"
PRED_SCORE_BLOCK="${PRED_SCORE_BLOCK:-0}"
NBATCHES="${NBATCHES:-0}"    # 0=full; >0 subsample first N batches (deterministic, same clips across jobs)

ANN_DIR="${ANN_DIR:-${DATA_ROOT}/data/egtea/vjepa_annotations/stream_16s_split/split1}"
VIDEO_ROOT="${VIDEO_ROOT:-${DATA_ROOT}/data/egtea/stream_session_videos}"
TRAIN_CSV="${TRAIN_CSV:-${ANN_DIR}/EGTEA_train_stream_mtp.csv}"
VAL_CSV="${VAL_CSV:-${ANN_DIR}/EGTEA_val_stream_mtp.csv}"
CHECKPOINT="${CHECKPOINT:-${DATA_ROOT}/checkpoints/vitl.pt}"
ENC_LL="${ENC_LL:-${DATA_ROOT}/outputs/egtea_split1_video_enc_ll_exact/action_anticipation_frozen/egtea-split1-video-enc-fixed-vitl16-256-10ep/encoder_lora_best.pt}"
PRED_LL="${PRED_LL:-${DATA_ROOT}/outputs/egtea_split1_video_pred_joint_heads_ll_exact/action_anticipation_frozen/egtea-split1-video-pred-joint-heads-vitl16-256-10ep/predictor_lora_best.pt}"
PARENT="${PARENT:-${DATA_ROOT}/outputs/egtea_stream_mtp/action_anticipation_frozen/egtea-split1-stream-mtp-2-4-6-vitl256-llwarm/best.pt}"
OUT_DIR="${OUT_DIR:-${DATA_ROOT}/outputs/attn_corner_sink/multi_strategy}"

mkdir -p "${OUT_DIR}"
for f in "${CHECKPOINT}" "${ENC_LL}" "${PRED_LL}" "${PARENT}" "${TRAIN_CSV}" "${VAL_CSV}"; do
  [[ -f "${f}" ]] || { echo "ERROR: missing ${f}" >&2; exit 2; }
done
MAX_VAL_BATCHES="${NBATCHES}"; WORKERS=4; TAG="ctx${ONLY_CONTEXT_SEC%.*}s-keep${KEEP_COUNT}-mf${MAX_FRAMES}-sb${PRED_SCORE_BLOCK}"
[[ "${NBATCHES}" != "0" ]] && TAG="${TAG}-nb${NBATCHES}"
if [[ "${MODE}" == "smoke" ]]; then MAX_VAL_BATCHES=2; WORKERS=0; TAG="${TAG}-smoke";
elif [[ "${MODE}" != "full" ]]; then echo "ERROR MODE" >&2; exit 2; fi

export DATA_ROOT CODE_ROOT VJEPA_ROOT PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
ROW_START="${ROW_START:-0}"; ROW_STOP="${ROW_STOP:-}"
TAG="${TAG}-rows${ROW_START}-${ROW_STOP:-end}"
export ROW_START ROW_STOP
export MODE STRATEGIES ONLY_CONTEXT_SEC BATCH_SIZE MAX_FRAMES KEEP_COUNT FPS SRC_FPS PRED_CALIB_PATH PRED_SCORE_BLOCK \
       TRAIN_CSV VAL_CSV VIDEO_ROOT CHECKPOINT ENC_LL PRED_LL PARENT OUT_DIR MAX_VAL_BATCHES WORKERS TAG

share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "$CODE_ROOT"
export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
extra=(); [[ -n "$ROW_STOP" ]] && extra+=(--row-stop "$ROW_STOP")
python -m app.hdepic_lora_action_anticipation.eval_stream_mtp_multi_strategy \
  --train-csv "$TRAIN_CSV" --val-csv "$VAL_CSV" --video-root "$VIDEO_ROOT" \
  --checkpoint "$CHECKPOINT" --init-from-ckpt "$PARENT" \
  --encoder-lora "$ENC_LL" --predictor-lora "$PRED_LL" \
  --strategies "$STRATEGIES" --pred-calib-path "$PRED_CALIB_PATH" --pred-score-block "$PRED_SCORE_BLOCK" \
  --max-frames "$MAX_FRAMES" --keep-count "$KEEP_COUNT" --fps "$FPS" --src-fps "$SRC_FPS" --img-size 256 \
  --horizons-sec 2,4,6 --primary-horizon-sec 2 --anticipation-sec 2 \
  --batch-size "$BATCH_SIZE" --num-workers "$WORKERS" --only-context-sec "$ONLY_CONTEXT_SEC" \
  --row-start "$ROW_START" "${extra[@]}" \
  --max-val-batches "$MAX_VAL_BATCHES" --out-dir "$OUT_DIR" --tag "$TAG"
EOF
echo "B18 multi-strategy eval ${MODE} done: ${OUT_DIR}"
