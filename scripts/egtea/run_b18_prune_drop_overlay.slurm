#!/bin/bash
# [ATTN-CORNER-SINK] B18 · Drop-visualization: black out DROPPED patches on the frames across
# slots, for each prune strategy. Env: STRATEGIES(space list) CALIB_PATH(for pred_offline_*)
# VIDEO_ID(optional) NSLOTS(8) CONTEXT_SEC(16).
#SBATCH --job-name=VJEPA2-EXP__b18_prune_drop_overlay
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=384G
#SBATCH --time=01:00:00

set -euo pipefail
SHARE_GPU="${SHARE_GPU:-1}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"
DATA_ROOT="${DATA_ROOT:-${DATA_ROOT}}"
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}}"
export VJEPA_ROOT

STRATEGIES="${STRATEGIES:-attention recent pred_attention_high pred_attention_low pred_offline_high pred_offline_low}"
CALIB_PATH="${CALIB_PATH:-${DATA_ROOT}/outputs/attn_corner_sink/pred_offline_calib/calib_predblk0_map_64x256.npy}"
VIDEO_ID="${VIDEO_ID:-}"; NSLOTS="${NSLOTS:-8}"; CONTEXT_SEC="${CONTEXT_SEC:-16}"
KEEP_COUNT="${KEEP_COUNT:-4096}"; FRAMES="${FRAMES:-128}"; IMG_SIZE="${IMG_SIZE:-256}"; FPS="${FPS:-8}"; SRC_FPS="${SRC_FPS:-8}"

ANN_DIR="${ANN_DIR:-${DATA_ROOT}/data/egtea/vjepa_annotations/stream_16s_split/split1}"
VIDEO_ROOT="${VIDEO_ROOT:-${DATA_ROOT}/data/egtea/stream_session_videos}"
VAL_CSV="${VAL_CSV:-${ANN_DIR}/EGTEA_val_stream_mtp.csv}"
CHECKPOINT="${CHECKPOINT:-${DATA_ROOT}/checkpoints/vitl.pt}"
ENC_LL="${ENC_LL:-${DATA_ROOT}/outputs/egtea_split1_video_enc_ll_exact/action_anticipation_frozen/egtea-split1-video-enc-fixed-vitl16-256-10ep/encoder_lora_best.pt}"
PRED_LL="${PRED_LL:-${DATA_ROOT}/outputs/egtea_split1_video_pred_joint_heads_ll_exact/action_anticipation_frozen/egtea-split1-video-pred-joint-heads-vitl16-256-10ep/predictor_lora_best.pt}"
PARENT="${PARENT:-${DATA_ROOT}/outputs/egtea_stream_mtp/action_anticipation_frozen/egtea-split1-stream-mtp-2-4-6-vitl256-llwarm/best.pt}"
OUT_DIR="${OUT_DIR:-${DATA_ROOT}/outputs/attn_corner_sink/prune_drop}"

mkdir -p "${OUT_DIR}"
for f in "${CHECKPOINT}" "${ENC_LL}" "${PRED_LL}" "${PARENT}" "${VAL_CSV}"; do
  [[ -f "${f}" ]] || { echo "ERROR: missing ${f}" >&2; exit 2; }
done
export DATA_ROOT CODE_ROOT VJEPA_ROOT PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
ROW_INDEX="${ROW_INDEX:-}"
FRAME_MODE="${FRAME_MODE:-csv}"
export FRAME_MODE
export ROW_INDEX
export STRATEGIES CALIB_PATH VIDEO_ID NSLOTS CONTEXT_SEC KEEP_COUNT FRAMES IMG_SIZE FPS SRC_FPS \
       VAL_CSV VIDEO_ROOT CHECKPOINT ENC_LL PRED_LL PARENT OUT_DIR

share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "$CODE_ROOT"
export SHARED_PROJECT_ROOT="$DATA_ROOT" PROJECT_ROOT="$CODE_ROOT"
export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
VID_ARG=(); [[ -n "${VIDEO_ID}" ]] && VID_ARG=(--video-id "${VIDEO_ID}")
[[ -n "$ROW_INDEX" ]] && VID_ARG+=(--row-index "$ROW_INDEX")
for STRAT in ${STRATEGIES}; do
  CALIB_ARG=(); [[ "$STRAT" == pred_offline_* ]] && CALIB_ARG=(--calib-path "$CALIB_PATH")
  echo "=== drop viz: ${STRAT} ==="
  python -m app.hdepic_lora_action_anticipation.probe_prune_drop_overlay \
    --checkpoint "$CHECKPOINT" --encoder-lora "$ENC_LL" --predictor-lora "$PRED_LL" --init-from-ckpt "$PARENT" \
    --val-csv "$VAL_CSV" --video-root "$VIDEO_ROOT" --out-dir "$OUT_DIR" \
    --frame-mode "$FRAME_MODE" --strategy "$STRAT" "${CALIB_ARG[@]}" \
    --keep-count "$KEEP_COUNT" --n-slots-show "$NSLOTS" --context-sec "$CONTEXT_SEC" \
    --img-size "$IMG_SIZE" --frames "$FRAMES" --fps "$FPS" --src-fps "$SRC_FPS" "${VID_ARG[@]}"
done
EOF
echo "B18 prune-drop overlay done: ${OUT_DIR}"
