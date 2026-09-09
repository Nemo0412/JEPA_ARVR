#!/bin/bash
# [ATTN-CORNER-SINK] B18 · Predictor port of ll's [T x 256] all-heads attn-mass figure.
# Captures PREDICTOR block-0 received-attention over the context, laid out [T_slots x 256].
# FRAMES=32 -> 16 slots (4s, post-prune memory budget); FRAMES=128 -> 64 slots (16s,
# pre-prune streaming input, OOD for the predictor -> num_patches auto-raised). Full model.
# Env: FRAMES(default 128) BLOCK(default 0) VIDEO_ID(optional) CONTEXT_SEC(default 10) ANTICIP_SEC(default 2)
#SBATCH --job-name=VJEPA2-EXP__b18_pred_all_heads_mass
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=384G
#SBATCH --time=00:30:00

set -euo pipefail
SHARE_GPU="${SHARE_GPU:-1}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"
DATA_ROOT="${DATA_ROOT:-${DATA_ROOT}}"
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}}"
export VJEPA_ROOT

FRAMES="${FRAMES:-128}"; FPS="${FPS:-8}"; SRC_FPS="${SRC_FPS:-8}"; IMG_SIZE="${IMG_SIZE:-256}"
BLOCK="${BLOCK:-0}"; CONTEXT_SEC="${CONTEXT_SEC:-10}"; ANTICIP_SEC="${ANTICIP_SEC:-2}"; VIDEO_ID="${VIDEO_ID:-}"

ANN_DIR="${ANN_DIR:-${DATA_ROOT}/data/egtea/vjepa_annotations/stream_half_split/split1}"
VIDEO_ROOT="${VIDEO_ROOT:-${DATA_ROOT}/data/egtea/stream_session_videos}"
VAL_CSV="${VAL_CSV:-${ANN_DIR}/EGTEA_val_stream_mtp.csv}"
CHECKPOINT="${CHECKPOINT:-${DATA_ROOT}/checkpoints/vitl.pt}"
ENC_LL="${DATA_ROOT}/outputs/egtea_split1_video_enc_ll_exact/action_anticipation_frozen/egtea-split1-video-enc-fixed-vitl16-256-10ep/encoder_lora_best.pt"
PRED_LL="${DATA_ROOT}/outputs/egtea_split1_video_pred_joint_heads_ll_exact/action_anticipation_frozen/egtea-split1-video-pred-joint-heads-vitl16-256-10ep/predictor_lora_best.pt"
PARENT="${DATA_ROOT}/outputs/egtea_stream_mtp/action_anticipation_frozen/egtea-split1-stream-mtp-2-4-6-vitl256-llwarm/best.pt"
OUT_ROOT="${OUT_ROOT:-${DATA_ROOT}/outputs/attn_corner_sink/pred_all_heads_mass}"

T_SLOTS=$(( FRAMES / 2 ))
TAG="${VIDEO_ID:-firstrow}_blk${BLOCK}_t${T_SLOTS}x256_${FRAMES}f"
OUT_DIR="${OUT_ROOT}/${TAG}"
mkdir -p "${OUT_ROOT}"
for f in "${CHECKPOINT}" "${ENC_LL}" "${PRED_LL}" "${PARENT}" "${VAL_CSV}"; do
  [[ -f "${f}" ]] || { echo "ERROR: missing ${f}" >&2; exit 2; }
done
[[ -d "${VIDEO_ROOT}" ]] || { echo "ERROR: missing ${VIDEO_ROOT}" >&2; exit 2; }

export DATA_ROOT CODE_ROOT VJEPA_ROOT PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CHECKPOINT ENC_LL PRED_LL PARENT VAL_CSV VIDEO_ROOT OUT_DIR IMG_SIZE FRAMES FPS SRC_FPS CONTEXT_SEC ANTICIP_SEC VIDEO_ID BLOCK

share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "$CODE_ROOT"
export SHARED_PROJECT_ROOT="$DATA_ROOT" PROJECT_ROOT="$CODE_ROOT"
export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
VID_ARG=(); [[ -n "${VIDEO_ID}" ]] && VID_ARG=(--video-id "${VIDEO_ID}")
python -m app.hdepic_lora_action_anticipation.probe_predictor_all_heads_mass_Tx256 \
  --checkpoint "$CHECKPOINT" --encoder-lora "$ENC_LL" --predictor-lora "$PRED_LL" --init-from-ckpt "$PARENT" \
  --val-csv "$VAL_CSV" --video-root "$VIDEO_ROOT" --out-dir "$OUT_DIR" \
  --block "$BLOCK" --img-size "$IMG_SIZE" --frames "$FRAMES" --fps "$FPS" --src-fps "$SRC_FPS" \
  --context-sec "$CONTEXT_SEC" --anticipation-sec "$ANTICIP_SEC" "${VID_ARG[@]}"
EOF
echo "B18 predictor all-heads-mass done: ${OUT_DIR}"
