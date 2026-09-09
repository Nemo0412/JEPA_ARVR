#!/bin/bash
# [ATTN-CORNER-SINK] B18 · Predictor FIRST-layer received-attention pattern (no pruning, 4s).
# Env: BLOCK(default 0) CONTEXT_SEC(default 4) ANTICIP_SEC(default 2) NSAMP(default 200).
#SBATCH --job-name=VJEPA2-EXP__b18_predictor_first_attn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=384G
#SBATCH --time=00:40:00
set -euo pipefail
SHARE_GPU="${SHARE_GPU:-1}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"
DATA_ROOT="${DATA_ROOT:-${DATA_ROOT}}"
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}}"
export VJEPA_ROOT

BLOCKS="${BLOCKS:-0}"; CONTEXT_SEC="${CONTEXT_SEC:-4}"; ANTICIP_SEC="${ANTICIP_SEC:-2}"; NSAMP="${NSAMP:-200}"
IMG_SIZE="${IMG_SIZE:-256}"; MAXF="${MAXF:-32}"; FPS="${FPS:-8}"; SRC_FPS="${SRC_FPS:-8}"
CAPTURE_ENCODER="${CAPTURE_ENCODER:-0}"; ENC_ROPE="${ENC_ROPE:-on}"; PRED_ROPE="${PRED_ROPE:-on}"; ROPE_FACTORIAL="${ROPE_FACTORIAL:-0}"
TAG="${TAG:-blocks${BLOCKS//,/_}_ctx${CONTEXT_SEC}s}"
ANN_DIR="${DATA_ROOT}/data/egtea/vjepa_annotations/stream_half_split/split1"
VAL_CSV="${ANN_DIR}/EGTEA_val_stream_mtp.csv"
VIDEO_ROOT="${DATA_ROOT}/data/egtea/stream_session_videos"
CHECKPOINT="${DATA_ROOT}/checkpoints/vitl.pt"
ENC_LL="${DATA_ROOT}/outputs/egtea_split1_video_enc_ll_exact/action_anticipation_frozen/egtea-split1-video-enc-fixed-vitl16-256-10ep/encoder_lora_best.pt"
PRED_LL="${DATA_ROOT}/outputs/egtea_split1_video_pred_joint_heads_ll_exact/action_anticipation_frozen/egtea-split1-video-pred-joint-heads-vitl16-256-10ep/predictor_lora_best.pt"
PARENT="${DATA_ROOT}/outputs/egtea_stream_mtp/action_anticipation_frozen/egtea-split1-stream-mtp-2-4-6-vitl256-llwarm/best.pt"
OUT_DIR="${OUT_DIR:-${DATA_ROOT}/outputs/attn_corner_sink/predictor_first/${TAG}}"
mkdir -p "${DATA_ROOT}/outputs/attn_corner_sink/predictor_first"
for f in "${CHECKPOINT}" "${ENC_LL}" "${PRED_LL}" "${PARENT}" "${VAL_CSV}"; do
  [[ -f "${f}" ]] || { echo "ERROR: missing ${f}" >&2; exit 2; }
done
export DATA_ROOT CODE_ROOT VJEPA_ROOT PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLOCKS CONTEXT_SEC ANTICIP_SEC NSAMP IMG_SIZE MAXF FPS SRC_FPS VAL_CSV VIDEO_ROOT CHECKPOINT ENC_LL PRED_LL PARENT OUT_DIR CAPTURE_ENCODER ENC_ROPE PRED_ROPE ROPE_FACTORIAL
share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "$CODE_ROOT"
export SHARED_PROJECT_ROOT="$DATA_ROOT" PROJECT_ROOT="$CODE_ROOT"
export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
CAP_ARG=""; [[ "$CAPTURE_ENCODER" == "1" ]] && CAP_ARG="--capture-encoder"
FAC_ARG=""; [[ "$ROPE_FACTORIAL" == "1" ]] && FAC_ARG="--rope-factorial"
python -m app.hdepic_lora_action_anticipation.probe_predictor_first_layer_attn \
  --checkpoint "$CHECKPOINT" --encoder-lora "$ENC_LL" --predictor-lora "$PRED_LL" --init-from-ckpt "$PARENT" \
  --val-csv "$VAL_CSV" --video-root "$VIDEO_ROOT" --out-dir "$OUT_DIR" \
  --blocks "$BLOCKS" --img-size "$IMG_SIZE" --max-frames "$MAXF" --fps "$FPS" --src-fps "$SRC_FPS" \
  --context-sec "$CONTEXT_SEC" --anticipation-sec "$ANTICIP_SEC" --n-samples "$NSAMP" \
  --enc-rope "$ENC_ROPE" --pred-rope "$PRED_ROPE" $CAP_ARG $FAC_ARG
EOF
echo "B18 predictor-first-attn done: ${OUT_DIR}"
