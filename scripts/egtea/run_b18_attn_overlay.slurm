#!/bin/bash
# [ATTN-CORNER-SINK] B18 · ll-style overlay of high attention-score tokens on the mid-frame.
# Env: SITE(encoder|predictor, default encoder) BLOCK(0) FRAMES(128) TOPK(8) PERSIST(0.6)
#      VIDEO_A VIDEO_B CONTEXT_SEC(10) ANTICIP_SEC(2)
#SBATCH --job-name=VJEPA2-EXP__b18_attn_overlay
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

SITE="${SITE:-encoder}"; BLOCK="${BLOCK:-0}"; FRAMES="${FRAMES:-128}"; TOPK="${TOPK:-8}"; PERSIST="${PERSIST:-0.6}"
FPS="${FPS:-8}"; SRC_FPS="${SRC_FPS:-8}"; IMG_SIZE="${IMG_SIZE:-256}"
CONTEXT_SEC="${CONTEXT_SEC:-10}"; ANTICIP_SEC="${ANTICIP_SEC:-2}"
VIDEO_A="${VIDEO_A:-OP01-R01-PastaSalad}"; VIDEO_B="${VIDEO_B:-OP01-R02-TurkeySandwich}"

ANN_DIR="${ANN_DIR:-${DATA_ROOT}/data/egtea/vjepa_annotations/stream_half_split/split1}"
VIDEO_ROOT="${VIDEO_ROOT:-${DATA_ROOT}/data/egtea/stream_session_videos}"
VAL_CSV="${VAL_CSV:-${ANN_DIR}/EGTEA_val_stream_mtp.csv}"
CHECKPOINT="${CHECKPOINT:-${DATA_ROOT}/checkpoints/vitl.pt}"
ENC_LL="${DATA_ROOT}/outputs/egtea_split1_video_enc_ll_exact/action_anticipation_frozen/egtea-split1-video-enc-fixed-vitl16-256-10ep/encoder_lora_best.pt"
PRED_LL="${DATA_ROOT}/outputs/egtea_split1_video_pred_joint_heads_ll_exact/action_anticipation_frozen/egtea-split1-video-pred-joint-heads-vitl16-256-10ep/predictor_lora_best.pt"
PARENT="${DATA_ROOT}/outputs/egtea_stream_mtp/action_anticipation_frozen/egtea-split1-stream-mtp-2-4-6-vitl256-llwarm/best.pt"
OUT_ROOT="${OUT_ROOT:-${DATA_ROOT}/outputs/attn_corner_sink/attn_overlay}"

T_SLOTS=$(( FRAMES / 2 ))
SITE_TAG="${SITE}"; [[ "${SITE}" == "predictor" ]] && SITE_TAG="predictor_blk${BLOCK}"
OUT_DIR="${OUT_ROOT}/${SITE_TAG}_t${T_SLOTS}x256_${FRAMES}f"
mkdir -p "${OUT_ROOT}"
for f in "${CHECKPOINT}" "${ENC_LL}" "${PRED_LL}" "${PARENT}" "${VAL_CSV}"; do
  [[ -f "${f}" ]] || { echo "ERROR: missing ${f}" >&2; exit 2; }
done
[[ -d "${VIDEO_ROOT}" ]] || { echo "ERROR: missing ${VIDEO_ROOT}" >&2; exit 2; }

export DATA_ROOT CODE_ROOT VJEPA_ROOT PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CHECKPOINT ENC_LL PRED_LL PARENT VAL_CSV VIDEO_ROOT OUT_DIR IMG_SIZE FRAMES FPS SRC_FPS \
       CONTEXT_SEC ANTICIP_SEC SITE BLOCK TOPK PERSIST VIDEO_A VIDEO_B

share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "$CODE_ROOT"
export SHARED_PROJECT_ROOT="$DATA_ROOT" PROJECT_ROOT="$CODE_ROOT"
export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
python -m app.hdepic_lora_action_anticipation.probe_attn_topk_overlay \
  --checkpoint "$CHECKPOINT" --encoder-lora "$ENC_LL" --predictor-lora "$PRED_LL" --init-from-ckpt "$PARENT" \
  --val-csv "$VAL_CSV" --video-root "$VIDEO_ROOT" --out-dir "$OUT_DIR" \
  --video-a "$VIDEO_A" --video-b "$VIDEO_B" --site "$SITE" --block "$BLOCK" \
  --topk "$TOPK" --persist-frac "$PERSIST" \
  --img-size "$IMG_SIZE" --frames "$FRAMES" --fps "$FPS" --src-fps "$SRC_FPS" \
  --context-sec "$CONTEXT_SEC" --anticipation-sec "$ANTICIP_SEC"
EOF
echo "B18 attn overlay done: ${OUT_DIR}"
