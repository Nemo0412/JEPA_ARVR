#!/bin/bash
# [ATTN-CORNER-SINK] B18 · Offline-calibrate the predictor blk0 [64x256] attention map on 512
# train clips @16s (genuine 64 slots). Produces the .npy consumed by pred_offline_{high,low}.
# Env: NCLIPS(512) BLOCK(0) CONTEXT_SEC(16)
#SBATCH --job-name=VJEPA2-EXP__b18_calib_predblk0
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
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}}"
export VJEPA_ROOT

NCLIPS="${NCLIPS:-512}"; BLOCK="${BLOCK:-0}"; CONTEXT_SEC="${CONTEXT_SEC:-16}"
IMG_SIZE="${IMG_SIZE:-256}"; MAXF="${MAXF:-128}"; FPS="${FPS:-8}"; SRC_FPS="${SRC_FPS:-8}"; ANTICIP_SEC="${ANTICIP_SEC:-2}"

ANN_DIR="${ANN_DIR:-${DATA_ROOT}/data/egtea/vjepa_annotations/stream_16s_split/split1}"
VIDEO_ROOT="${VIDEO_ROOT:-${DATA_ROOT}/data/egtea/stream_session_videos}"
CALIB_CSV="${CALIB_CSV:-${ANN_DIR}/EGTEA_train_stream_mtp.csv}"
CHECKPOINT="${CHECKPOINT:-${DATA_ROOT}/checkpoints/vitl.pt}"
ENC_LL="${ENC_LL:-${DATA_ROOT}/outputs/egtea_split1_video_enc_ll_exact/action_anticipation_frozen/egtea-split1-video-enc-fixed-vitl16-256-10ep/encoder_lora_best.pt}"
PRED_LL="${PRED_LL:-${DATA_ROOT}/outputs/egtea_split1_video_pred_joint_heads_ll_exact/action_anticipation_frozen/egtea-split1-video-pred-joint-heads-vitl16-256-10ep/predictor_lora_best.pt}"
PARENT="${PARENT:-${DATA_ROOT}/outputs/egtea_stream_mtp/action_anticipation_frozen/egtea-split1-stream-mtp-2-4-6-vitl256-llwarm/best.pt}"
OUT_DIR="${OUT_DIR:-${DATA_ROOT}/outputs/attn_corner_sink/pred_offline_calib}"

mkdir -p "${OUT_DIR}"
for f in "${CHECKPOINT}" "${ENC_LL}" "${PRED_LL}" "${PARENT}" "${CALIB_CSV}"; do
  [[ -f "${f}" ]] || { echo "ERROR: missing ${f}" >&2; exit 2; }
done
export DATA_ROOT CODE_ROOT VJEPA_ROOT PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NCLIPS BLOCK CONTEXT_SEC IMG_SIZE MAXF FPS SRC_FPS ANTICIP_SEC CALIB_CSV VIDEO_ROOT CHECKPOINT ENC_LL PRED_LL PARENT OUT_DIR

share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "$CODE_ROOT"
export SHARED_PROJECT_ROOT="$DATA_ROOT" PROJECT_ROOT="$CODE_ROOT"
export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
python -m app.hdepic_lora_action_anticipation.calibrate_predictor_blk0_offline \
  --checkpoint "$CHECKPOINT" --encoder-lora "$ENC_LL" --predictor-lora "$PRED_LL" --init-from-ckpt "$PARENT" \
  --calib-csv "$CALIB_CSV" --video-root "$VIDEO_ROOT" --out-dir "$OUT_DIR" \
  --block "$BLOCK" --n-clips "$NCLIPS" --context-sec "$CONTEXT_SEC" --anticipation-sec "$ANTICIP_SEC" \
  --img-size "$IMG_SIZE" --max-frames "$MAXF" --fps "$FPS" --src-fps "$SRC_FPS"
EOF
echo "B18 predblk0 offline calibration done: ${OUT_DIR}"
