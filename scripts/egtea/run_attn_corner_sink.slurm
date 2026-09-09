#!/bin/bash
# [ATTN-CORNER-SINK] Per-head encoder last-block received-attention corner analysis.
# Encoder-only forward on an EGTEA val subset; no finetune, no predictor/classifier.
#
# CODE_ROOT defaults to the experimental worktree (the analysis module lives only
# there). Data/checkpoints/overlay are read from DATA_ROOT (main tree, read-only).
#
# Env knobs:
#   VARIANT={finetuned20|base20|base21}  IMG_SIZE  MAX_FRAMES  N_SAMPLES
#   MODE={smoke|full}  BLOCK(default -1)  ONLY_CONTEXT_SEC(default 10)
#SBATCH --job-name=VJEPA2-EXP__attn_corner_sink
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

VARIANT="${VARIANT:-finetuned20}"
MODE="${MODE:-smoke}"
IMG_SIZE="${IMG_SIZE:-256}"
MAX_FRAMES="${MAX_FRAMES:-80}"
FPS="${FPS:-8}"
SRC_FPS="${SRC_FPS:-8}"
BLOCK="${BLOCK:--1}"
ONLY_CONTEXT_SEC="${ONLY_CONTEXT_SEC:-10}"
N_SAMPLES="${N_SAMPLES:-200}"

ANN_DIR="${ANN_DIR:-${DATA_ROOT}/data/egtea/vjepa_annotations/stream_half_split/split1}"
VIDEO_ROOT="${VIDEO_ROOT:-${DATA_ROOT}/data/egtea/stream_session_videos}"
VAL_CSV="${VAL_CSV:-${ANN_DIR}/EGTEA_val_stream_mtp.csv}"
# non-egocentric control: point at a directory of arbitrary MP4s instead of the CSV
VIDEO_DIR="${VIDEO_DIR:-}"

# checkpoint + finetune sidecars default by variant
ENC_LL="${DATA_ROOT}/outputs/egtea_split1_video_enc_ll_exact/action_anticipation_frozen/egtea-split1-video-enc-fixed-vitl16-256-10ep/encoder_lora_best.pt"
PRED_LL="${DATA_ROOT}/outputs/egtea_split1_video_pred_joint_heads_ll_exact/action_anticipation_frozen/egtea-split1-video-pred-joint-heads-vitl16-256-10ep/predictor_lora_best.pt"
PARENT="${DATA_ROOT}/outputs/egtea_stream_mtp/action_anticipation_frozen/egtea-split1-stream-mtp-2-4-6-vitl256-llwarm/best.pt"
if [[ "${VARIANT}" == "base21" ]]; then
  CHECKPOINT="${CHECKPOINT:-${DATA_ROOT}/checkpoints/vjepa2_1_vitl_dist_vitG_384.pt}"
else
  CHECKPOINT="${CHECKPOINT:-${DATA_ROOT}/checkpoints/vitl.pt}"
fi

VERIFY_FLAG=()
EXTRA=()
if [[ "${VARIANT}" == "finetuned20" ]]; then
  EXTRA+=(--encoder-lora "${ENC_LL}" --predictor-lora "${PRED_LL}" --init-from-ckpt "${PARENT}")
  [[ -f "${ENC_LL}" && -f "${PRED_LL}" && -f "${PARENT}" ]] \
    || { echo "ERROR: finetuned sidecars/parent missing" >&2; exit 2; }
fi

OUT_ROOT="${OUT_ROOT:-${DATA_ROOT}/outputs/attn_corner_sink}"
TAG="${VARIANT}_res${IMG_SIZE}_f${MAX_FRAMES}"
[[ -n "${VIDEO_DIR}" ]] && TAG="${TAG}_${DATASET_TAG:-nonego}"
[[ "${BLOCK}" != "-1" ]] && TAG="${TAG}_b${BLOCK}"
[[ -n "${HFLIP:-}" ]] && { EXTRA+=(--hflip); TAG="${TAG}_hflip"; }
[[ -n "${DUMP:-}" ]] && EXTRA+=(--dump-per-sample)
if [[ "${MODE}" == "smoke" ]]; then
  N_SAMPLES=4
  TAG="${TAG}_smoke"
  [[ "${VARIANT}" == "base21" ]] || VERIFY_FLAG=(--verify)
elif [[ "${MODE}" != "full" ]]; then
  echo "ERROR: MODE must be smoke or full" >&2; exit 2
fi
OUT_DIR="${OUT_ROOT}/${TAG}"
mkdir -p "${OUT_ROOT}"

[[ -f "${CHECKPOINT}" ]] || { echo "ERROR: missing checkpoint: ${CHECKPOINT}" >&2; exit 2; }
if [[ -n "${VIDEO_DIR}" ]]; then
  [[ -d "${VIDEO_DIR}" ]] || { echo "ERROR: missing VIDEO_DIR: ${VIDEO_DIR}" >&2; exit 2; }
  DATA_STR="--video-dir ${VIDEO_DIR}"
else
  [[ -f "${VAL_CSV}" ]] || { echo "ERROR: missing val csv: ${VAL_CSV}" >&2; exit 2; }
  [[ -d "${VIDEO_ROOT}" ]] || { echo "ERROR: missing video root: ${VIDEO_ROOT}" >&2; exit 2; }
  DATA_STR="--val-csv ${VAL_CSV} --video-root ${VIDEO_ROOT}"
fi
[[ -f "${CODE_ROOT}/app/hdepic_lora_action_anticipation/analyze_encoder_head_attn_corners.py" ]] \
  || { echo "ERROR: analysis module missing in CODE_ROOT=${CODE_ROOT}" >&2; exit 2; }

export DATA_ROOT CODE_ROOT VJEPA_ROOT PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export VARIANT CHECKPOINT OUT_DIR IMG_SIZE MAX_FRAMES FPS SRC_FPS BLOCK ONLY_CONTEXT_SEC N_SAMPLES
export DATA_STR EXTRA_STR="${EXTRA[*]:-}" VERIFY_STR="${VERIFY_FLAG[*]:-}"

share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "$CODE_ROOT"
export SHARED_PROJECT_ROOT="$DATA_ROOT" PROJECT_ROOT="$CODE_ROOT"
export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
python -m app.hdepic_lora_action_anticipation.analyze_encoder_head_attn_corners \
  --variant "$VARIANT" --checkpoint "$CHECKPOINT" \
  ${DATA_STR} --out-dir "$OUT_DIR" \
  --img-size "$IMG_SIZE" --max-frames "$MAX_FRAMES" --fps "$FPS" --src-fps "$SRC_FPS" \
  --block "$BLOCK" --only-context-sec "$ONLY_CONTEXT_SEC" --n-samples "$N_SAMPLES" \
  ${EXTRA_STR} ${VERIFY_STR}
EOF
