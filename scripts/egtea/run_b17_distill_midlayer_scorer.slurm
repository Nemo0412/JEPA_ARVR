#!/bin/bash
# [B17-LEARNED-EARLY-PRUNER] Stage 1: distill a MidLayerTokenScorer to per-token
# task saliency (+ attention control arm) from block-L* features. Frozen base +
# mtp_clf; per-sample one backward for saliency; trains a small MLP; saves
# scorer_{task,attn}.pt for Stage-2 co-adaptation.
#
# Env: MODE={smoke|full} N_SAMPLES PRUNE_LAYER TARGETS. Submit with CODE_ROOT=<worktree>.
#SBATCH --job-name=VJEPA2-EXP__b17_distill_scorer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=01:30:00

set -euo pipefail
SHARE_GPU="${SHARE_GPU:-1}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"

DATA_ROOT="${DATA_ROOT:-${DATA_ROOT}}"
CODE_ROOT="${CODE_ROOT:-${DATA_ROOT}}"
export VJEPA_ROOT

MODE="${MODE:-smoke}"
N_SAMPLES="${N_SAMPLES:-400}"
PRUNE_LAYER="${PRUNE_LAYER:-8}"
TARGETS="${TARGETS:-task,attn}"
KEEP_COUNT="${KEEP_COUNT:-4096}"
FPS="${FPS:-8}"
MAX_FRAMES="${MAX_FRAMES:-80}"
VAL_SUBSET_SEED="${VAL_SUBSET_SEED:-0}"

ANN_DIR="${ANN_DIR:-${DATA_ROOT}/data/egtea/vjepa_annotations/stream_half_split/split1}"
VIDEO_ROOT="${VIDEO_ROOT:-${DATA_ROOT}/data/egtea/stream_session_videos}"
CHECKPOINT="${CHECKPOINT:-${DATA_ROOT}/checkpoints/vitl.pt}"
ENCODER_LORA="${ENCODER_LORA:-${DATA_ROOT}/outputs/egtea_split1_video_enc_ll_exact/action_anticipation_frozen/egtea-split1-video-enc-fixed-vitl16-256-10ep/encoder_lora_best.pt}"
PREDICTOR_LORA="${PREDICTOR_LORA:-${DATA_ROOT}/outputs/egtea_split1_video_pred_joint_heads_ll_exact/action_anticipation_frozen/egtea-split1-video-pred-joint-heads-vitl16-256-10ep/predictor_lora_best.pt}"
PARENT_CKPT="${PARENT_CKPT:-${DATA_ROOT}/outputs/egtea_stream_mtp/action_anticipation_frozen/egtea-split1-stream-mtp-2-4-6-vitl256-llwarm/best.pt}"
OUT_ROOT="${OUT_ROOT:-${DATA_ROOT}/outputs/egtea_stream_mtp_learned_prune}"
TRAIN_CSV="${TRAIN_CSV:-${ANN_DIR}/EGTEA_train_stream_mtp.csv}"
VAL_CSV="${VAL_CSV:-${ANN_DIR}/EGTEA_val_stream_mtp.csv}"

for path in "${CHECKPOINT}" "${ENCODER_LORA}" "${PREDICTOR_LORA}" "${PARENT_CKPT}" \
  "${TRAIN_CSV}" "${VAL_CSV}"; do
  [[ -f "${path}" ]] || { echo "ERROR: missing required file: ${path}" >&2; exit 2; }
done
[[ -d "${VIDEO_ROOT}" ]] || { echo "ERROR: missing video root: ${VIDEO_ROOT}" >&2; exit 2; }
[[ -f "${CODE_ROOT}/app/hdepic_lora_action_anticipation/distill_midlayer_scorer.py" ]] \
  || { echo "ERROR: distill script missing in CODE_ROOT (submit with CODE_ROOT=<worktree>)" >&2; exit 2; }

TAG="distill-L${PRUNE_LAYER}-s${VAL_SUBSET_SEED}-n${N_SAMPLES}"
if [[ "${MODE}" == "smoke" ]]; then
  N_SAMPLES=8
  TAG="${TAG}-smoke"
elif [[ "${MODE}" != "full" ]]; then
  echo "ERROR: MODE must be smoke or full, got ${MODE}" >&2; exit 2
fi
OUT_DIR="${OUT_ROOT}/${TAG}"
mkdir -p "${OUT_DIR}"

export DATA_ROOT CODE_ROOT VJEPA_ROOT PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TRAIN_CSV VAL_CSV VIDEO_ROOT CHECKPOINT ENCODER_LORA PREDICTOR_LORA PARENT_CKPT
export N_SAMPLES PRUNE_LAYER TARGETS KEEP_COUNT FPS MAX_FRAMES VAL_SUBSET_SEED OUT_DIR

share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "$CODE_ROOT"
export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
python -m app.hdepic_lora_action_anticipation.distill_midlayer_scorer \
  --train-csv "$TRAIN_CSV" --val-csv "$VAL_CSV" --video-root "$VIDEO_ROOT" \
  --checkpoint "$CHECKPOINT" --init-from-ckpt "$PARENT_CKPT" \
  --encoder-lora "$ENCODER_LORA" --predictor-lora "$PREDICTOR_LORA" \
  --prune-layer "$PRUNE_LAYER" --keep-count "$KEEP_COUNT" --targets "$TARGETS" \
  --fps "$FPS" --max-frames "$MAX_FRAMES" --img-size 256 \
  --horizons-sec 2,4,6 --anticipation-sec 2 \
  --n-samples "$N_SAMPLES" --val-subset-seed "$VAL_SUBSET_SEED" \
  --out-dir "$OUT_DIR"
EOF

echo "B13 distill ${MODE} completed: ${OUT_DIR}"
