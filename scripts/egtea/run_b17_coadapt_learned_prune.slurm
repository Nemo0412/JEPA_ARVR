#!/bin/bash
# [B17-LEARNED-EARLY-PRUNER] Stage 2: co-adapt reader with a FROZEN distilled L*
# scorer (candidate #1 main experiment). Budget-finetune recipe: encoder +
# encoder-LoRA frozen, warm-start parent-best, predictor-LoRA + MTP heads trainable,
# 4096 budget, learned prune INSIDE the encoder at L*. Trainer runs full val each
# epoch -> native Action Top-5 @ +2/+4/+6 under the learned pruner is the answer.
#
# Env: SCORER_CKPT (required, Stage-1 scorer_task.pt), PRUNE_LAYER, EPOCHS, LR, MODE.
# Submit with CODE_ROOT=<worktree>.  bf16 GradScaler forced disabled in the wrapper.
#SBATCH --job-name=VJEPA2-EXP__b17_coadapt_learned
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=384G
#SBATCH --time=08:00:00

set -euo pipefail
SHARE_GPU="${SHARE_GPU:-1}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"

DATA_ROOT="${DATA_ROOT:-${DATA_ROOT}}"
CODE_ROOT="${CODE_ROOT:-${DATA_ROOT}}"
export VJEPA_ROOT

MODE="${MODE:-smoke}"
PRUNE_LAYER="${PRUNE_LAYER:-8}"
KEEP_COUNT="${KEEP_COUNT:-4096}"
FPS="${FPS:-8}"
MAX_FRAMES="${MAX_FRAMES:-80}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-2e-5}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SCORER_CKPT="${SCORER_CKPT:?set SCORER_CKPT to a Stage-1 scorer_*.pt}"

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
  "${TRAIN_CSV}" "${VAL_CSV}" "${SCORER_CKPT}"; do
  [[ -f "${path}" ]] || { echo "ERROR: missing required file: ${path}" >&2; exit 2; }
done
[[ -d "${VIDEO_ROOT}" ]] || { echo "ERROR: missing video root: ${VIDEO_ROOT}" >&2; exit 2; }
[[ -f "${CODE_ROOT}/app/hdepic_lora_action_anticipation/train_stream_mtp_learned_prune.py" ]] \
  || { echo "ERROR: Stage-2 wrapper missing in CODE_ROOT (submit with CODE_ROOT=<worktree>)" >&2; exit 2; }

SNAME="$(basename "${SCORER_CKPT}" .pt)"
TAG="coadapt-L${PRUNE_LAYER}-${SNAME}-ep${EPOCHS}"
if [[ "${MODE}" == "smoke" ]]; then
  EPOCHS=1; TAG="${TAG}-smoke"
elif [[ "${MODE}" != "full" ]]; then
  echo "ERROR: MODE must be smoke or full, got ${MODE}" >&2; exit 2
fi
OUT_DIR="${OUT_ROOT}/${TAG}"
mkdir -p "${OUT_DIR}"
# Smoke: tiny head-CSVs (header + 40 rows) so a full epoch is a few steps. The
# canonical trainer has no batch-limit flag; a subset CSV is the clean lever.
SMOKE_ARGS=()
if [[ "${MODE}" == "smoke" ]]; then
  head -n 41 "${TRAIN_CSV}" > "${OUT_DIR}/train_smoke.csv"
  head -n 41 "${VAL_CSV}"   > "${OUT_DIR}/val_smoke.csv"
  TRAIN_CSV="${OUT_DIR}/train_smoke.csv"
  VAL_CSV="${OUT_DIR}/val_smoke.csv"
  # Tiny CSV -> tiny vocab (mismatches the parent MTP head size); skip the parent
  # head load. Smoke only validates pruner build + forward/backward + bf16 no-NaN.
  SMOKE_ARGS+=(--heads-from-scratch)
fi
export SMOKE_HEADS="${SMOKE_ARGS[*]:-}"

export DATA_ROOT CODE_ROOT VJEPA_ROOT PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export SCORER_CKPT PRUNE_LAYER
export TRAIN_CSV VAL_CSV VIDEO_ROOT CHECKPOINT ENCODER_LORA PREDICTOR_LORA PARENT_CKPT
export OUT_DIR EPOCHS LR BATCH_SIZE KEEP_COUNT FPS MAX_FRAMES MODE

source "${CODE_ROOT}/scripts/egtea/gpu_monitor.sh" 2>/dev/null || true
type start_gpu_monitor >/dev/null 2>&1 && start_gpu_monitor "${DATA_ROOT}/logs/b13_coadapt_${SLURM_JOB_ID}.gpu.csv" 30 || true

share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "$CODE_ROOT"
export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
ARGS=(
  --train-csv "$TRAIN_CSV" --val-csv "$VAL_CSV" --video-root "$VIDEO_ROOT"
  --checkpoint "$CHECKPOINT" --init-from-ckpt "$PARENT_CKPT"
  --encoder-lora "$ENCODER_LORA" --predictor-lora "$PREDICTOR_LORA"
  --out-dir "$OUT_DIR" --keep-count "$KEEP_COUNT"
  --fps "$FPS" --max-frames "$MAX_FRAMES" --img-size 256
  --horizons-sec 2,4,6 --loss-weights 1.0,0.7,0.5 --primary-horizon-sec 2 --anticipation-sec 2
  --epochs "$EPOCHS" --lr "$LR" --batch-size "$BATCH_SIZE" --num-workers 8 --save-every 200
)
# shellcheck disable=SC2206
[[ -n "${SMOKE_HEADS:-}" ]] && ARGS+=(${SMOKE_HEADS})
python -m app.hdepic_lora_action_anticipation.train_stream_mtp_learned_prune "${ARGS[@]}"
EOF

type stop_gpu_monitor >/dev/null 2>&1 && stop_gpu_monitor || true
echo "B13 co-adapt ${MODE} completed: ${OUT_DIR}"
