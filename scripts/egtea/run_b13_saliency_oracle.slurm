#!/bin/bash
# [B13-ORACLE] Prune by a per-sample ORACLE saliency (task|attention|jepa), native
# Action Top-5. Upper bound of a learned scorer for that signal. Submit CODE_ROOT=<worktree>.
# Env: SALIENCY={task|attention|jepa} MODE={smoke|full} VAL_SUBSET_N.
#SBATCH --job-name=VJEPA2-EXP__b13_sal_oracle
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=02:00:00

set -euo pipefail
SHARE_GPU="${SHARE_GPU:-1}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"
DATA_ROOT="${DATA_ROOT:-${DATA_ROOT}}"
CODE_ROOT="${CODE_ROOT:-${DATA_ROOT}}"
export VJEPA_ROOT
MODE="${MODE:-smoke}"
SALIENCY="${SALIENCY:-task}"
SALIENCY_AGG="${SALIENCY_AGG:-innerprod}"
KEEP_COUNT="${KEEP_COUNT:-4096}"
FPS="${FPS:-8}"; MAX_FRAMES="${MAX_FRAMES:-80}"
VAL_SUBSET_N="${VAL_SUBSET_N:-0}"; VAL_SUBSET_SEED="${VAL_SUBSET_SEED:-0}"
JEPA_HELD_OUT_SEC="${JEPA_HELD_OUT_SEC:-2.0}"

ANN_DIR="${ANN_DIR:-${DATA_ROOT}/data/egtea/vjepa_annotations/stream_half_split/split1}"
VIDEO_ROOT="${VIDEO_ROOT:-${DATA_ROOT}/data/egtea/stream_session_videos}"
CHECKPOINT="${CHECKPOINT:-${DATA_ROOT}/checkpoints/vitl.pt}"
ENCODER_LORA="${ENCODER_LORA:-${DATA_ROOT}/outputs/egtea_split1_video_enc_ll_exact/action_anticipation_frozen/egtea-split1-video-enc-fixed-vitl16-256-10ep/encoder_lora_best.pt}"
PREDICTOR_LORA="${PREDICTOR_LORA:-${DATA_ROOT}/outputs/egtea_split1_video_pred_joint_heads_ll_exact/action_anticipation_frozen/egtea-split1-video-pred-joint-heads-vitl16-256-10ep/predictor_lora_best.pt}"
PARENT_CKPT="${PARENT_CKPT:-${DATA_ROOT}/outputs/egtea_stream_mtp/action_anticipation_frozen/egtea-split1-stream-mtp-2-4-6-vitl256-llwarm/best.pt}"
OUT_ROOT="${OUT_ROOT:-${DATA_ROOT}/outputs/egtea_stream_mtp_kvcache_prune}"
TRAIN_CSV="${TRAIN_CSV:-${ANN_DIR}/EGTEA_train_stream_mtp.csv}"
VAL_CSV="${VAL_CSV:-${ANN_DIR}/EGTEA_val_stream_mtp.csv}"

for p in "$CHECKPOINT" "$ENCODER_LORA" "$PREDICTOR_LORA" "$PARENT_CKPT" "$TRAIN_CSV" "$VAL_CSV"; do
  [[ -f "$p" ]] || { echo "ERROR missing $p" >&2; exit 2; }; done
[[ -f "${CODE_ROOT}/app/hdepic_lora_action_anticipation/eval_saliency_oracle_prune.py" ]] \
  || { echo "ERROR oracle script missing in CODE_ROOT (submit CODE_ROOT=<worktree>)" >&2; exit 2; }

TAG="oracle-${SALIENCY}-${SALIENCY_AGG}-keep${KEEP_COUNT}"
if [[ "$MODE" == "smoke" ]]; then VAL_SUBSET_N=8; TAG="${TAG}-smoke";
elif [[ "$MODE" != "full" ]]; then echo "MODE must be smoke|full" >&2; exit 2; fi
[[ "$VAL_SUBSET_N" != "0" ]] && TAG="${TAG}-sub${VAL_SUBSET_N}"
OUT_JSON="${OUT_ROOT}/${TAG}.json"
mkdir -p "$OUT_ROOT"

export DATA_ROOT CODE_ROOT VJEPA_ROOT PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TRAIN_CSV VAL_CSV VIDEO_ROOT CHECKPOINT ENCODER_LORA PREDICTOR_LORA PARENT_CKPT
export SALIENCY SALIENCY_AGG KEEP_COUNT FPS MAX_FRAMES VAL_SUBSET_N VAL_SUBSET_SEED JEPA_HELD_OUT_SEC OUT_JSON

share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "$CODE_ROOT"; export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
python -m app.hdepic_lora_action_anticipation.eval_saliency_oracle_prune \
  --train-csv "$TRAIN_CSV" --val-csv "$VAL_CSV" --video-root "$VIDEO_ROOT" \
  --checkpoint "$CHECKPOINT" --init-from-ckpt "$PARENT_CKPT" \
  --encoder-lora "$ENCODER_LORA" --predictor-lora "$PREDICTOR_LORA" \
  --saliency "$SALIENCY" --saliency-agg "$SALIENCY_AGG" --keep-count "$KEEP_COUNT" --jepa-held-out-sec "$JEPA_HELD_OUT_SEC" \
  --fps "$FPS" --max-frames "$MAX_FRAMES" --img-size 256 \
  --horizons-sec 2,4,6 --anticipation-sec 2 \
  --val-subset-n "$VAL_SUBSET_N" --val-subset-seed "$VAL_SUBSET_SEED" --out-json "$OUT_JSON"
EOF
echo "B13 oracle ${MODE} completed: ${OUT_JSON}"
