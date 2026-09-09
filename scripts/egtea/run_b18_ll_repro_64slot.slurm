#!/bin/bash
# [ATTN-CORNER-SINK] B18 · Run ll's OWN eval_64slot_pred0_prune_vs_last16.py (opencl 4b3975b) in
# our env at LOW NOISE (stride 2, all samples) to resolve the enc_topk discrepancy (ll n=275 vs
# ours n=3092). Faithful to ll: same 10s val CSV extended-back-to-16s, same 3 arms, same parent.
#SBATCH --job-name=VJEPA2-EXP__b18_ll_repro_64slot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=384G
#SBATCH --time=02:00:00
set -euo pipefail
SHARE_GPU="${SHARE_GPU:-1}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"
DATA_ROOT="${DATA_ROOT:-${DATA_ROOT}}"
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}}"
export VJEPA_ROOT
MAX_SAMPLES="${MAX_SAMPLES:-0}"; STRIDE="${STRIDE:-2}"
ANN_DIR=${HDEPIC_ANN_DIR:?Set HDEPIC_ANN_DIR}
VIDEO_ROOT=${HDEPIC_VIDEO_ROOT:?Set HDEPIC_VIDEO_ROOT}
CHECKPOINT="${DATA_ROOT}/checkpoints/vitl.pt"
ENC_LORA="${DATA_ROOT}/checkpoints/hdepic_p01_ll/encoder_lora_best.pt"
PRED_LORA="${DATA_ROOT}/checkpoints/hdepic_p01_ll/predictor_lora_best.pt"
STREAM_CKPT=${HDEPIC_PARENT:?Set HDEPIC_PARENT}
OUT_DIR="${DATA_ROOT}/outputs/attn_corner_sink/ll_repro_64slot"
mkdir -p "${OUT_DIR}"
for f in "$CHECKPOINT" "$ENC_LORA" "$PRED_LORA" "$STREAM_CKPT" "$ANN_DIR/HD_EPIC_val_stream_mtp.csv"; do
  [[ -f "$f" ]] || { echo "ERROR missing $f" >&2; exit 2; }
done
export DATA_ROOT CODE_ROOT VJEPA_ROOT PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export ANN_DIR VIDEO_ROOT CHECKPOINT ENC_LORA PRED_LORA STREAM_CKPT OUT_DIR MAX_SAMPLES STRIDE
share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "$CODE_ROOT"
export VJEPA_ROOT PROJECT_ROOT="$CODE_ROOT" SHARED_PROJECT_ROOT="$DATA_ROOT"
export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
python scripts/eval_64slot_pred0_prune_vs_last16.py \
  --val-csv "$ANN_DIR/HD_EPIC_val_stream_mtp.csv" \
  --train-csv "$ANN_DIR/HD_EPIC_train_stream_mtp.csv" \
  --video-root "$VIDEO_ROOT" --checkpoint "$CHECKPOINT" \
  --encoder-lora "$ENC_LORA" --predictor-lora "$PRED_LORA" \
  --stream-ckpt "$STREAM_CKPT" --out-dir "$OUT_DIR" \
  --modes "last16,pred0_topk,enc_topk" --keep-count 4096 --context-sec 16 --require-ctx-sec 10 \
  --horizons-sec "2,4,6" --anticipation-sec 2.0 --max-samples "$MAX_SAMPLES" --stride "$STRIDE" \
  --num-workers 8 --log-every 50
EOF
echo "ll repro done: ${OUT_DIR}"
