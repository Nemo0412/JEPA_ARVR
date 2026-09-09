#!/bin/bash
# [ATTN-CORNER-SINK] B18 · Cheap EGTEA diagnostic: online-vs-offline predictor-score keep-set
# overlap + topK(mean)!=mean(topK) evidence. ~64 clips, encoder+1 predictor pass each, no clf.
#SBATCH --job-name=VJEPA2-EXP__b18_diag_on_off
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
NCLIPS="${NCLIPS:-64}"
ANN_DIR="${DATA_ROOT}/data/egtea/vjepa_annotations/stream_16s_split/split1"
VAL_CSV="${ANN_DIR}/EGTEA_val_stream_mtp.csv"
VIDEO_ROOT="${DATA_ROOT}/data/egtea/stream_session_videos"
CHECKPOINT="${DATA_ROOT}/checkpoints/vitl.pt"
ENC_LL="${DATA_ROOT}/outputs/egtea_split1_video_enc_ll_exact/action_anticipation_frozen/egtea-split1-video-enc-fixed-vitl16-256-10ep/encoder_lora_best.pt"
PRED_LL="${DATA_ROOT}/outputs/egtea_split1_video_pred_joint_heads_ll_exact/action_anticipation_frozen/egtea-split1-video-pred-joint-heads-vitl16-256-10ep/predictor_lora_best.pt"
PARENT="${DATA_ROOT}/outputs/egtea_stream_mtp/action_anticipation_frozen/egtea-split1-stream-mtp-2-4-6-vitl256-llwarm/best.pt"
CALIB="${DATA_ROOT}/outputs/attn_corner_sink/pred_offline_calib/calib_predblk0_map_64x256.npy"
OUT_DIR="${DATA_ROOT}/outputs/attn_corner_sink/diag_online_offline"
mkdir -p "${OUT_DIR}"
for f in "$CHECKPOINT" "$ENC_LL" "$PRED_LL" "$PARENT" "$VAL_CSV" "$CALIB"; do
  [[ -f "$f" ]] || { echo "ERROR missing $f" >&2; exit 2; }
done
export DATA_ROOT CODE_ROOT VJEPA_ROOT PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NCLIPS VAL_CSV VIDEO_ROOT CHECKPOINT ENC_LL PRED_LL PARENT CALIB OUT_DIR
share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "$CODE_ROOT"; export SHARED_PROJECT_ROOT="$DATA_ROOT" PROJECT_ROOT="$CODE_ROOT"
export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
python -m app.hdepic_lora_action_anticipation.diag_selection_online_vs_offline \
  --checkpoint "$CHECKPOINT" --encoder-lora "$ENC_LL" --predictor-lora "$PRED_LL" --init-from-ckpt "$PARENT" \
  --val-csv "$VAL_CSV" --video-root "$VIDEO_ROOT" --out-dir "$OUT_DIR" --calib-path "$CALIB" \
  --block 0 --keep-count 4096 --n-clips "$NCLIPS" --context-sec 16 --anticipation-sec 2 \
  --img-size 256 --max-frames 128 --fps 8 --src-fps 8
EOF
echo "diag done: ${OUT_DIR}"
