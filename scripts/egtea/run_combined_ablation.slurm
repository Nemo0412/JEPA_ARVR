#!/bin/bash
# [ATTN-CORNER-SINK] B18 · RoPE x border 2x2 factorial. VARIANT-parametrized. Slurm only.
#SBATCH --job-name=VJEPA2-EXP__combined_abl
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=384G
#SBATCH --time=00:40:00
set -euo pipefail
SHARE_GPU="${SHARE_GPU:-1}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"
DATA_ROOT=${DATA_ROOT}
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}}"
export VJEPA_ROOT
VARIANT="${VARIANT:-base21}"; IMG_SIZE="${IMG_SIZE:-384}"; MAXF="${MAXF:-32}"; CTX="${CTX:-4}"
VAL="${DATA_ROOT}/data/egtea/vjepa_annotations/stream_half_split/split1/EGTEA_val_stream_mtp.csv"
VID="${DATA_ROOT}/data/egtea/stream_session_videos"
OUT="${DATA_ROOT}/outputs/attn_corner_sink/combined_ablation_${VARIANT}/combined.json"
if [[ "${VARIANT}" == "base21" ]]; then CKPT="${DATA_ROOT}/checkpoints/vjepa2_1_vitl_dist_vitG_384.pt"
else CKPT="${DATA_ROOT}/checkpoints/vitl.pt"; fi
ENC_LL="${DATA_ROOT}/outputs/egtea_split1_video_enc_ll_exact/action_anticipation_frozen/egtea-split1-video-enc-fixed-vitl16-256-10ep/encoder_lora_best.pt"
PRED_LL="${DATA_ROOT}/outputs/egtea_split1_video_pred_joint_heads_ll_exact/action_anticipation_frozen/egtea-split1-video-pred-joint-heads-vitl16-256-10ep/predictor_lora_best.pt"
PARENT="${DATA_ROOT}/outputs/egtea_stream_mtp/action_anticipation_frozen/egtea-split1-stream-mtp-2-4-6-vitl256-llwarm/best.pt"
EXTRA=""
[[ "${VARIANT}" == "finetuned20" ]] && EXTRA="--encoder-lora ${ENC_LL} --predictor-lora ${PRED_LL} --init-from-ckpt ${PARENT}"
export DATA_ROOT CODE_ROOT VJEPA_ROOT VARIANT IMG_SIZE MAXF CTX VAL VID OUT CKPT EXTRA
share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
export SHARED_PROJECT_ROOT="$DATA_ROOT" PROJECT_ROOT="$CODE_ROOT"
export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:\${PYTHONPATH:-}"
cd "$CODE_ROOT"
python app/hdepic_lora_action_anticipation/probe_combined_ablation.py \
  --variant "$VARIANT" --checkpoint "$CKPT" --val-csv "$VAL" --video-root "$VID" --out-json "$OUT" \
  --img-size "$IMG_SIZE" --max-frames "$MAXF" --context-sec "$CTX" --n-eval 20 --border 3 ${EXTRA}
EOF
