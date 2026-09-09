#!/bin/bash
# [ATTN-CORNER-SINK] B18 · layer-resolved RoPE ablation + per-block pre/post-RoPE sweep.
# VARIANT-parametrized; default finetuned20 = the PRUNING MODEL (@256), the relevant target.
# Env: VARIANT(finetuned20|base20|base21) IMG_SIZE MAXF CTX ROPE_OFF SWEEP. Slurm only.
#SBATCH --job-name=VJEPA2-EXP__rope_layer_resolved
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
DATA_ROOT=${DATA_ROOT}
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}}"
export VJEPA_ROOT
VARIANT="${VARIANT:-finetuned20}"
if [[ "${VARIANT}" == "base21" ]]; then
  IMG_SIZE="${IMG_SIZE:-384}"; CKPT="${DATA_ROOT}/checkpoints/vjepa2_1_vitl_dist_vitG_384.pt"
else
  IMG_SIZE="${IMG_SIZE:-256}"; CKPT="${DATA_ROOT}/checkpoints/vitl.pt"
fi
MAXF="${MAXF:-32}"; CTX="${CTX:-4}"; ROPE_OFF="${ROPE_OFF:-}"; SWEEP="${SWEEP:-0}"; SCAN="${SCAN:-both}"; SCAN_STEP="${SCAN_STEP:-2}"; PROFILE="${PROFILE:-}"; LAST_MAPS="${LAST_MAPS:-}"
VAL="${DATA_ROOT}/data/egtea/vjepa_annotations/stream_half_split/split1/EGTEA_val_stream_mtp.csv"
VID="${DATA_ROOT}/data/egtea/stream_session_videos"
OUT="${OUT:-${DATA_ROOT}/outputs/attn_corner_sink/rope_layer_resolved_${VARIANT}/rope_layer_resolved.json}"
ENC_LL="${DATA_ROOT}/outputs/egtea_split1_video_enc_ll_exact/action_anticipation_frozen/egtea-split1-video-enc-fixed-vitl16-256-10ep/encoder_lora_best.pt"
PRED_LL="${DATA_ROOT}/outputs/egtea_split1_video_pred_joint_heads_ll_exact/action_anticipation_frozen/egtea-split1-video-pred-joint-heads-vitl16-256-10ep/predictor_lora_best.pt"
PARENT="${DATA_ROOT}/outputs/egtea_stream_mtp/action_anticipation_frozen/egtea-split1-stream-mtp-2-4-6-vitl256-llwarm/best.pt"
EXTRA=""
[[ "${VARIANT}" == "finetuned20" ]] && EXTRA="--encoder-lora ${ENC_LL} --predictor-lora ${PRED_LL} --init-from-ckpt ${PARENT}"
export DATA_ROOT CODE_ROOT VJEPA_ROOT VARIANT IMG_SIZE MAXF CTX ROPE_OFF SWEEP SCAN SCAN_STEP PROFILE LAST_MAPS VAL VID OUT CKPT EXTRA
share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
export SHARED_PROJECT_ROOT="$DATA_ROOT" PROJECT_ROOT="$CODE_ROOT"
export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:\${PYTHONPATH:-}"
cd "$CODE_ROOT"
SWEEP_ARG=""; [[ "$SWEEP" == "1" ]] && SWEEP_ARG="--sweep"
python app/hdepic_lora_action_anticipation/probe_rope_layer_resolved.py \
  --variant "$VARIANT" --checkpoint "$CKPT" --val-csv "$VAL" --video-root "$VID" --out-json "$OUT" \
  --img-size "$IMG_SIZE" --max-frames "$MAXF" --context-sec "$CTX" --n-eval 20 \
  --rope-off "$ROPE_OFF" --scan "$SCAN" --scan-step "$SCAN_STEP" --profile "$PROFILE" --last-maps "$LAST_MAPS" \$SWEEP_ARG ${EXTRA}
EOF
