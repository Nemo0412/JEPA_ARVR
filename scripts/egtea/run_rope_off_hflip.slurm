#!/bin/bash
# [ATTN-CORNER-SINK] B18 · RoPE-off picture across datasets + content(hflip) test. Slurm only.
#SBATCH --job-name=VJEPA2-EXP__ropeoff_hflip
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=384G
#SBATCH --time=00:35:00
set -euo pipefail
SHARE_GPU="${SHARE_GPU:-1}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"
DATA_ROOT=${DATA_ROOT}
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}}"
export VJEPA_ROOT
CKPT="${DATA_ROOT}/checkpoints/vitl.pt"
VARIANT="${VARIANT:-base20}"; DATASET="${DATASET:-egtea}"
VAL="${DATA_ROOT}/data/egtea/vjepa_annotations/stream_half_split/split1/EGTEA_val_stream_mtp.csv"
VID="${DATA_ROOT}/data/egtea/stream_session_videos"
OUT="${DATA_ROOT}/outputs/attn_corner_sink/rope_off_datasets/${VARIANT}_${DATASET}.json"
ENC_LL="${DATA_ROOT}/outputs/egtea_split1_video_enc_ll_exact/action_anticipation_frozen/egtea-split1-video-enc-fixed-vitl16-256-10ep/encoder_lora_best.pt"
PRED_LL="${DATA_ROOT}/outputs/egtea_split1_video_pred_joint_heads_ll_exact/action_anticipation_frozen/egtea-split1-video-pred-joint-heads-vitl16-256-10ep/predictor_lora_best.pt"
PARENT="${DATA_ROOT}/outputs/egtea_stream_mtp/action_anticipation_frozen/egtea-split1-stream-mtp-2-4-6-vitl256-llwarm/best.pt"
FT=""; [[ "${VARIANT}" == "finetuned20" ]] && FT="--encoder-lora ${ENC_LL} --predictor-lora ${PRED_LL} --init-from-ckpt ${PARENT}"
if [[ "${DATASET}" == "kinetics" ]]; then SRC="--video-dir ${DATA_ROOT}/data/nonego_kinetics"; else SRC="--val-csv ${VAL} --video-root ${VID}"; fi
export DATA_ROOT CODE_ROOT VJEPA_ROOT CKPT VARIANT OUT FT SRC
share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
export SHARED_PROJECT_ROOT="$DATA_ROOT" PROJECT_ROOT="$CODE_ROOT"
export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:\${PYTHONPATH:-}"
cd "$CODE_ROOT"
python app/hdepic_lora_action_anticipation/probe_rope_off_hflip.py \
  --variant "$VARIANT" --checkpoint "$CKPT" --out-json "$OUT" ${SRC} ${FT} \
  --img-size 256 --max-frames 32 --context-sec 4 --n-eval 20
EOF
