#!/bin/bash
# [ATTN-CORNER-SINK] B18 · causal probe #0: patch-embed border effect. Slurm only.
#SBATCH --job-name=VJEPA2-EXP__pe_border
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=384G
#SBATCH --time=00:30:00
set -euo pipefail
SHARE_GPU="${SHARE_GPU:-1}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"
DATA_ROOT=${DATA_ROOT}
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}}"
export VJEPA_ROOT
CKPT="${DATA_ROOT}/checkpoints/vjepa2_1_vitl_dist_vitG_384.pt"
VAL="${DATA_ROOT}/data/egtea/vjepa_annotations/stream_half_split/split1/EGTEA_val_stream_mtp.csv"
VID="${DATA_ROOT}/data/egtea/stream_session_videos"
OUT="${DATA_ROOT}/outputs/attn_corner_sink/patch_embed_border/border.json"
export DATA_ROOT CODE_ROOT VJEPA_ROOT CKPT VAL VID OUT
share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
export SHARED_PROJECT_ROOT="$DATA_ROOT" PROJECT_ROOT="$CODE_ROOT"
export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:\${PYTHONPATH:-}"
cd "$CODE_ROOT"
python app/hdepic_lora_action_anticipation/probe_patch_embed_border.py \
  --checkpoint "$CKPT" --val-csv "$VAL" --video-root "$VID" --out-json "$OUT" \
  --img-size 384 --max-frames 32 --context-sec 4 --n-eval 20 --border 3
EOF
