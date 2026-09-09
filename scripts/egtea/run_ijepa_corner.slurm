#!/bin/bash
# [ATTN-CORNER-SINK] I-JEPA corner-sink control. Uses the jepa-3dgs transformers
# env (can3tok). IMG_DIR + OUT_DIR set per invocation.
#SBATCH --job-name=VJEPA2-EXP__ijepa_corner
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
set -euo pipefail
SHARE_GPU="${SHARE_GPU:-1}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"
CODE=${PROJECT_ROOT}
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
IMG_DIR="${IMG_DIR:?set IMG_DIR}"
OUT_DIR="${OUT_DIR:?set OUT_DIR}"
BLOCK="${BLOCK:--1}"
HFLIP_FLAG=""; [[ -n "${HFLIP:-}" ]] && HFLIP_FLAG="--hflip"
[[ -n "${DUMP:-}" ]] && HFLIP_FLAG="${HFLIP_FLAG} --dump-per-sample"
export CODE IMG_DIR OUT_DIR BLOCK HFLIP_FLAG
share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
export HF_HOME=${HF_HOME} HF_HUB_OFFLINE=1
cd "$CODE"
python app/hdepic_lora_action_anticipation/analyze_ijepa_corner.py \
  --img-dir "$IMG_DIR" --out-dir "$OUT_DIR" --block "$BLOCK" --sweep ${HFLIP_FLAG}
EOF
