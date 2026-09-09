#!/bin/bash
# [ATTN-CORNER-SINK] Generic HF ViT corner-sink control (DINOv2 ±reg / CLIP / ViT).
#SBATCH --job-name=VJEPA2-EXP__hfvit_corner
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:25:00
set -euo pipefail
SHARE_GPU="${SHARE_GPU:-1}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"
CODE=${PROJECT_ROOT}
MODEL_ID="${MODEL_ID:?}"; OUT_DIR="${OUT_DIR:?}"; IMG_DIR="${IMG_DIR:-${DATA_ROOT}/data/ijepa_probe_imgs/nonego}"
HFLIP_FLAG=""; [[ -n "${HFLIP:-}" ]] && HFLIP_FLAG="--hflip"
[[ -n "${DUMP:-}" ]] && HFLIP_FLAG="${HFLIP_FLAG} --dump-per-sample"
export CODE MODEL_ID OUT_DIR IMG_DIR HFLIP_FLAG
share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
export HF_HOME=${HF_HOME} HF_HUB_OFFLINE=1
cd "$CODE"
python app/hdepic_lora_action_anticipation/analyze_hf_vit_corner.py \
  --model-id "$MODEL_ID" --img-dir "$IMG_DIR" --out-dir "$OUT_DIR" --sweep ${HFLIP_FLAG}
EOF
