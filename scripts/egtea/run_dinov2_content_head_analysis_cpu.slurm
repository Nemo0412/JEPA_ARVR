#!/bin/bash
# [ATTN-CORNER-SINK] CPU-only postprocess of saved no-register DINOv2 attention maps.
#SBATCH --job-name=VJEPA2-EXP__dino_content_heads
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00

set -euo pipefail
SHARE_GPU="${SHARE_GPU:-0}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"
DATA_ROOT=${DATA_ROOT}
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}}"
ORIG_DIR="${DATA_ROOT}/outputs/attn_corner_sink/hfvit_dinov2"
FLIP_DIR="${DATA_ROOT}/outputs/attn_corner_sink/hfvit_dinov2_hflip_dump"
IMAGE_DIR="${DATA_ROOT}/data/ijepa_probe_imgs/nonego"
OUT_DIR="${DATA_ROOT}/outputs/attn_corner_sink/dinov2_no_register_content_heads"
mkdir -p "${OUT_DIR}"

export CODE_ROOT ORIG_DIR FLIP_DIR IMAGE_DIR OUT_DIR PYTHONNOUSERSITE=1
share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
export HF_HOME=${HF_HOME} HF_HUB_OFFLINE=1
cd "${CODE_ROOT}"
python -m py_compile app/hdepic_lora_action_anticipation/analyze_dinov2_content_heads.py
python app/hdepic_lora_action_anticipation/analyze_dinov2_content_heads.py \
  --orig-dir "${ORIG_DIR}" \
  --flip-dir "${FLIP_DIR}" \
  --image-dir "${IMAGE_DIR}" \
  --out-dir "${OUT_DIR}"
EOF
