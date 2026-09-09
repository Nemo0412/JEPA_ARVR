#!/bin/bash
# [ATTN-CORNER-SINK] Regenerate no-register DINOv2 hflip maps with per-image dump.
#SBATCH --job-name=VJEPA2-EXP__dino_hflip_dump
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:10:00

set -euo pipefail
SHARE_GPU="${SHARE_GPU:-1}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"
DATA_ROOT=${DATA_ROOT}
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}}"
IMAGE_DIR="${DATA_ROOT}/data/ijepa_probe_imgs/nonego"
OUT_DIR="${OUT_DIR:-${DATA_ROOT}/outputs/attn_corner_sink/hfvit_dinov2_hflip_dump}"
N_IMAGES="${N_IMAGES:-36}"
MODEL_ID=facebook/dinov2-large
mkdir -p "${OUT_DIR}"

source "${CODE_ROOT}/scripts/egtea/gpu_monitor.sh"
start_gpu_monitor "${OUT_DIR}/gpu_${SLURM_JOB_ID}.csv" 5
trap stop_gpu_monitor EXIT INT TERM

export CODE_ROOT IMAGE_DIR OUT_DIR MODEL_ID N_IMAGES PYTHONNOUSERSITE=1
share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
export HF_HOME=${HF_HOME} HF_HUB_OFFLINE=1
cd "${CODE_ROOT}"
python app/hdepic_lora_action_anticipation/analyze_hf_vit_corner.py \
  --model-id "${MODEL_ID}" \
  --img-dir "${IMAGE_DIR}" \
  --out-dir "${OUT_DIR}" \
  --n-images "${N_IMAGES}" \
  --sweep --hflip --dump-per-sample
EOF

stop_gpu_monitor
report_gpu_monitor
trap - EXIT INT TERM
