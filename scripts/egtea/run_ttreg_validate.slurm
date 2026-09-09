#!/bin/bash
# [ATTN-CORNER-SINK] Validate test-time-registers (no-train) on DINOv2 via Slurm.
#SBATCH --job-name=VJEPA2-EXP__ttreg_validate
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
HUB=${DATA_ROOT}/refer_repo/test-time-registers
IMG=${DATA_ROOT}/data/ijepa_probe_imgs/nonego
OUT=${DATA_ROOT}/outputs/attn_corner_sink/ttreg/validate.json
export CODE HUB IMG OUT
share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
export TORCH_HOME=${TORCH_HOME} HF_HUB_OFFLINE=1
cd "$CODE"
python app/hdepic_lora_action_anticipation/validate_ttreg_dinov2.py --hub-dir "$HUB" --img-dir "$IMG" --out-json "$OUT"
EOF
