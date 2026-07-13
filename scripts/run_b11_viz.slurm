#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__b11_viz
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=01:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/b11_viz_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/b11_viz_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"

FIG_DIR="${FIG_DIR:-${PROJECT_ROOT}/logs/figures}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/generated/hdepic_lora_encoder_lora_gaze_pose_matrix_20head_lowlr_fullsmoke.yaml}"

mkdir -p "${FIG_DIR}/adapter_input" "${FIG_DIR}/smoke_metrics" "${PROJECT_ROOT}/logs"

export PROJECT_ROOT ENV_NAME CONFIG FIG_DIR

payload=$(cat <<'BASH'
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/vjepa2:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"

echo "[b11-viz] plotting smoke metrics"
python "${PROJECT_ROOT}/scripts/plot_hdepic_smoke_metrics.py" --out-dir "${FIG_DIR}/smoke_metrics"

echo "[b11-viz] adapter input (5ch x 384x384)"
python "${PROJECT_ROOT}/scripts/audit_b11_pose_channels.py" \
    --config "${CONFIG}" \
    --split val \
    --batches 3 \
    --batch-size 2 \
    --num-workers 0 \
    --save-viz-dir "${FIG_DIR}/adapter_input"

echo "[b11-viz] done -> ${FIG_DIR}"
BASH
)

singularity exec --nv \
    --overlay "${OVERLAY}:ro" \
    "${SIF}" \
    /bin/bash -lc "${payload}"
