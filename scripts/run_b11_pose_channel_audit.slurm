#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__b11_pose_audit
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=01:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/b11_pose_audit_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/b11_pose_audit_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"

CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/generated/hdepic_lora_encoder_lora_gaze_pose_matrix_20head_lowlr_fulltrain.yaml}"
SPLIT="${SPLIT:-val}"
BATCHES="${BATCHES:-4}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DEBUG_SUBSET_PATH="${DEBUG_SUBSET_PATH:-}"
JSON_OUT="${JSON_OUT:-1}"
OUT_JSON="${OUT_JSON:-${PROJECT_ROOT}/logs/b11_pose_audit_${SLURM_JOB_ID:-manual}.json}"

mkdir -p "${PROJECT_ROOT}/logs"

export PROJECT_ROOT ENV_NAME CONFIG SPLIT BATCHES BATCH_SIZE NUM_WORKERS DEBUG_SUBSET_PATH JSON_OUT OUT_JSON

payload=$(cat <<'BASH'
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/vjepa2:${PYTHONPATH:-}"

cd "${PROJECT_ROOT}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "[b11-pose-audit] ERROR: config not found: ${CONFIG}" >&2
    exit 1
fi
if [[ ! -f "${PROJECT_ROOT}/scripts/audit_b11_pose_channels.py" ]]; then
    echo "[b11-pose-audit] ERROR: audit script not found under PROJECT_ROOT/scripts" >&2
    exit 1
fi

args=(
    --config "${CONFIG}"
    --split "${SPLIT}"
    --batches "${BATCHES}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
)
if [[ -n "${DEBUG_SUBSET_PATH}" ]]; then
    args+=(--debug-subset-path "${DEBUG_SUBSET_PATH}")
fi
if [[ "${JSON_OUT}" == "1" ]]; then
    args+=(--json)
fi

echo "[b11-pose-audit] CONFIG=${CONFIG}"
echo "[b11-pose-audit] SPLIT=${SPLIT} BATCHES=${BATCHES} BATCH_SIZE=${BATCH_SIZE} NUM_WORKERS=${NUM_WORKERS}"
python "${PROJECT_ROOT}/scripts/audit_b11_pose_channels.py" "${args[@]}" | tee "${OUT_JSON}"
echo "[b11-pose-audit] wrote ${OUT_JSON}"
BASH
)

singularity exec --nv \
    --overlay "${OVERLAY}:ro" \
    "${SIF}" \
    /bin/bash -lc "${payload}"
