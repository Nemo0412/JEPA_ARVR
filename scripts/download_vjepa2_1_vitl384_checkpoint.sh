#!/bin/bash
set -euo pipefail

# Download the V-JEPA2.1 ViT-L 384 checkpoint to the default path expected by
# scripts/run_hdepic_action_anticipation.slurm.
#
# Run on HPC:
#   bash scripts/download_vjepa2_1_vitl384_checkpoint.sh
#
# Optional:
#   CHECKPOINT_DIR=/some/path bash scripts/download_vjepa2_1_vitl384_checkpoint.sh

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/yh6416/VJEPA2-EXP}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_ROOT}/checkpoints}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-vjepa2_1_vitl_dist_vitG_384.pt}"
CHECKPOINT_URL="${CHECKPOINT_URL:-https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${CHECKPOINT_DIR}/${CHECKPOINT_NAME}}"
TMP_PATH="${CHECKPOINT_PATH}.part"

mkdir -p "${CHECKPOINT_DIR}"

echo "Checkpoint directory: ${CHECKPOINT_DIR}"
echo "Checkpoint path     : ${CHECKPOINT_PATH}"
echo "Checkpoint URL      : ${CHECKPOINT_URL}"

if [[ -s "${CHECKPOINT_PATH}" ]]; then
    echo "Checkpoint already exists and is non-empty; skipping download."
    ls -lh "${CHECKPOINT_PATH}"
    exit 0
fi

if command -v wget >/dev/null 2>&1; then
    wget -c "${CHECKPOINT_URL}" -O "${TMP_PATH}"
elif command -v curl >/dev/null 2>&1; then
    curl -L --continue-at - "${CHECKPOINT_URL}" -o "${TMP_PATH}"
else
    echo "Neither wget nor curl is available." >&2
    exit 1
fi

mv "${TMP_PATH}" "${CHECKPOINT_PATH}"
echo "Downloaded checkpoint:"
ls -lh "${CHECKPOINT_PATH}"
