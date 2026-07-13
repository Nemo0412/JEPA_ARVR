#!/bin/bash
set -euo pipefail

# Download official V-JEPA2 ViT-L checkpoint (PhD JEPA_ARVR: vitl.pt).
#
#   bash scripts/download_vitl_checkpoint.sh

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_ROOT}/checkpoints}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-vitl.pt}"
CHECKPOINT_URL="${CHECKPOINT_URL:-https://dl.fbaipublicfiles.com/vjepa2/vitl.pt}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${CHECKPOINT_DIR}/${CHECKPOINT_NAME}}"
TMP_PATH="${CHECKPOINT_PATH}.part"

mkdir -p "${CHECKPOINT_DIR}"

echo "Checkpoint path: ${CHECKPOINT_PATH}"
echo "Checkpoint URL : ${CHECKPOINT_URL}"

if [[ -s "${CHECKPOINT_PATH}" ]]; then
    echo "Checkpoint already exists; skipping download."
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
ls -lh "${CHECKPOINT_PATH}"
