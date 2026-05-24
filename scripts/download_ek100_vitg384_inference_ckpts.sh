#!/bin/bash
set -euo pipefail

# Download the official V-JEPA2 ViT-g/384 encoder checkpoint and EK100 action
# anticipation attentive-probe checkpoint for val_only inference.
#
# Run on HPC:
#   bash scripts/download_ek100_vitg384_inference_ckpts.sh

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/yh6416/VJEPA2-EXP}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_ROOT}/checkpoints}"
PROBE_ROOT="${PROBE_ROOT:-${PROJECT_ROOT}/outputs/ek100_vitg384_probe}"
PROBE_TAG="${PROBE_TAG:-ek100-vitg16-384}"

ENCODER_NAME="vitg-384.pt"
ENCODER_URL="https://dl.fbaipublicfiles.com/vjepa2/vitg-384.pt"
PROBE_NAME="ek100-vitg-384.pt"
PROBE_URL="https://dl.fbaipublicfiles.com/vjepa2/evals/ek100-vitg-384.pt"

download_file() {
    local url="$1"
    local out="$2"
    local tmp="${out}.part"

    mkdir -p "$(dirname "${out}")"
    if [[ -s "${out}" ]]; then
        echo "Already exists: ${out}"
        ls -lh "${out}"
        return
    fi

    if command -v wget >/dev/null 2>&1; then
        wget -c "${url}" -O "${tmp}"
    elif command -v curl >/dev/null 2>&1; then
        curl -L --continue-at - "${url}" -o "${tmp}"
    else
        echo "Neither wget nor curl is available." >&2
        exit 1
    fi
    mv "${tmp}" "${out}"
    ls -lh "${out}"
}

download_file "${ENCODER_URL}" "${CHECKPOINT_DIR}/${ENCODER_NAME}"

probe_latest="${PROBE_ROOT}/action_anticipation_frozen/${PROBE_TAG}/latest.pt"
download_file "${PROBE_URL}" "${probe_latest}"

echo "Encoder checkpoint: ${CHECKPOINT_DIR}/${ENCODER_NAME}"
echo "Probe checkpoint  : ${probe_latest}"
