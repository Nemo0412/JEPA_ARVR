#!/bin/bash
set -euo pipefail

# Diagnostic smoke for B11 encoder-LoRA + gaze/pose matrix numerics.
# Uses the regular full-P01 smoke launcher but writes into a fresh tag so
# older fullsmoke outputs stay intact.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LORA_TAG="${LORA_TAG:-hdepic-20head-lora-enclora-gaze-pose-lrscale002-r4-last4-bs2-graddiag-smoke-ep1-i600}"

exec bash "${SCRIPT_DIR}/submit_b11_enclora_20head_lowlr_fullsmoke.sh"
