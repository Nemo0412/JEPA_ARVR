#!/bin/bash
set -euo pipefail

# Resumable wrapper around submit_b11_vlm_baseline_1s_p01fixed_rgbonly_probeonly_llama32vision.sh
# for unattended relaunch by watch_and_relaunch_vlm_baseline_1s_p01fixed_rgbonly_probeonly_llama32vision.sh.
# Always sets RESUME_CHECKPOINT=1 and pins LORA_TAG to the in-flight run's tag so a
# relaunch picks up outputs/.../<tag>/latest.pt instead of restarting from epoch 0.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LORA_TAG="${LORA_TAG:-hdepic-vlm-llama32vision11b-1s-p01fixed-rgbonly-probeonly-8f4s-pooled-bs8-w10}"
export RESUME_CHECKPOINT=1
exec bash "${SCRIPT_DIR}/submit_b11_vlm_baseline_1s_p01fixed_rgbonly_probeonly_llama32vision.sh"
