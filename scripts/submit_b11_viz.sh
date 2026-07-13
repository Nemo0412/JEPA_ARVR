#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
SLURM_SCRIPT="${PROJECT_ROOT}/scripts/run_b11_viz.slurm"

echo "[submit-b11-viz] submitting visualization job"
sbatch "${SLURM_SCRIPT}"
