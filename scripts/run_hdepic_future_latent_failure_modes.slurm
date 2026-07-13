#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__hdepic_future_latent_fm
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128GB
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/hdepic_future_latent_fm_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/hdepic_future_latent_fm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

# Dedicated B6 diagnostics wrapper.
# This script is meant to be easy to remove once the three failure-mode
# checks are finished and summarized.

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
VJEPA_ROOT="${VJEPA_ROOT:-${PROJECT_ROOT}/vjepa2}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"

CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/generated/future_latent_compare/hdepic_future_latent_compare.yaml}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/hdepic_lora_action_anticipation/future_latent_failure_modes}"
HORIZON="${HORIZON:-10.0}"
METHODS="${METHODS:-encoder,ar,oracle}"
# When submitting via `sbatch --export=...`, Slurm treats commas as
# separators between env assignments. To make custom method lists robust,
# support semicolon-delimited METHOD values and convert them back to commas.
METHODS="${METHODS//;/,}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-2}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
DROP_INCOMPLETE_HISTORY="${DROP_INCOMPLETE_HISTORY:-1}"
TOPK="${TOPK:-10}"
METRIC_SCOPE="${METRIC_SCOPE:-native}"

mkdir -p "${PROJECT_ROOT}/logs" "${OUT_DIR}"

export PROJECT_ROOT VJEPA_ROOT SIF OVERLAY ENV_NAME
export CONFIG_PATH OUT_DIR HORIZON METHODS EVAL_BATCH_SIZE EVAL_NUM_WORKERS MAX_SAMPLES DROP_INCOMPLETE_HISTORY TOPK METRIC_SCOPE

payload=$(cat <<'BASH'
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}
export PYTHONPATH="${PROJECT_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"

cd "${PROJECT_ROOT}"
python scripts/analyze_future_latent_failure_modes.py \
  --config "${CONFIG_PATH}" \
  --out-dir "${OUT_DIR}" \
  --horizon "${HORIZON}" \
  --methods "${METHODS}" \
  --batch-size "${EVAL_BATCH_SIZE}" \
  --num-workers "${EVAL_NUM_WORKERS}" \
  --max-samples "${MAX_SAMPLES}" \
  $(if [[ "${DROP_INCOMPLETE_HISTORY}" == "1" ]]; then echo "--drop-incomplete-history"; fi) \
  --topk "${TOPK}" \
  --metric-scope "${METRIC_SCOPE}"
BASH
)

singularity exec --nv \
  --overlay "${OVERLAY}:ro" \
  "${SIF}" \
  /bin/bash -lc "${payload}"
