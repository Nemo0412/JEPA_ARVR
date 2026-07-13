#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__D3-E2-direct-rope
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128GB
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/d3_e2_direct_rope_scale_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/d3_e2_direct_rope_scale_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

# D3 E2: direct mask + RoPE temporal NTK scaling @ 1/3/5/8/10s.
# Baseline direct_single/direct_dense and scaled direct_*_rope variants.

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
VJEPA_ROOT="${VJEPA_ROOT:-${PROJECT_ROOT}/vjepa2}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"

CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/generated/future_latent_compare/hdepic_future_latent_compare.yaml}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/hdepic_lora_action_anticipation/d3_e2_direct_rope_scale}"
RESULT_CSV="${RESULT_CSV:-${OUT_DIR}/future_latent_compare.csv}"
HORIZONS="${HORIZONS:-1,3,5,8,10}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-2}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
DROP_INCOMPLETE_HISTORY="${DROP_INCOMPLETE_HISTORY:-1}"
METRIC_SCOPE="${METRIC_SCOPE:-native}"
HEAD_SELECTION="${HEAD_SELECTION:-vjepa2}"

mkdir -p "${PROJECT_ROOT}/logs" "${OUT_DIR}"

export PROJECT_ROOT VJEPA_ROOT SIF OVERLAY ENV_NAME
export CONFIG_PATH OUT_DIR RESULT_CSV HORIZONS EVAL_BATCH_SIZE EVAL_NUM_WORKERS
export MAX_SAMPLES DROP_INCOMPLETE_HISTORY METRIC_SCOPE HEAD_SELECTION

payload=$(cat <<'BASH'
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}
export PYTHONPATH="${PROJECT_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"

cd "${PROJECT_ROOT}"
python -m app.hdepic_lora_action_anticipation.future_latent_compare \
  --config "${CONFIG_PATH}" \
  --horizons "${HORIZONS}" \
  --out "${RESULT_CSV}" \
  --batch-size "${EVAL_BATCH_SIZE}" \
  --num-workers "${EVAL_NUM_WORKERS}" \
  --max-samples "${MAX_SAMPLES}" \
  --metric-scope "${METRIC_SCOPE}" \
  --head-selection "${HEAD_SELECTION}" \
  --include-rope-scale \
  $(if [[ "${DROP_INCOMPLETE_HISTORY}" == "1" ]]; then echo "--drop-incomplete-history"; fi)
BASH
)

singularity exec --nv \
  --overlay "${OVERLAY}:ro" \
  "${SIF}" \
  /bin/bash -lc "${payload}"
