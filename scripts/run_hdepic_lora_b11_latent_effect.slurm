#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__b11_latent_effect
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --time=02:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/hdepic_lora_b11_latent_effect_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/hdepic_lora_b11_latent_effect_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

# B11 adapter latent-effect diagnostic (same protocol as B5 check_binary_adapter_latent_effect).

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
VJEPA_ROOT="${VJEPA_ROOT:-${PROJECT_ROOT}/vjepa2}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-16}"
DROP_INCOMPLETE_HISTORY="${DROP_INCOMPLETE_HISTORY:-1}"
BASE_OUT="${BASE_OUT:-${PROJECT_ROOT}/outputs/hdepic_lora_action_anticipation/binary_adapter_latent_effect_b11_matrix_5ep}"

mkdir -p "${PROJECT_ROOT}/logs" "${BASE_OUT}"

export PROJECT_ROOT VJEPA_ROOT SIF OVERLAY ENV_NAME
export EVAL_BATCH_SIZE EVAL_NUM_WORKERS MAX_SAMPLES DROP_INCOMPLETE_HISTORY SCRIPT_PATH

payload=$(cat <<'BASH'
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}
export PYTHONPATH="${PROJECT_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"
SCRIPT_PATH="${SCRIPT_PATH:-scripts/check_binary_adapter_latent_effect.py}"

for H in 1.0 10.0; do
  if [[ "${H}" == "1.0" ]]; then CFG_SUFFIX=1s; else CFG_SUFFIX=10s; fi
  export CONFIG_PATH="${PROJECT_ROOT}/configs/generated/future_latent_compare/path_y_b11_matrix_${CFG_SUFFIX}.yaml"
  export OUT_DIR="${BASE_OUT}/horizon_${CFG_SUFFIX}"
  export HORIZON="${H}"
  echo ""
  echo "========== B11 latent effect horizon=${H} -> ${OUT_DIR} =========="
  python "${SCRIPT_PATH}" \
    --config "${CONFIG_PATH}" \
    --out-dir "${OUT_DIR}" \
    --horizon "${HORIZON}" \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --num-workers "${EVAL_NUM_WORKERS}" \
    --max-samples "${MAX_SAMPLES}" \
    $(if [[ "${DROP_INCOMPLETE_HISTORY}" == "1" ]]; then echo "--drop-incomplete-history"; fi)
done
BASH
)

export BASE_OUT
singularity exec --nv \
  --overlay "${OVERLAY}:ro" \
  "${SIF}" \
  /bin/bash -lc "${payload}"
