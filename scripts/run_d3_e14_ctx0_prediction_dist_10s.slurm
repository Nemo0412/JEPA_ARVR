#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__D3-E14-pred
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128GB
#SBATCH --time=04:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/d3_e14_ctx0_prediction_dist_10s_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/d3_e14_ctx0_prediction_dist_10s_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

# D3 E14: ctx0 + direct_rope top-1 prediction distribution @ 10s.

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
VJEPA_ROOT="${VJEPA_ROOT:-${PROJECT_ROOT}/vjepa2}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"

CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/generated/future_latent_compare/hdepic_future_latent_compare.yaml}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/hdepic_lora_action_anticipation/d3_e14_ctx0_prediction_dist_10s}"
BASELINE_SAMPLE_HITS="${BASELINE_SAMPLE_HITS:-${PROJECT_ROOT}/outputs/hdepic_lora_action_anticipation/future_latent_failure_modes_native/B1-clean-10s-input-ablation/sample_hits.csv}"
HORIZON="${HORIZON:-10.0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-2}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
DROP_INCOMPLETE_HISTORY="${DROP_INCOMPLETE_HISTORY:-1}"
BASELINE_HEAD="${BASELINE_HEAD:-17}"

mkdir -p "${PROJECT_ROOT}/logs" "${OUT_DIR}"

export PROJECT_ROOT VJEPA_ROOT SIF OVERLAY ENV_NAME
export CONFIG_PATH OUT_DIR BASELINE_SAMPLE_HITS HORIZON EVAL_BATCH_SIZE EVAL_NUM_WORKERS
export MAX_SAMPLES DROP_INCOMPLETE_HISTORY BASELINE_HEAD

payload=$(cat <<'BASH'
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}
export PYTHONPATH="${PROJECT_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"

cd "${PROJECT_ROOT}"
python scripts/analyze_d3_e14_ctx0_prediction_dist.py \
  --config "${CONFIG_PATH}" \
  --out-dir "${OUT_DIR}" \
  --horizon "${HORIZON}" \
  --batch-size "${EVAL_BATCH_SIZE}" \
  --num-workers "${EVAL_NUM_WORKERS}" \
  --max-samples "${MAX_SAMPLES}" \
  --baseline-sample-hits "${BASELINE_SAMPLE_HITS}" \
  --baseline-head "${BASELINE_HEAD}" \
  $(if [[ "${DROP_INCOMPLETE_HISTORY}" == "1" ]]; then echo "--drop-incomplete-history"; fi)
BASH
)

singularity exec --nv \
  --overlay "${OVERLAY}:ro" \
  "${SIF}" \
  /bin/bash -lc "${payload}"
