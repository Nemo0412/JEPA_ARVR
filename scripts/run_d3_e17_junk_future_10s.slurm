#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__D3-E17-junk
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128GB
#SBATCH --time=06:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/d3_e17_junk_future_10s_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/d3_e17_junk_future_10s_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
VJEPA_ROOT="${VJEPA_ROOT:-${PROJECT_ROOT}/vjepa2}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"

CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/generated/future_latent_compare/hdepic_future_latent_compare.yaml}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/hdepic_lora_action_anticipation/d3_e17_junk_future_10s}"
E11_STATS_DIR="${E11_STATS_DIR:-${PROJECT_ROOT}/outputs/hdepic_lora_action_anticipation/d3_e11_latent_distribution_10s}"
HORIZON="${HORIZON:-10.0}"
CONTEXT_CHUNKS="${CONTEXT_CHUNKS:-16,0}"
FUTURE_MODES="${FUTURE_MODES:-direct_rope_10s,oracle_target,zero,gaussian,unrelated_oracle,observed_tail_repeat,direct_rope_1s}"

mkdir -p "${PROJECT_ROOT}/logs" "${OUT_DIR}"
export PROJECT_ROOT VJEPA_ROOT SIF OVERLAY ENV_NAME
export CONFIG_PATH OUT_DIR E11_STATS_DIR HORIZON CONTEXT_CHUNKS FUTURE_MODES
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}" EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-2}"
export MAX_SAMPLES="${MAX_SAMPLES:-0}" DROP_INCOMPLETE_HISTORY="${DROP_INCOMPLETE_HISTORY:-1}"
export METRIC_SCOPE="${METRIC_SCOPE:-native}"

payload=$(cat <<'BASH'
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}
export PYTHONPATH="${PROJECT_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"
python scripts/analyze_d3_e17_junk_future_ctx16.py \
  --config "${CONFIG_PATH}" \
  --out-dir "${OUT_DIR}" \
  --e11-stats-dir "${E11_STATS_DIR}" \
  --horizon "${HORIZON}" \
  --context-chunks "${CONTEXT_CHUNKS}" \
  --future-modes "${FUTURE_MODES}" \
  --batch-size "${EVAL_BATCH_SIZE}" \
  --num-workers "${EVAL_NUM_WORKERS}" \
  --max-samples "${MAX_SAMPLES}" \
  --metric-scope "${METRIC_SCOPE}" \
  $(if [[ "${DROP_INCOMPLETE_HISTORY}" == "1" ]]; then echo "--drop-incomplete-history"; fi)
BASH
)

singularity exec --nv --overlay "${OVERLAY}:ro" "${SIF}" /bin/bash -lc "${payload}"
