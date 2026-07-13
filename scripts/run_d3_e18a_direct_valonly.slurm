#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__D3-E18a-valonly
#SBATCH --account=your_slurm_account
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=256GB
#SBATCH --time=4:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/d3_e18a_direct_valonly_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/d3_e18a_direct_valonly_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

# D3 E18a: val-only run on existing latest.pt checkpoint (partial epoch 5 weights).
# Runs full ctx scan (ctx0 + ctx16) and saves d3_e18a_direct_ctx_eval.json.
# Config: val_only=true, resume_checkpoint=true, num_epochs=1 (forces is_last=True -> full validate).

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
VJEPA_ROOT="${VJEPA_ROOT:-${PROJECT_ROOT}/vjepa2}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"

CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/generated/d3_train/hdepic-lora-d3-e18a-direct-10s-valonly.yaml}"
DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-65536}"
LORA_VAL_METRIC_SCOPE="${LORA_VAL_METRIC_SCOPE:-native}"

mkdir -p "${PROJECT_ROOT}/logs"

export PROJECT_ROOT VJEPA_ROOT SIF OVERLAY ENV_NAME CONFIG_PATH
export DECORD_EOF_RETRY_MAX LORA_VAL_METRIC_SCOPE

payload=$(cat <<'BASH'
set -euo pipefail
export PYTHONNOUSERSITE=1
export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-65536}"
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}
export PYTHONPATH="${PROJECT_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"

cd "${VJEPA_ROOT}"
eval_cmd=(python -m evals.main --fname "${CONFIG_PATH}" --devices cuda:0 --debugmode true)
echo "[E18a-valonly] Launch: ${eval_cmd[*]}"
"${eval_cmd[@]}"
BASH
)

singularity exec --nv \
  --overlay "${OVERLAY}:ro" \
  "${SIF}" \
  /bin/bash -lc "${payload}"
