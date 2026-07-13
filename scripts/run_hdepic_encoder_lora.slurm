#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__hdepic_encoder_lora
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=256GB
#SBATCH --time=24:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/hdepic_encoder_lora_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/hdepic_encoder_lora_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

# Encoder-LoRA action anticipation runs (PhD feedback: frozen encoder is why the
# gaze binary-map underperformed). Uses a pre-written config verbatim (no env ->
# config regeneration) so the experiment.lora.encoder_lora block is preserved.
#
# IMPORTANT: code is taken from the hdepic_exp_dev worktree so the new
# encoder_lora implementation is on PYTHONPATH. Data/checkpoints are shared with
# the main checkout via absolute paths inside the config.
#
# Submit (from the worktree):
#   CONFIG_PATH=/path/to/VJEPA2-EXP-hdepic/configs/generated/hdepic_lora_encoder_lora_baseline.yaml \
#     sbatch scripts/run_hdepic_encoder_lora.slurm
#   CONFIG_PATH=/path/to/VJEPA2-EXP-hdepic/configs/generated/hdepic_lora_encoder_lora_gaze_binary.yaml \
#     sbatch scripts/run_hdepic_encoder_lora.slurm
#
# Smoke (1 epoch, capped iters):
#   EVAL_MAX_TRAIN_ITERS=20 CONFIG_PATH=... sbatch scripts/run_hdepic_encoder_lora.slurm

# Code root = this worktree (defaults to the dir this script lives in / ..).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "${SCRIPT_DIR}")}"
VJEPA_ROOT="${VJEPA_ROOT:-${PROJECT_ROOT}/vjepa2}"
# Shared assets (overlay, checkpoints, data) live in the main checkout.
SHARED_ROOT="${SHARED_ROOT:-/path/to/VJEPA2-EXP}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${SHARED_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"

CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/generated/hdepic_lora_encoder_lora_baseline.yaml}"
EVAL_GPUS="${EVAL_GPUS:-1}"
EVAL_DEBUGMODE="${EVAL_DEBUGMODE:-1}"
EVAL_MAX_TRAIN_ITERS="${EVAL_MAX_TRAIN_ITERS:-0}"
ENCODER_LORA_ENABLED="${ENCODER_LORA_ENABLED:-1}"

mkdir -p "${SHARED_ROOT}/logs"

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "[enclora] Missing config: ${CONFIG_PATH}" >&2
    exit 1
fi
if [[ ! -f "${PROJECT_ROOT}/app/hdepic_lora_action_anticipation/encoder_lora.py" ]]; then
    echo "[enclora] Missing encoder_lora.py under ${PROJECT_ROOT}/app/..." >&2
    exit 1
fi

export PROJECT_ROOT VJEPA_ROOT SHARED_ROOT SIF OVERLAY ENV_NAME
export CONFIG_PATH EVAL_GPUS EVAL_DEBUGMODE EVAL_MAX_TRAIN_ITERS ENCODER_LORA_ENABLED

echo "========================================"
echo "Job ID      : ${SLURM_JOB_ID:-not-in-slurm}"
echo "Node        : $(hostname)"
echo "Code root   : ${PROJECT_ROOT}"
echo "Config      : ${CONFIG_PATH}"
echo "Max iters   : ${EVAL_MAX_TRAIN_ITERS}"
echo "========================================"

payload=$(cat <<'BASH'
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}
# Worktree code first on PYTHONPATH so encoder_lora is picked up.
export PYTHONPATH="${PROJECT_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
export EVAL_MAX_TRAIN_ITERS ENCODER_LORA_ENABLED

cd "${PROJECT_ROOT}"
echo "[enclora] Python: $(which python)"

python - <<'PY'
import importlib, sys
print("[enclora] sys.path head:", sys.path[:4])
m = importlib.import_module("app.hdepic_lora_action_anticipation.encoder_lora")
print("[enclora] encoder_lora import OK from", m.__file__)
importlib.import_module("app.hdepic_lora_action_anticipation.eval")
print("[enclora] eval import OK")
PY

cd "${VJEPA_ROOT}"
device_args=()
for ((gpu_idx=0; gpu_idx<${EVAL_GPUS}; gpu_idx++)); do
    device_args+=("cuda:${gpu_idx}")
done
eval_cmd=(python -m evals.main --fname "${CONFIG_PATH}" --devices "${device_args[@]}")
if [[ "${EVAL_DEBUGMODE}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
    eval_cmd+=(--debugmode true)
fi
echo "[enclora] Launch: ${eval_cmd[*]}"
"${eval_cmd[@]}"
BASH
)

singularity exec --nv \
    --overlay "${OVERLAY}:ro" \
    "${SIF}" \
    /bin/bash -lc "${payload}"

echo "[enclora] Done: $(date)"
