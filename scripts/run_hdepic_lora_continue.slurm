#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__hdepic_lora_cont
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128GB
#SBATCH --time=48:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/hdepic_lora_continue_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/hdepic_lora_continue_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

# Continue the current clean P01 LoRA probe run from its latest checkpoint.
# The upstream eval treats EVAL_NUM_EPOCHS as the total epoch count, not the
# number of additional epochs. If latest.pt stores epoch 3 and EVAL_NUM_EPOCHS=10,
# the job runs epochs 4..10.

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
LORA_TAG="${LORA_TAG:-hdepic-lora-vitg16-384-cleanp01-refmetrics}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/hdepic_lora_action_anticipation}"
RUN_DIR="${OUTPUT_DIR}/action_anticipation_frozen/${LORA_TAG}"
LATEST="${RUN_DIR}/latest.pt"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-${RUN_DIR}/snapshots}"
SNAPSHOT_NAME="${SNAPSHOT_NAME:-best_clean_p01_before_continue.pt}"

EVAL_NUM_EPOCHS="${EVAL_NUM_EPOCHS:-10}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-2}"

mkdir -p "${PROJECT_ROOT}/logs" "${SNAPSHOT_DIR}"

echo "========================================"
echo "Mode       : continue LoRA training"
echo "Job ID     : ${SLURM_JOB_ID:-unknown}"
echo "Node       : $(hostname)"
echo "Start      : $(date)"
echo "Run dir    : ${RUN_DIR}"
echo "Latest     : ${LATEST}"
echo "Total epoch: ${EVAL_NUM_EPOCHS}"
echo "Batch      : ${EVAL_BATCH_SIZE}"
echo "Workers    : ${EVAL_NUM_WORKERS}"
echo "========================================"

if [[ ! -f "${LATEST}" ]]; then
    echo "[continue] Missing checkpoint: ${LATEST}" >&2
    echo "[continue] Run scripts/run_hdepic_lora_probe.slurm first or set LORA_TAG/OUTPUT_DIR." >&2
    exit 1
fi

snapshot_path="${SNAPSHOT_DIR}/${SNAPSHOT_NAME}"
if [[ ! -f "${snapshot_path}" ]]; then
    cp -p "${LATEST}" "${snapshot_path}"
    echo "[continue] Preserved current latest checkpoint at ${snapshot_path}"
else
    echo "[continue] Snapshot already exists, leaving it untouched: ${snapshot_path}"
fi

export PROJECT_ROOT OUTPUT_DIR LORA_TAG EVAL_NUM_EPOCHS EVAL_BATCH_SIZE EVAL_NUM_WORKERS
export RESUME_CHECKPOINT=1

bash "${PROJECT_ROOT}/scripts/run_hdepic_lora_probe.slurm"
