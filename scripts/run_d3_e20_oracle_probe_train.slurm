#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__D3-E20-oracle-train
#SBATCH --account=your_slurm_account
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=640GB
#SBATCH --time=24:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/d3_e20_oracle_probe_train_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/d3_e20_oracle_probe_train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

# D3 E20: probe upper-bound test — train [ctx16 ; oracle_target] @ 10s label.
# Warm-start from 1s-direct baseline (cleanp01-refmetrics). Eval ctx0/ctx16 oracle each epoch.

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
VJEPA_ROOT="${VJEPA_ROOT:-${PROJECT_ROOT}/vjepa2}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"

CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/generated/d3_train/hdepic-lora-d3-e20-oracle-10s-train.yaml}"
EVAL_GPUS="${EVAL_GPUS:-1}"
EVAL_NUM_EPOCHS="${EVAL_NUM_EPOCHS:-5}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-8}"
EVAL_VAL_NUM_WORKERS="${EVAL_VAL_NUM_WORKERS:-2}"
D3_EVAL_BATCH_SIZE="${D3_EVAL_BATCH_SIZE:-2}"
D3_EVAL_NUM_WORKERS="${D3_EVAL_NUM_WORKERS:-2}"
D3_SKIP_INTERIM_VALIDATE="${D3_SKIP_INTERIM_VALIDATE:-1}"
DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-65536}"
PERF_MONITOR="${PERF_MONITOR:-1}"
PERF_INTERVAL="${PERF_INTERVAL:-30}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-0}"
LORA_VAL_METRIC_SCOPE="${LORA_VAL_METRIC_SCOPE:-native}"

mkdir -p "${PROJECT_ROOT}/logs"

export PROJECT_ROOT VJEPA_ROOT SIF OVERLAY ENV_NAME CONFIG_PATH
export EVAL_GPUS EVAL_NUM_EPOCHS EVAL_BATCH_SIZE EVAL_NUM_WORKERS EVAL_VAL_NUM_WORKERS
export D3_EVAL_BATCH_SIZE D3_EVAL_NUM_WORKERS D3_SKIP_INTERIM_VALIDATE
export RESUME_CHECKPOINT LORA_VAL_METRIC_SCOPE DECORD_EOF_RETRY_MAX PERF_MONITOR PERF_INTERVAL

monitor_pids=()
stop_perf_monitors() {
  for pid in "${monitor_pids[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
}

if [[ "${PERF_MONITOR}" == "1" ]]; then
  perf_prefix="${PROJECT_ROOT}/logs/perf_${SLURM_JOB_ID:-manual}"
  echo "[perf] Sampling nvidia-smi every ${PERF_INTERVAL}s -> ${perf_prefix}_nvidia_smi.csv"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
      --format=csv -l "${PERF_INTERVAL}" > "${perf_prefix}_nvidia_smi.csv" 2> "${perf_prefix}_nvidia_smi.err" &
    monitor_pids+=("$!")
    trap stop_perf_monitors EXIT
  else
    echo "[perf] nvidia-smi not available" > "${perf_prefix}_nvidia_smi.err"
  fi
fi

payload=$(cat <<'BASH'
set -euo pipefail
export PYTHONNOUSERSITE=1
export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-65536}"
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}
export PYTHONPATH="${PROJECT_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"

cd "${VJEPA_ROOT}"
base_config_path="${CONFIG_PATH}"
runtime_config_dir="${D3_RUNTIME_CONFIG_DIR:-${PROJECT_ROOT}/configs/generated/d3_train/runtime}"
mkdir -p "${runtime_config_dir}"
runtime_config_path="${runtime_config_dir}/$(basename "${base_config_path%.yaml}")-${SLURM_JOB_ID:-manual}.yaml"
export RUNTIME_CONFIG_PATH="${runtime_config_path}"
python - <<'PY'
import os
from pathlib import Path

import yaml

base = Path(os.environ["CONFIG_PATH"])
out = Path(os.environ["RUNTIME_CONFIG_PATH"])
cfg = yaml.safe_load(base.read_text(encoding="utf-8"))
data = cfg["experiment"]["data"]
opt = cfg["experiment"]["optimization"]
lora = cfg["experiment"].get("lora", {})
d3 = lora.get("d3_oracle_concat_train") or lora.get("d3_concat_train")
if d3 is None:
    raise RuntimeError("D3 concat config block not found")

overrides = {}

def set_int(env_name, target, key):
    raw = os.environ.get(env_name, "").strip()
    if raw:
        target[key] = int(raw)
        overrides[key] = int(raw)

set_int("EVAL_BATCH_SIZE", opt, "batch_size")
set_int("EVAL_NUM_EPOCHS", opt, "num_epochs")
set_int("EVAL_NUM_WORKERS", data, "num_workers")
set_int("EVAL_VAL_NUM_WORKERS", data, "val_num_workers")
set_int("D3_EVAL_BATCH_SIZE", d3, "eval_batch_size")
set_int("D3_EVAL_NUM_WORKERS", d3, "eval_num_workers")

skip_interim = os.environ.get("D3_SKIP_INTERIM_VALIDATE", "").strip()
if skip_interim:
    d3["skip_interim_validate"] = skip_interim == "1"
    overrides["skip_interim_validate"] = d3["skip_interim_validate"]
    if d3["skip_interim_validate"]:
        d3.pop("validate_interim_max_samples", None)

interim = os.environ.get("D3_VALIDATE_INTERIM_MAX_SAMPLES", "").strip()
if interim:
    d3["validate_interim_max_samples"] = int(interim)
    d3.pop("skip_interim_validate", None)
    overrides["validate_interim_max_samples"] = int(interim)

cfg["runtime_overrides"] = overrides
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(f"[D3 runtime config] {base} -> {out}; overrides={overrides}")
PY
export CONFIG_PATH="${runtime_config_path}"
device_args=(cuda:0)
eval_cmd=(python -m evals.main --fname "${CONFIG_PATH}" --devices "${device_args[@]}" --debugmode true)
echo "[E20] Launch: ${eval_cmd[*]}"
"${eval_cmd[@]}"
BASH
)

singularity exec --nv \
  --overlay "${OVERLAY}:ro" \
  "${SIF}" \
  /bin/bash -lc "${payload}"
train_rc=$?

trap - EXIT
stop_perf_monitors
exit "${train_rc}"
