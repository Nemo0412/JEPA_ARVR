#!/usr/bin/env bash
# Start GPU csv sampler + low-usage alert watcher for D3 training jobs.
#
# Usage:
#   bash scripts/start_d3_gpu_watch.sh 10154559 10154560

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
interval="${MONITOR_INTERVAL:-30}"
duration="${MONITOR_DURATION:-86400}"

jobs=("$@")
if ((${#jobs[@]} == 0)); then
  echo "Usage: $0 JOBID [JOBID ...]" >&2
  exit 1
fi

mkdir -p "${PROJECT_ROOT}/logs"

for jid in "${jobs[@]}"; do
  sampler_log="${PROJECT_ROOT}/logs/gpu_sampler_${jid}.log"
  watch_log="${PROJECT_ROOT}/logs/gpu_usage_monitor_${jid}.log"

  if pgrep -f "monitor_d3_job_gpu.sh.*${jid}" >/dev/null 2>&1; then
    echo "[watch] sampler already running for ${jid}"
  else
    nohup bash "${PROJECT_ROOT}/scripts/monitor_d3_job_gpu.sh" \
      --interval "${interval}" --duration "${duration}" "${jid}" \
      >> "${sampler_log}" 2>&1 &
    echo "[watch] sampler pid=$! job=${jid} -> ${sampler_log}"
  fi

  if pgrep -f "monitor_perf_gpu_usage.sh --job-id ${jid}" >/dev/null 2>&1; then
    echo "[watch] alert watcher already running for ${jid}"
  else
    nohup bash "${PROJECT_ROOT}/scripts/monitor_perf_gpu_usage.sh" \
      --job-id "${jid}" \
      --poll-sec 300 \
      --window-samples 20 \
      --min-avg-util 35 \
      --min-max-util 60 \
      --low-streak-limit 3 \
      --duration-sec "${duration}" \
      >> "${watch_log}" 2>&1 &
    echo "[watch] alert pid=$! job=${jid} -> ${watch_log}"
  fi
done

echo "[watch] CSV: ${PROJECT_ROOT}/logs/perf_<jobid>_nvidia_smi.csv"
