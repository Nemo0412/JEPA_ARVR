#!/bin/bash
# Shared nvidia-smi sampler for B13 GPU launchers.

start_gpu_monitor() {
  local log_path=$1
  local interval_seconds=${2:-30}
  mkdir -p "$(dirname "${log_path}")"
  nvidia-smi \
    --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used \
    --format=csv,noheader,nounits \
    --loop="${interval_seconds}" > "${log_path}" &
  GPU_MONITOR_PID=$!
  GPU_MONITOR_LOG=${log_path}
  GPU_MONITOR_INTERVAL=${interval_seconds}
}

stop_gpu_monitor() {
  if [[ -n "${GPU_MONITOR_PID:-}" ]]; then
    kill "${GPU_MONITOR_PID}" 2>/dev/null || true
    wait "${GPU_MONITOR_PID}" 2>/dev/null || true
    GPU_MONITOR_PID=
  fi
}

report_gpu_monitor() {
  awk -F, '
    { value=$3; gsub(/[[:space:]]/, "", value) }
    value ~ /^[0-9]+([.][0-9]+)?$/ { total += value; count += 1 }
    END {
      if (count == 0) exit 1
      printf "avg_gpu_util=%.2f%% samples=%d interval_seconds=%s log=%s\n", total / count, count, ENVIRON["GPU_MONITOR_INTERVAL"], FILENAME
    }
  ' "${GPU_MONITOR_LOG}"
}
