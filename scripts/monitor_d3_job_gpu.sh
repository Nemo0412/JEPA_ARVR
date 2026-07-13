#!/usr/bin/env bash
# Sample GPU metrics from a running SLURM job via `srun --overlap` (compute node).
# Writes logs/perf_<jobid>_nvidia_smi.csv for monitor_perf_gpu_usage.sh.
#
# Usage:
#   bash scripts/monitor_d3_job_gpu.sh 10154559 [10154560 ...]
#   bash scripts/monitor_d3_job_gpu.sh --interval 30 --duration 86400 10154559

set -euo pipefail

interval=30
duration=86400
jobs=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --interval) interval="$2"; shift 2 ;;
    --duration) duration="$2"; shift 2 ;;
    --help|-h)
      sed -n '1,12p' "$0"
      exit 0
      ;;
    *)
      jobs+=("$1")
      shift
      ;;
  esac
done

if ((${#jobs[@]} == 0)); then
  echo "Need at least one SLURM job id" >&2
  exit 1
fi

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
mkdir -p "${PROJECT_ROOT}/logs"

sample_job() {
  local job_id="$1"
  local csv="${PROJECT_ROOT}/logs/perf_${job_id}_nvidia_smi.csv"
  local log="${PROJECT_ROOT}/logs/gpu_sampler_${job_id}.log"
  local start_ts end_ts now elapsed

  start_ts="$(date +%s)"
  end_ts=$((start_ts + duration))

  {
    echo "gpu_sampler start job=${job_id} interval=${interval}s duration=${duration}s csv=${csv}"
  } >> "${log}"

  if [[ ! -f "${csv}" ]]; then
    echo "timestamp,index,utilization.gpu [%],utilization.memory [%],memory.used [MiB],memory.total [MiB],power.draw [W],temperature.gpu" > "${csv}"
  fi

  while squeue -j "${job_id}" -h -o "%T" 2>/dev/null | grep -qiE 'RUNNING|COMPLETING'; do
    now="$(date +%s)"
    elapsed=$((now - start_ts))
    if (( elapsed >= duration )); then
      echo "$(date -Iseconds) gpu_sampler done job=${job_id} reason=duration" >> "${log}"
      return 0
    fi

    if line="$(srun --jobid="${job_id}" --overlap --ntasks=1 --quiet \
      nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
      --format=csv,noheader 2>>"${log}")"; then
      echo "${line}" >> "${csv}"
      util="$(echo "${line}" | awk -F, '{gsub(/ /,"",$3); print $3}')"
      echo "$(date -Iseconds) sample job=${job_id} util_gpu=${util}" >> "${log}"
    else
      echo "$(date -Iseconds) sample job=${job_id} failed" >> "${log}"
    fi
    sleep "${interval}"
  done

  echo "$(date -Iseconds) gpu_sampler done job=${job_id} reason=job_ended" >> "${log}"
}

for jid in "${jobs[@]}"; do
  sample_job "${jid}" &
  echo "Started GPU sampler for job ${jid} (pid $!)"
done

wait
