#!/usr/bin/env bash
set -euo pipefail

# Unattended watchdog: monitor binary+pose full train, relaunch on OOM or low-GPU cancel.
#
# Usage:
#   bash scripts/watch_and_relaunch_binary_pose_train.sh --job-id 9946241
#   bash scripts/launch_binary_pose_train_watchdog.sh   # submit + watch
#
# Logs: logs/watchdog_binary_pose_train_<first_jobid>.log

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
job_id=""
max_relaunch=5
poll_sec=120
gpu_monitor_duration=86400

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-id) job_id="$2"; shift 2 ;;
    --max-relaunch) max_relaunch="$2"; shift 2 ;;
    --poll-sec) poll_sec="$2"; shift 2 ;;
    *)
      echo "[watchdog] unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${job_id}" ]]; then
  echo "Usage: $0 --job-id <slurm_job_id> [--max-relaunch N] [--poll-sec S]" >&2
  exit 1
fi

first_job_id="${job_id}"
log_file="${PROJECT_ROOT}/logs/watchdog_binary_pose_train_${first_job_id}.log"
state_file="${PROJECT_ROOT}/logs/watchdog_binary_pose_train_${first_job_id}.state"

exec >>"${log_file}" 2>&1
echo "[watchdog] $(date -Is) start first_job=${first_job_id} max_relaunch=${max_relaunch} poll=${poll_sec}s"

relaunch_count=0
if [[ -f "${state_file}" ]]; then
  # shellcheck disable=SC1090
  source "${state_file}"
fi

start_gpu_monitor() {
  local jid="$1"
  local mon_log="${PROJECT_ROOT}/logs/gpu_usage_monitor_${jid}.log"
  if pgrep -f "monitor_perf_gpu_usage.sh --job-id ${jid}" >/dev/null 2>&1; then
    echo "[watchdog] gpu monitor already running for job ${jid}"
    return 0
  fi
  nohup bash "${PROJECT_ROOT}/scripts/monitor_perf_gpu_usage.sh" \
    --job-id "${jid}" \
    --poll-sec 300 \
    --window-samples 20 \
    --low-streak-limit 2 \
    --duration-sec "${gpu_monitor_duration}" \
    >"${mon_log}" 2>&1 &
  echo "[watchdog] started gpu monitor for job ${jid} -> ${mon_log}"
}

classify_failure() {
  local jid="$1"
  local state="$2"
  local reason="$3"
  local state_u reason_u
  state_u="$(echo "${state}" | tr '[:lower:]' '[:upper:]')"
  reason_u="$(echo "${reason}" | tr '[:lower:]' '[:upper:]')"

  if [[ "${state_u}" == COMPLETED* ]]; then
    echo "ok"
    return 0
  fi

  if [[ "${state_u}" == *OUT_OF_MEM* || "${state_u}" == *OOM* ]]; then
    echo "oom"
    return 0
  fi

  local err_file=""
  for f in \
    "${PROJECT_ROOT}/logs/hdepic_lora_binary_pose_train_${jid}.err"; do
    if [[ -f "${f}" ]]; then
      err_file="${f}"
      break
    fi
  done
  if [[ -n "${err_file}" ]] && grep -qiE 'oom_kill|out of memory|DataLoader worker.*Killed' "${err_file}"; then
    echo "oom"
    return 0
  fi

  if [[ "${state_u}" == CANCELLED* && ( "${reason_u}" == *GPU* || "${reason_u}" == *USAGE* || "${reason_u}" == *UTIL* ) ]]; then
    echo "low_gpu"
    return 0
  fi

  local mon_log="${PROJECT_ROOT}/logs/gpu_usage_monitor_${jid}.log"
  if [[ -f "${mon_log}" ]] && grep -q 'LOW_USAGE_ALERT' "${mon_log}"; then
    echo "low_gpu"
    return 0
  fi

  echo "other"
}

while true; do
  state_live="$(squeue -h -j "${job_id}" -o "%T" 2>/dev/null || true)"
  if [[ -n "${state_live}" ]]; then
    echo "[watchdog] $(date -Is) job ${job_id} live state=${state_live}"
    start_gpu_monitor "${job_id}"
    sleep "${poll_sec}"
    continue
  fi

  sacct_line="$(sacct -n -X -j "${job_id}" --format=JobIDRaw,State,ExitCode,Reason -P 2>/dev/null | awk -F'|' -v id="${job_id}" '$1==id {print; exit}')"
  if [[ -z "${sacct_line}" ]]; then
    echo "[watchdog] job ${job_id} not in sacct yet"
    sleep "${poll_sec}"
    continue
  fi

  state="$(echo "${sacct_line}" | awk -F'|' '{print $2}')"
  exit_code="$(echo "${sacct_line}" | awk -F'|' '{print $3}')"
  reason="$(echo "${sacct_line}" | awk -F'|' '{print $4}')"
  kind="$(classify_failure "${job_id}" "${state}" "${reason}")"
  echo "[watchdog] $(date -Is) terminal job ${job_id}: state=${state} exit=${exit_code} reason=${reason} class=${kind}"

  if [[ "${kind}" == "ok" ]]; then
    echo "[watchdog] success, stop"
    exit 0
  fi

  if [[ "${kind}" != "oom" && "${kind}" != "low_gpu" ]]; then
    echo "[watchdog] not auto-relaunchable, stop"
    exit 4
  fi

  if (( relaunch_count >= max_relaunch )); then
    echo "[watchdog] max relaunch ${max_relaunch} reached, stop"
    exit 2
  fi

  relaunch_count=$((relaunch_count + 1))
  echo "relaunch_count=${relaunch_count}" >"${state_file}"
  echo "last_job_id=${job_id}" >>"${state_file}"
  echo "last_kind=${kind}" >>"${state_file}"

  if [[ "${kind}" == "oom" ]]; then
    export FAIL_REASON="${reason} oom_kill"
    unset CANCEL_REASON
  else
    export CANCEL_REASON="${reason}"
    unset FAIL_REASON
  fi

  submit_out="$(RELAUNCH_COUNT="${relaunch_count}" bash "${PROJECT_ROOT}/scripts/submit_binary_pose_train_adaptive.sh")"
  echo "[watchdog] submit: ${submit_out}"
  new_id="$(echo "${submit_out}" | awk '/Submitted batch job/ {print $4}' | tail -n 1)"
  if [[ -z "${new_id}" ]]; then
    echo "[watchdog] failed to parse new job id"
    exit 3
  fi
  job_id="${new_id}"
  sleep "${poll_sec}"
done
