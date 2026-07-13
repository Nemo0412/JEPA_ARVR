#!/usr/bin/env bash
set -euo pipefail

# Unattended watchdog for the p01_fixed RGB-only probe-only baseline: monitor the job,
# and relaunch (with RESUME_CHECKPOINT=1) on low-GPU-utilization cancel or OOM.
#
# This run is structurally prone to the cluster's low-GPU-utilization watchdog: the
# encoder is frozen (no encoder-LoRA backward), so GPU compute per step is too small
# to hide the dataloader's ~3.5-4s/batch decode latency. Two attempts (10912829,
# 10912919) were both killed at ~2h12m even at num_workers=10. Rather than fight that
# (raising batch_size would break comparability with the bs8 RGB+gaze/+pose baselines),
# this watchdog just resumes from the last checkpoint and resubmits.
#
# Usage:
#   bash scripts/watch_and_relaunch_p01fixed_rgbonly_probeonly.sh --job-id 10912919
#
# Logs: logs/watchdog_p01fixed_rgbonly_probeonly_<first_jobid>.log

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
SUBMIT_SCRIPT="${SUBMIT_SCRIPT:-${PROJECT_ROOT}/scripts/submit_b11_singleprobe_1s_p01fixed_rgbonly_probeonly_resumable.sh}"
job_id=""
max_relaunch=8
poll_sec=120

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
log_file="${PROJECT_ROOT}/logs/watchdog_p01fixed_rgbonly_probeonly_${first_job_id}.log"
state_file="${PROJECT_ROOT}/logs/watchdog_p01fixed_rgbonly_probeonly_${first_job_id}.state"

exec >>"${log_file}" 2>&1
echo "[watchdog] $(date -Is) start first_job=${first_job_id} max_relaunch=${max_relaunch} poll=${poll_sec}s"

relaunch_count=0
if [[ -f "${state_file}" ]]; then
  # shellcheck disable=SC1090
  source "${state_file}"
fi

classify_failure() {
  local jid="$1"
  local state="$2"
  local state_u
  state_u="$(echo "${state}" | tr '[:lower:]' '[:upper:]')"

  if [[ "${state_u}" == COMPLETED* ]]; then
    echo "ok"
    return 0
  fi

  local err_file="${PROJECT_ROOT}/logs/hdepic_lora_probe_${jid}.err"
  if [[ -f "${err_file}" ]]; then
    if grep -qiE 'oom_kill|out of memory|DataLoader worker.*Killed|CUDA out of memory' "${err_file}"; then
      echo "oom"
      return 0
    fi
    # This cluster's low-GPU-utilization kill leaves no traceback: only the slurmstepd
    # "CANCELLED ... DUE to SIGNAL Terminated" line, with no OOM/Killed signature above it.
    if [[ "${state_u}" == CANCELLED* ]] && grep -q 'DUE to SIGNAL Terminated' "${err_file}" \
        && ! grep -qiE 'Traceback|Error' "${err_file}"; then
      echo "low_gpu"
      return 0
    fi
  fi

  if [[ "${state_u}" == CANCELLED* ]]; then
    # Default unexplained cancellations to low_gpu too: that's been the only observed
    # cause for this job so far, and resuming is cheap/safe either way.
    echo "low_gpu"
    return 0
  fi

  echo "other"
}

while true; do
  state_live="$(squeue -h -j "${job_id}" -o "%T" 2>/dev/null || true)"
  if [[ -n "${state_live}" ]]; then
    echo "[watchdog] $(date -Is) job ${job_id} live state=${state_live}"
    sleep "${poll_sec}"
    continue
  fi

  sacct_line="$(sacct -n -X -j "${job_id}" --format=JobIDRaw,State,ExitCode -P 2>/dev/null | awk -F'|' -v id="${job_id}" '$1==id {print; exit}')"
  if [[ -z "${sacct_line}" ]]; then
    echo "[watchdog] job ${job_id} not in sacct yet"
    sleep "${poll_sec}"
    continue
  fi

  state="$(echo "${sacct_line}" | awk -F'|' '{print $2}')"
  exit_code="$(echo "${sacct_line}" | awk -F'|' '{print $3}')"
  kind="$(classify_failure "${job_id}" "${state}")"
  echo "[watchdog] $(date -Is) terminal job ${job_id}: state=${state} exit=${exit_code} class=${kind}"

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

  submit_out="$(bash "${SUBMIT_SCRIPT}")"
  echo "[watchdog] submit: ${submit_out}"
  new_id="$(echo "${submit_out}" | awk '/Submitted batch job/ {print $4}' | tail -n 1)"
  if [[ -z "${new_id}" ]]; then
    echo "[watchdog] failed to parse new job id"
    exit 3
  fi
  job_id="${new_id}"
  sleep "${poll_sec}"
done
