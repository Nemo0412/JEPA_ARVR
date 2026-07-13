#!/usr/bin/env bash
set -euo pipefail

# Watch a SLURM job; if canceled for low GPU usage, auto-relaunch.
# Usage:
#   bash scripts/watch_and_relaunch_low_gpu_cancel.sh \
#     --job-id 1234567 \
#     --submit "sbatch scripts/run_hdepic_lora_binary_pose_smoke_long.slurm" \
#     --max-relaunch 3 \
#     --poll-sec 90

job_id=""
submit_cmd=""
max_relaunch=2
poll_sec=90

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-id) job_id="$2"; shift 2 ;;
    --submit) submit_cmd="$2"; shift 2 ;;
    --max-relaunch) max_relaunch="$2"; shift 2 ;;
    --poll-sec) poll_sec="$2"; shift 2 ;;
    *)
      echo "[watchdog] Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${job_id}" || -z "${submit_cmd}" ]]; then
  echo "Usage: $0 --job-id <id> --submit \"<sbatch ...>\" [--max-relaunch N] [--poll-sec S]" >&2
  exit 1
fi

if ! [[ "${max_relaunch}" =~ ^[0-9]+$ ]]; then
  echo "[watchdog] --max-relaunch must be integer" >&2
  exit 1
fi
if ! [[ "${poll_sec}" =~ ^[0-9]+$ ]]; then
  echo "[watchdog] --poll-sec must be integer" >&2
  exit 1
fi

relaunch_count=0
echo "[watchdog] start: job=${job_id} max_relaunch=${max_relaunch} poll=${poll_sec}s"
echo "[watchdog] submit cmd: ${submit_cmd}"

while true; do
  # Still in queue/running/pending?
  state_live="$(squeue -h -j "${job_id}" -o "%T" 2>/dev/null || true)"
  if [[ -n "${state_live}" ]]; then
    echo "[watchdog] job ${job_id} live state=${state_live}"
    sleep "${poll_sec}"
    continue
  fi

  # Finished: inspect terminal state + reason.
  sacct_line="$(sacct -n -X -j "${job_id}" --format=JobIDRaw,State,Reason -P 2>/dev/null | awk -F'|' -v id="${job_id}" '$1==id {print; exit}')"
  if [[ -z "${sacct_line}" ]]; then
    echo "[watchdog] job ${job_id} not found in sacct yet; retry"
    sleep "${poll_sec}"
    continue
  fi

  state="$(echo "${sacct_line}" | awk -F'|' '{print $2}')"
  reason="$(echo "${sacct_line}" | awk -F'|' '{print $3}')"
  state_u="$(echo "${state}" | tr '[:lower:]' '[:upper:]')"
  reason_u="$(echo "${reason}" | tr '[:lower:]' '[:upper:]')"
  echo "[watchdog] terminal job ${job_id}: state=${state} reason=${reason}"

  if [[ "${state_u}" == COMPLETED* ]]; then
    echo "[watchdog] completed, stop monitoring"
    exit 0
  fi

  if [[ "${state_u}" == CANCELLED* && ( "${reason_u}" == *GPU* || "${reason_u}" == *USAGE* || "${reason_u}" == *UTIL* ) ]]; then
    if (( relaunch_count >= max_relaunch )); then
      echo "[watchdog] reached max relaunch=${max_relaunch}, stop"
      exit 2
    fi
    relaunch_count=$((relaunch_count + 1))
    echo "[watchdog] low-gpu-style cancel detected, relaunch #${relaunch_count}"
    submit_out="$(RELAUNCH_COUNT="${relaunch_count}" CANCEL_REASON="${reason}" bash -lc "${submit_cmd}")"
    echo "[watchdog] submit output: ${submit_out}"
    new_id="$(echo "${submit_out}" | awk '/Submitted batch job/ {print $4}' | tail -n 1)"
    if [[ -z "${new_id}" ]]; then
      echo "[watchdog] failed to parse new job id" >&2
      exit 3
    fi
    job_id="${new_id}"
    sleep "${poll_sec}"
    continue
  fi

  # Other terminal states: fail/timeout/cancelled-for-other-reason.
  echo "[watchdog] terminal state not auto-relaunched; stop"
  exit 4
done

