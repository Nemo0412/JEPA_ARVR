#!/usr/bin/env bash
set -euo pipefail

# Submit full P01 train (w10 default) and start unattended watchdog on login/submit node.
#
# Usage:
#   nohup bash scripts/launch_binary_pose_train_watchdog.sh &
#   bash scripts/launch_binary_pose_train_watchdog.sh --job-id 9946241   # attach to existing

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
attach_job=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-id) attach_job="$2"; shift 2 ;;
    *)
      echo "Usage: $0 [--job-id existing_slurm_id]" >&2
      exit 1
      ;;
  esac
done

if [[ -n "${attach_job}" ]]; then
  job_id="${attach_job}"
  echo "[launch] attaching watchdog to existing job ${job_id}"
else
  submit_out="$(RELAUNCH_COUNT=0 bash "${PROJECT_ROOT}/scripts/submit_binary_pose_train_adaptive.sh")"
  echo "${submit_out}"
  job_id="$(echo "${submit_out}" | awk '/Submitted batch job/ {print $4}' | tail -n 1)"
  if [[ -z "${job_id}" ]]; then
    echo "[launch] failed to parse job id" >&2
    exit 1
  fi
fi

nohup bash "${PROJECT_ROOT}/scripts/watch_and_relaunch_binary_pose_train.sh" \
  --job-id "${job_id}" \
  --max-relaunch 5 \
  --poll-sec 120 \
  >"${PROJECT_ROOT}/logs/watchdog_binary_pose_train_${job_id}.log" 2>&1 &

echo "[launch] watchdog pid=$! job=${job_id}"
echo "[launch] tail -f ${PROJECT_ROOT}/logs/watchdog_binary_pose_train_${job_id}.log"
