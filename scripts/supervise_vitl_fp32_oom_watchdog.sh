#!/usr/bin/env bash
set -euo pipefail

# Run 1s + AR10s cgroup-OOM watchdogs until both tracks finish (for Slurm job body).

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
WATCH_SCRIPT="${PROJECT_ROOT}/scripts/watch_vitl_fp32_bs8_oom_fallback.sh"

job_1s="${JOB_1S:?JOB_1S required}"
job_ar10s="${JOB_AR10S:?JOB_AR10S required}"
poll_sec="${POLL_SEC:-120}"

echo "[supervisor] $(date -Is) start job_1s=${job_1s} job_ar10s=${job_ar10s} poll=${poll_sec}s"

bash "${WATCH_SCRIPT}" --track 1s --job-id "${job_1s}" --poll-sec "${poll_sec}" &
pid_1s=$!
bash "${WATCH_SCRIPT}" --track ar10s --job-id "${job_ar10s}" --poll-sec "${poll_sec}" &
pid_ar10s=$!

wait "${pid_1s}" || rc_1s=$?
wait "${pid_ar10s}" || rc_ar10s=$?
rc_1s="${rc_1s:-0}"
rc_ar10s="${rc_ar10s:-0}"

echo "[supervisor] $(date -Is) done rc_1s=${rc_1s} rc_ar10s=${rc_ar10s}"
if (( rc_1s != 0 || rc_ar10s != 0 )); then
  exit 1
fi
