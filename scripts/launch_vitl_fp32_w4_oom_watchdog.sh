#!/usr/bin/env bash
set -euo pipefail

# Launch cgroup-OOM watchdogs for ViT-L fp32 bs8 w4 fulltrain (1s + AR10s).
#
# Default: submit a Slurm watchdog job (survives SSH disconnect).
# Login-node nohup (--login-nohup) is fragile when you log out.
#
# Usage:
#   bash scripts/launch_vitl_fp32_w4_oom_watchdog.sh \
#     --job-1s 10806392 --job-ar10s 10806867
#
# Logs:
#   Slurm: logs/vitl_fp32_oom_watchdog_<watchdog_jobid>.{out,err}
#   Per-track: logs/watchdog_vitl_fp32_{1s,ar10s}_<train_jobid>.log

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
WATCH_SCRIPT="${PROJECT_ROOT}/scripts/watch_vitl_fp32_bs8_oom_fallback.sh"

job_1s="${JOB_1S:-10806392}"
job_ar10s=""
cancel_job=""
resubmit_ar10s_w4=0
poll_sec=120
via_slurm=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-1s) job_1s="$2"; shift 2 ;;
    --job-ar10s) job_ar10s="$2"; shift 2 ;;
    --cancel-job) cancel_job="$2"; shift 2 ;;
    --resubmit-ar10s-w4) resubmit_ar10s_w4=1; shift ;;
    --poll-sec) poll_sec="$2"; shift 2 ;;
    --login-nohup) via_slurm=0; shift ;;
    --via-slurm) via_slurm=1; shift ;;
    *)
      echo "[launch-watch] unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -n "${cancel_job}" ]]; then
  echo "[launch-watch] scancel ${cancel_job}"
  scancel "${cancel_job}" || true
fi

if (( resubmit_ar10s_w4 == 1 )); then
  echo "[launch-watch] submitting AR10s ViT-L fp32 bs8 w4 fulltrain"
  submit_out="$(bash "${PROJECT_ROOT}/scripts/submit_b11_singleprobe_ar10s_vitl_fp32_bs8_noac_fulltrain.sh")"
  echo "${submit_out}"
  job_ar10s="$(echo "${submit_out}" | awk '/Submitted batch job/ {print $4}' | tail -n 1)"
  if [[ -z "${job_ar10s}" ]]; then
    echo "[launch-watch] failed to parse AR10s job id" >&2
    exit 1
  fi
fi

if [[ -z "${job_ar10s}" ]]; then
  echo "[launch-watch] --job-ar10s required (or use --resubmit-ar10s-w4)" >&2
  exit 1
fi

if (( via_slurm == 1 )); then
  echo "[launch-watch] submitting Slurm watchdog (SSH-disconnect safe)"
  submit_out="$(bash "${PROJECT_ROOT}/scripts/submit_vitl_fp32_oom_watchdog.sh" \
    --job-1s "${job_1s}" --job-ar10s "${job_ar10s}" --poll-sec "${poll_sec}")"
  echo "${submit_out}"
  wd_id="$(echo "${submit_out}" | awk '/Submitted batch job/ {print $4}' | tail -n 1)"
  echo "[launch-watch] watchdog Slurm job=${wd_id:-?}"
  echo "[launch-watch] tail: logs/vitl_fp32_oom_watchdog_${wd_id:-<id>}.out"
  echo "[launch-watch] training jobs: 1s=${job_1s} ar10s=${job_ar10s}"
  exit 0
fi

start_watch_login() {
  local track="$1"
  local jid="$2"
  local log="${PROJECT_ROOT}/logs/watchdog_vitl_fp32_${track}_${jid}.log"
  if pgrep -f "watch_vitl_fp32_bs8_oom_fallback.sh --track ${track} --job-id ${jid}" >/dev/null 2>&1; then
    echo "[launch-watch] login watchdog already running track=${track} job=${jid}"
    return 0
  fi
  nohup bash "${WATCH_SCRIPT}" \
    --track "${track}" \
    --job-id "${jid}" \
    --poll-sec "${poll_sec}" \
    >>"${log}" 2>&1 &
  disown -h $! 2>/dev/null || true
  echo "[launch-watch] login nohup watchdog track=${track} job=${jid} pid=$! (may die on SSH logout)"
}

start_watch_login "1s" "${job_1s}"
start_watch_login "ar10s" "${job_ar10s}"
