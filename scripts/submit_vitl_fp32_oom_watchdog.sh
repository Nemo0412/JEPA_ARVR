#!/usr/bin/env bash
set -euo pipefail

# Submit CPU-only Slurm watchdog (no GPU / no partition / no QOS — SSH-disconnect safe).
#
# Usage:
#   bash scripts/submit_vitl_fp32_oom_watchdog.sh --job-1s 10806392 --job-ar10s 10806867

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
SLURM_SCRIPT="${PROJECT_ROOT}/scripts/run_vitl_fp32_oom_watchdog.slurm"

job_1s=""
job_ar10s=""
poll_sec=120

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-1s) job_1s="$2"; shift 2 ;;
    --job-ar10s) job_ar10s="$2"; shift 2 ;;
    --poll-sec) poll_sec="$2"; shift 2 ;;
    *)
      echo "Usage: $0 --job-1s ID --job-ar10s ID [--poll-sec S]" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${job_1s}" || -z "${job_ar10s}" ]]; then
  echo "Usage: $0 --job-1s ID --job-ar10s ID [--poll-sec S]" >&2
  exit 1
fi

export_csv="ALL,PROJECT_ROOT=${PROJECT_ROOT},JOB_1S=${job_1s},JOB_AR10S=${job_ar10s},POLL_SEC=${poll_sec}"

echo "[submit-watchdog] Slurm watchdog for 1s=${job_1s} ar10s=${job_ar10s} poll=${poll_sec}s"
sbatch --export="${export_csv}" "${SLURM_SCRIPT}"
