#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__vitl_oom_watchdog
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=7-00:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/vitl_fp32_oom_watchdog_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/vitl_fp32_oom_watchdog_%j.err

set -euo pipefail

# CPU-only Slurm watchdog (no --gres → auto CPU node; no partition/QOS).
# Survives SSH logout. Cgroup mem OOM via perf_*_cgroup_memory.csv + oom_kill lines.

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
export PROJECT_ROOT

: "${JOB_1S:?JOB_1S must be set via sbatch --export}"
: "${JOB_AR10S:?JOB_AR10S must be set via sbatch --export}"

echo "[watchdog-slurm] $(date -Is) watchdog job ${SLURM_JOB_ID:-local} watching 1s=${JOB_1S} ar10s=${JOB_AR10S}"

bash "${PROJECT_ROOT}/scripts/supervise_vitl_fp32_oom_watchdog.sh"
