#!/usr/bin/env bash
set -euo pipefail

# Watch ViT-L fp32 bs8 fulltrain (w4 primary). On **cgroup host-RAM OOM** only, submit w2 fallback once.
#
# Detection (priority):
#   1. Live: logs/perf_<jobid>_cgroup_memory.csv peak/limit >= threshold while RUNNING
#   2. Terminal: Slurm "oom_kill event", sacct OUT_OF_MEMORY, classify_dataloader_failure cgroup-oom
#   CUDA OOM is logged but does NOT trigger w2 fallback (needs bs/GPU tuning, not fewer workers).
#
# Usage:
#   bash scripts/watch_vitl_fp32_bs8_oom_fallback.sh --track 1s --job-id 10806392
#
# Prefer Slurm-hosted watchdog (SSH-safe):
#   bash scripts/submit_vitl_fp32_oom_watchdog.sh --job-1s ... --job-ar10s ...
#
# Logs: logs/watchdog_vitl_fp32_<track>_<first_jobid>.log

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
track=""
job_id=""
poll_sec=120
max_oom_relaunch=1
cgroup_pressure_pct="${CGROUP_PRESSURE_PCT:-92}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --track) track="$2"; shift 2 ;;
    --job-id) job_id="$2"; shift 2 ;;
    --poll-sec) poll_sec="$2"; shift 2 ;;
    --max-oom-relaunch) max_oom_relaunch="$2"; shift 2 ;;
    --cgroup-pressure-pct) cgroup_pressure_pct="$2"; shift 2 ;;
    *)
      echo "[vitl-watch] unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${track}" || -z "${job_id}" ]]; then
  echo "Usage: $0 --track {1s|ar10s} --job-id <slurm_job_id> [options]" >&2
  exit 1
fi

case "${track}" in
  1s|ar10s) ;;
  *)
    echo "[vitl-watch] --track must be 1s or ar10s" >&2
    exit 1
    ;;
esac

first_job_id="${job_id}"
log_file="${PROJECT_ROOT}/logs/watchdog_vitl_fp32_${track}_${first_job_id}.log"
state_file="${PROJECT_ROOT}/logs/watchdog_vitl_fp32_${track}_${first_job_id}.state"

exec >>"${log_file}" 2>&1
echo "[vitl-watch] $(date -Is) start track=${track} first_job=${first_job_id} poll=${poll_sec}s cgroup_pressure_pct=${cgroup_pressure_pct}"

oom_relaunch_count=0
if [[ -f "${state_file}" ]]; then
  # shellcheck disable=SC1090
  source "${state_file}"
fi

log_paths_for_job() {
  local jid="$1"
  echo "${PROJECT_ROOT}/logs/hdepic_singleprobe_enclora_pose_${jid}.out"
  echo "${PROJECT_ROOT}/logs/hdepic_singleprobe_enclora_pose_${jid}.err"
}

cgroup_csv_for_job() {
  local jid="$1"
  echo "${PROJECT_ROOT}/logs/perf_${jid}_cgroup_memory.csv"
}

log_live_cgroup_stats() {
  local jid="$1"
  local csv
  csv="$(cgroup_csv_for_job "${jid}")"
  if [[ ! -f "${csv}" ]]; then
    return 0
  fi
  tail -n 1 "${csv}" | awk -F, -v thresh="${cgroup_pressure_pct}" '
    NR == 1 {
      peak = $4 + 0
      limit = $6 + 0
      if (limit > 0 && peak > 0) {
        pct = 100.0 * peak / limit
        printf "[vitl-watch] cgroup job=%s peak=%.0f MiB limit=%.0f MiB (%.1f%%)\n", "'"${jid}"'", peak, limit, pct
        if (pct >= thresh) print "CGROUP_PRESSURE_HIGH"
      }
    }
  '
}

is_cuda_oom_in_logs() {
  local jid="$1"
  local f
  for f in $(log_paths_for_job "${jid}"); do
    if [[ -f "${f}" ]] && grep -qiE 'cuda out of memory|torch\.OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED' "${f}"; then
      return 0
    fi
  done
  return 1
}

is_cgroup_oom_failure() {
  local jid="$1"
  local state="$2"
  local reason="$3"
  local state_u reason_u
  state_u="$(echo "${state}" | tr '[:lower:]' '[:upper:]')"
  reason_u="$(echo "${reason}" | tr '[:lower:]' '[:upper:]')"

  if [[ "${state_u}" == *OUT_OF_MEM* ]]; then
    return 0
  fi
  if [[ "${reason_u}" == *OOM_KILL* || "${reason_u}" == *OUT_OF_MEM* ]]; then
    return 0
  fi

  local f
  for f in $(log_paths_for_job "${jid}"); do
    if [[ ! -f "${f}" ]]; then
      continue
    fi
    if grep -qiE 'oom_kill event|oom-kill|Detected [0-9]+ oom_kill' "${f}"; then
      return 0
    fi
    if grep -qiE 'DataLoader worker \(pid [0-9]+\) is killed by signal: Killed' "${f}"; then
      return 0
    fi
  done

  if command -v python3 >/dev/null 2>&1; then
    local classify_json
    classify_json="$(python3 "${PROJECT_ROOT}/scripts/classify_dataloader_failure.py" --json \
      "${PROJECT_ROOT}/logs/hdepic_singleprobe_enclora_pose_${jid}.out" \
      "${PROJECT_ROOT}/logs/hdepic_singleprobe_enclora_pose_${jid}.err" 2>/dev/null || true)"
    if [[ "${classify_json}" == *'"label": "cgroup-oom"'* ]]; then
      return 0
    fi
  fi

  # Terminal cgroup csv: peak essentially at Slurm mem cap right before crash.
  local csv peak limit
  csv="$(cgroup_csv_for_job "${jid}")"
  if [[ -f "${csv}" ]]; then
    read -r _peak _limit < <(tail -n 1 "${csv}" | awk -F, '{print $4, $6}')
    peak="${_peak:-0}"
    limit="${_limit:-0}"
    if awk -v p="${peak}" -v l="${limit}" -v t="${cgroup_pressure_pct}" 'BEGIN { exit !(l>0 && p>0 && (100*p/l)>=t) }'; then
      if is_cuda_oom_in_logs "${jid}"; then
        : # cuda-only; do not treat as cgroup
      else
        return 0
      fi
    fi
  fi

  return 1
}

submit_fallback_w2() {
  local relaunch_n="$1"
  local tag_suffix="w2-oomfb-r${relaunch_n}"
  echo "[vitl-watch] cgroup-OOM fallback: ${track} workers=2 tag_suffix=${tag_suffix}"

  if [[ "${track}" == "1s" ]]; then
    LORA_TAG="hdepic-singleprobe-vitl-fp32-bs8-noac-10ep-${tag_suffix}" \
      EVAL_NUM_WORKERS=2 EVAL_VAL_NUM_WORKERS=2 \
      bash "${PROJECT_ROOT}/scripts/submit_b11_singleprobe_vitl_fp32_bs8_noac_fulltrain.sh"
  else
    LORA_TAG="hdepic-singleprobe-ar10s-vitl-fp32-bs8-noac-10ep-${tag_suffix}" \
      EVAL_NUM_WORKERS=2 EVAL_VAL_NUM_WORKERS=2 \
      bash "${PROJECT_ROOT}/scripts/submit_b11_singleprobe_ar10s_vitl_fp32_bs8_noac_fulltrain.sh"
  fi
}

while true; do
  state_live="$(squeue -h -j "${job_id}" -o "%T" 2>/dev/null || true)"
  if [[ -n "${state_live}" ]]; then
    echo "[vitl-watch] $(date -Is) track=${track} job ${job_id} live state=${state_live}"
    log_live_cgroup_stats "${job_id}" || true
    sleep "${poll_sec}"
    continue
  fi

  sacct_line="$(sacct -n -X -j "${job_id}" --format=JobIDRaw,State,ExitCode,Reason -P 2>/dev/null | awk -F'|' -v id="${job_id}" '$1==id {print; exit}')"
  if [[ -z "${sacct_line}" ]]; then
    echo "[vitl-watch] job ${job_id} not in sacct yet"
    sleep "${poll_sec}"
    continue
  fi

  state="$(echo "${sacct_line}" | awk -F'|' '{print $2}')"
  exit_code="$(echo "${sacct_line}" | awk -F'|' '{print $3}')"
  reason="$(echo "${sacct_line}" | awk -F'|' '{print $4}')"
  echo "[vitl-watch] $(date -Is) track=${track} terminal job ${job_id}: state=${state} exit=${exit_code} reason=${reason}"

  if [[ "${state}" == COMPLETED* ]]; then
    echo "[vitl-watch] track=${track} success, stop"
    exit 0
  fi

  if is_cgroup_oom_failure "${job_id}" "${state}" "${reason}"; then
    if (( oom_relaunch_count >= max_oom_relaunch )); then
      echo "[vitl-watch] track=${track} cgroup-OOM but max_oom_relaunch=${max_oom_relaunch} reached, stop"
      exit 2
    fi
    oom_relaunch_count=$((oom_relaunch_count + 1))
    {
      echo "oom_relaunch_count=${oom_relaunch_count}"
      echo "last_job_id=${job_id}"
      echo "last_state=${state}"
      echo "last_reason=${reason}"
      echo "oom_kind=cgroup"
    } >"${state_file}"

    submit_out="$(submit_fallback_w2 "${oom_relaunch_count}")"
    echo "[vitl-watch] submit: ${submit_out}"
    new_id="$(echo "${submit_out}" | awk '/Submitted batch job/ {print $4}' | tail -n 1)"
    if [[ -z "${new_id}" ]]; then
      echo "[vitl-watch] failed to parse fallback job id"
      exit 3
    fi
    job_id="${new_id}"
    echo "fallback_job_id=${new_id}" >>"${state_file}"
    sleep "${poll_sec}"
    continue
  fi

  if is_cuda_oom_in_logs "${job_id}"; then
    echo "[vitl-watch] track=${track} CUDA OOM detected — not auto-fallback (reduce bs / enable AC, not workers)"
  fi

  echo "[vitl-watch] track=${track} terminal non-cgroup-OOM failure, no fallback"
  exit 4
done
