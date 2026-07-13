#!/usr/bin/env bash
set -euo pipefail

# Monitor nvidia-smi perf csv produced by run_hdepic_lora_probe.slurm PERF_MONITOR=1.
# Emits:
# - GPU_USAGE_STATUS ... (periodic summary)
# - LOW_USAGE_ALERT ...  (when sustained low utilization is detected)
#
# Usage:
#   bash scripts/monitor_perf_gpu_usage.sh --job-id 9930741
#
# Optional:
#   --csv PATH
#   --poll-sec 300
#   --window-samples 20
#   --min-avg-util 35
#   --min-max-util 60
#   --low-streak-limit 3
#   --duration-sec 7200

job_id=""
csv_path=""
poll_sec=300
window_samples=20
min_avg_util=35
min_max_util=60
low_streak_limit=3
duration_sec=7200

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-id) job_id="$2"; shift 2 ;;
    --csv) csv_path="$2"; shift 2 ;;
    --poll-sec) poll_sec="$2"; shift 2 ;;
    --window-samples) window_samples="$2"; shift 2 ;;
    --min-avg-util) min_avg_util="$2"; shift 2 ;;
    --min-max-util) min_max_util="$2"; shift 2 ;;
    --low-streak-limit) low_streak_limit="$2"; shift 2 ;;
    --duration-sec) duration_sec="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${csv_path}" ]]; then
  if [[ -z "${job_id}" ]]; then
    echo "Need --job-id or --csv" >&2
    exit 1
  fi
  csv_path="/path/to/VJEPA2-EXP/logs/perf_${job_id}_nvidia_smi.csv"
fi

start_ts="$(date +%s)"
low_streak=0

echo "GPU_USAGE_STATUS start csv=${csv_path} poll=${poll_sec}s window=${window_samples} avg>=${min_avg_util}% max>=${min_max_util}% duration=${duration_sec}s"

while true; do
  now_ts="$(date +%s)"
  elapsed=$((now_ts - start_ts))
  if (( elapsed >= duration_sec )); then
    echo "GPU_USAGE_STATUS done elapsed_sec=${elapsed} reason=duration_reached"
    exit 0
  fi

  if [[ ! -f "${csv_path}" ]]; then
    echo "GPU_USAGE_STATUS waiting_for_file elapsed_sec=${elapsed} csv=${csv_path}"
    sleep "${poll_sec}"
    continue
  fi

  metrics="$(python - "${csv_path}" "${window_samples}" <<'PY'
import csv, re, sys
path = sys.argv[1]
window = max(1, int(sys.argv[2]))
vals = []
with open(path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, skipinitialspace=True)
    for row in reader:
        raw = (row.get("utilization.gpu [%]") or "").strip()
        m = re.search(r"(-?\d+)", raw)
        if m:
            vals.append(int(m.group(1)))
vals = vals[-window:]
if not vals:
    print("count=0 avg=0.0 min=0 max=0 latest=0")
else:
    print(
        f"count={len(vals)} avg={sum(vals)/len(vals):.1f} min={min(vals)} max={max(vals)} latest={vals[-1]}"
    )
PY
)"

  count="$(echo "${metrics}" | sed -n 's/.*count=\([0-9]\+\).*/\1/p')"
  avg="$(echo "${metrics}" | sed -n 's/.*avg=\([0-9.]\+\).*/\1/p')"
  maxv="$(echo "${metrics}" | sed -n 's/.*max=\([0-9]\+\).*/\1/p')"
  latest="$(echo "${metrics}" | sed -n 's/.*latest=\([0-9]\+\).*/\1/p')"

  if [[ -z "${count}" || "${count}" == "0" ]]; then
    echo "GPU_USAGE_STATUS waiting_for_samples elapsed_sec=${elapsed}"
    sleep "${poll_sec}"
    continue
  fi

  avg_int="${avg%.*}"
  if [[ -z "${avg_int}" ]]; then
    avg_int=0
  fi

  if (( avg_int < min_avg_util && maxv < min_max_util )); then
    low_streak=$((low_streak + 1))
    echo "GPU_USAGE_STATUS elapsed_sec=${elapsed} avg=${avg}% max=${maxv}% latest=${latest}% low_streak=${low_streak}"
    if (( low_streak >= low_streak_limit )); then
      echo "LOW_USAGE_ALERT elapsed_sec=${elapsed} avg=${avg}% max=${maxv}% latest=${latest}% low_streak=${low_streak}"
    fi
  else
    low_streak=0
    echo "GPU_USAGE_STATUS elapsed_sec=${elapsed} avg=${avg}% max=${maxv}% latest=${latest}% low_streak=${low_streak}"
  fi

  sleep "${poll_sec}"
done

