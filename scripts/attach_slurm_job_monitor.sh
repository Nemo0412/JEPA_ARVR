#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <job_id> [output_dir] [interval_seconds] [duration_seconds]" >&2
    exit 2
fi

JOB_ID="$1"
OUT_DIR="${2:-logs}"
INTERVAL="${3:-30}"
DURATION="${4:-0}"
mkdir -p "${OUT_DIR}"

PREFIX="${OUT_DIR}/attached_${JOB_ID}"

echo "[attach-monitor] job=${JOB_ID}"
echo "[attach-monitor] output prefix=${PREFIX}"
echo "[attach-monitor] interval=${INTERVAL}s"
if [[ "${DURATION}" != "0" ]]; then
    echo "[attach-monitor] duration=${DURATION}s"
else
    echo "[attach-monitor] duration=until stopped"
fi

if ! squeue -h -j "${JOB_ID}" >/dev/null 2>&1; then
    echo "[attach-monitor] Slurm does not report job ${JOB_ID} in the active queue." >&2
    echo "[attach-monitor] Check whether the job id is correct, still running, or on this cluster/account:" >&2
    echo "  squeue -u \"\$USER\"" >&2
    echo "  sacct -j ${JOB_ID} --format=JobID,JobName,State,ExitCode,Elapsed,NodeList%20" >&2
    echo "  scontrol show job -dd ${JOB_ID}" >&2
    exit 1
fi

cat > "${PREFIX}_squeue.txt" <<EOF
== squeue ==
$(squeue -j "${JOB_ID}" -o "%.18i %.9P %.24j %.8u %.2t %.10M %.10l %.6D %.20R %.20b %.10m")

== scontrol show job -dd ==
$(scontrol show job -dd "${JOB_ID}")
EOF

if command -v sstat >/dev/null 2>&1; then
    sstat -j "${JOB_ID}.batch" \
        --format=JobID,AveCPU,AveRSS,MaxRSS,MaxVMSize,MaxDiskRead,MaxDiskWrite \
        > "${PREFIX}_sstat_batch.txt" 2> "${PREFIX}_sstat_batch.err" || true
fi

payload=$(cat <<'BASH'
set -euo pipefail
prefix="$1"
interval="$2"
duration="$3"

echo "[attach-monitor] node=$(hostname)"
echo "[attach-monitor] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
        --format=csv -l "${interval}" > "${prefix}_nvidia_smi.csv" 2> "${prefix}_nvidia_smi.err" &
    nvidia_pid="$!"
else
    echo "nvidia-smi not found" > "${prefix}_nvidia_smi.err"
    nvidia_pid=""
fi

if command -v vmstat >/dev/null 2>&1; then
    vmstat -t "${interval}" > "${prefix}_vmstat.log" 2> "${prefix}_vmstat.err" &
    vmstat_pid="$!"
else
    echo "vmstat not found" > "${prefix}_vmstat.err"
    vmstat_pid=""
fi

while true; do
    {
        date '+== %Y-%m-%dT%H:%M:%S%z =='
        ps -u "${USER}" -o pid,ppid,stat,pcpu,pmem,rss,vsz,etime,cmd --sort=-rss | head -40
        echo
    } >> "${prefix}_ps_top_rss.log" 2> "${prefix}_ps_top_rss.err"
    sleep "${interval}"
done &
ps_pid="$!"

cleanup() {
    for pid in "${nvidia_pid:-}" "${vmstat_pid:-}" "${ps_pid:-}"; do
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done
}
trap cleanup EXIT
if [[ "${duration}" != "0" ]]; then
    sleep "${duration}"
else
    wait
fi
BASH
)

echo "[attach-monitor] starting overlap monitor step; stop with scancel ${JOB_ID}.<stepid> or Ctrl-C if interactive"
srun --jobid="${JOB_ID}" --overlap --nodes=1 --ntasks=1 \
    bash -lc "${payload}" bash "${PREFIX}" "${INTERVAL}" "${DURATION}"
