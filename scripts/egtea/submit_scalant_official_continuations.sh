#!/bin/bash
# Submit the remaining official split-1 ScalAnt epochs as guarded pairs.
set -euo pipefail
CODE_ROOT=${CODE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}
PROJECT_ROOT=${PROJECT_ROOT:-${CODE_ROOT}}
export PROJECT_ROOT CODE_ROOT
source "${CODE_ROOT}/scripts/egtea/scalant_runtime.sh"

LAUNCHER=${CODE_ROOT}/scripts/egtea/scalant_official_train_gpu.slurm
MANIFEST=${PROJECT_ROOT}/outputs/b19_scalant/train/official_s1_continuation_manifest.tsv
if [ -e "$MANIFEST" ]; then
  printf 'Refusing to reuse existing continuation manifest: %s\n' "$MANIFEST" >&2
  exit 2
fi
mkdir -p "$(dirname "$MANIFEST")"
printf 'horizon\tstart_epoch\tepoch_count\tjob_id\tdependency_job_id\n' > "$MANIFEST"

for horizon in 2 6; do
  previous_job=
  for ((start_epoch=2; start_epoch<50; start_epoch+=2)); do
    if [ -n "$previous_job" ]; then
      response=$(sbatch --dependency="afterok:${previous_job}" \
        "$LAUNCHER" "$horizon" "$start_epoch" 2)
    else
      # Epoch 1 is already terminal and independently verified; some Slurm
      # installations reject a new afterok dependency on a terminal job.
      response=$(sbatch "$LAUNCHER" "$horizon" "$start_epoch" 2)
    fi
    job_id=${response##* }
    case "$job_id" in *[!0-9]*|'') printf 'Cannot parse sbatch response: %s\n' "$response" >&2; exit 2;; esac
    printf '%s\t%s\t2\t%s\t%s\n' "$horizon" "$start_epoch" "$job_id" "${previous_job:--}" >> "$MANIFEST"
    previous_job=$job_id
  done
done
printf 'Submitted %s guarded continuation jobs; manifest: %s\n' \
  "$(($(wc -l < "$MANIFEST") - 1))" "$MANIFEST"
