#!/bin/bash
# Submit a guarded 2-epoch chain after epochs 0 and 1-2 have passed.
set -euo pipefail
CODE_ROOT=${CODE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}
PROJECT_ROOT=${PROJECT_ROOT:-${CODE_ROOT}}
export PROJECT_ROOT CODE_ROOT
source "${CODE_ROOT}/scripts/egtea/scalant_runtime.sh"

SPLIT=${1:?expected official split 2 or 3}
HORIZON=${2:?expected horizon 2 or 6}
case "$SPLIT" in 2|3) ;; *) printf 'Invalid split: %s\n' "$SPLIT" >&2; exit 2;; esac
case "$HORIZON" in 2|6) ;; *) printf 'Invalid horizon: %s\n' "$HORIZON" >&2; exit 2;; esac

LAUNCHER=${CODE_ROOT}/scripts/egtea/scalant_official_train_gpu.slurm
RUN_DIR=${PROJECT_ROOT}/outputs/b19_scalant/train/official_s${SPLIT}_h${HORIZON}
MANIFEST=${PROJECT_ROOT}/outputs/b19_scalant/train/official_s${SPLIT}_h${HORIZON}_continuation_manifest.tsv
if [ -e "$MANIFEST" ]; then
  printf 'Refusing to reuse continuation manifest: %s\n' "$MANIFEST" >&2
  exit 2
fi
if [ ! -s "${RUN_DIR}/epoch_02.json" ] || [ ! -s "${RUN_DIR}/latest.pt" ]; then
  printf 'Missing completed epoch-2 record or checkpoint: %s\n' "$RUN_DIR" >&2
  exit 2
fi
if [ -e "${RUN_DIR}/epoch_03.json" ]; then
  printf 'Epoch 3 already exists; refusing duplicate submission: %s\n' "$RUN_DIR" >&2
  exit 2
fi

printf 'split\thorizon\tstart_epoch\tepoch_count\tjob_id\tdependency_job_id\trequested_time\n' > "$MANIFEST"
previous_job=
for ((start=3; start<50; start+=2)); do
  count=$((50 - start))
  if [ "$count" -gt 2 ]; then count=2; fi
  wall=01:15:00
  if [ "$count" -eq 1 ]; then wall=01:00:00; fi
  if [ -n "$previous_job" ]; then
    response=$(sbatch --parsable --gres=gpu:1 --time="$wall" \
      --dependency="afterok:${previous_job}" \
      "$LAUNCHER" "$HORIZON" "$start" "$count" "$SPLIT")
  else
    # The epoch-2 job has already ended; the trainer asserts cursor 3.
    response=$(sbatch --parsable --gres=gpu:1 --time="$wall" \
      "$LAUNCHER" "$HORIZON" "$start" "$count" "$SPLIT")
  fi
  job_id=${response%%;*}
  case "$job_id" in *[!0-9]*|'') printf 'Cannot parse sbatch response: %s\n' "$response" >&2; exit 2;; esac
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$SPLIT" "$HORIZON" "$start" "$count" \
    "$job_id" "${previous_job:--}" "$wall" >> "$MANIFEST"
  previous_job=$job_id
done
printf 'Submitted %s guarded jobs: %s\n' "$(($(wc -l < "$MANIFEST") - 1))" "$MANIFEST"
