#!/usr/bin/env bash
# 2s KV-self + RoPE all, three prune rules. Two GPUs each, cold start vitl.pt.
#
#   last : each tick drops the 34 frames with the lowest probe-blk0 score
#          from that pass (the current last-prediction rule)
#   rank : same drop order, but skip frames that were Top-4 in any of the
#          previous 3 score passes
#   abs  : no rank. Over this pass and the two before it, each frame keeps
#          its max |score|; drop the 34 smallest and keep 94
#
# stream_steps=4 (T=128+4*34=264) so the last prune has three earlier passes
# on record. Probe still sees a packed 128. New 34 frames self-attend only.
#
# If the four GPUs are busy, this waits. Then abs uses GPU 0,1 and rank uses
# GPU 2,3. last starts on whichever pair finishes first.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export VJEPA_ROOT="${VJEPA_ROOT:-$ROOT/vjepa2}"
export PYTHONPATH="$ROOT:$VJEPA_ROOT"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PY="${PY:-/mnt/hdd/datasets/HD-EPIC/conda_envs/vjepa/bin/python}"
BASE="${BASE:-/mnt/hdd/datasets/HD-EPIC/experiments/kvprune_joint_2s_kvself_rope_all}"
LOG="${LOG:-/mnt/hdd/datasets/HD-EPIC/experiments/logs}"
CKPT="${CKPT:-/mnt/hdd/jepa/models/vjepa2-vitl/vitl.pt}"
mkdir -p "$BASE" "$LOG"

gpu_mem() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1"
}

gpus_idle() {
  local i mem
  for i in 0 1 2 3; do
    mem="$(gpu_mem "$i")"
    if [ "$mem" -gt 1500 ]; then
      return 1
    fi
  done
  return 0
}

pair_idle() {
  local a b
  a="$(gpu_mem "$1")"
  b="$(gpu_mem "$2")"
  [ "$a" -lt 1500 ] && [ "$b" -lt 1500 ]
}

launch() {
  local name="$1" gpus="$2" port="$3" mode="$4"
  local out="$BASE/prune_${name}"
  mkdir -p "$out"
  local -a extra=()
  case "$mode" in
    last) extra+=(--prune-mode last) ;;
    rank) extra+=(--prune-mode rank --protect-hist 3 --protect-k 4) ;;
    abs) extra+=(--prune-mode abs --score-hist 3) ;;
    *) echo "unknown mode $mode" >&2; return 1 ;;
  esac
  echo "[$(date -Is)] launch kvself rope_all prune=${name} GPUs=${gpus} out=${out}"
  CUDA_VISIBLE_DEVICES="$gpus" MASTER_PORT="$port" "$PY" -m torch.distributed.run \
    --standalone --nproc_per_node=2 \
    scripts/run_local_kvprune_joint_2s.py \
    --horizon 2 --rope 1 --only-block0 0 --rope-time abs \
    --new-self-only 1 --stream-steps 4 \
    --epochs 8 --patience 0 --resume 0 --batch-size 1 --num-workers 2 \
    --checkpoint "$CKPT" --out-dir "$out" \
    "${extra[@]}" \
    >>"$LOG/kvprune_joint_2s_kvself_rope_all_${name}.log" 2>&1 &
  echo $!
}

echo "[$(date -Is)] waiting for 4 idle GPUs before 2s kvself rope_all prunes"
while ! gpus_idle; do
  sleep 60
done

abs_pid="$(launch abs 0,1 29811 abs)"
rank_pid="$(launch rank 2,3 29812 rank)"
echo "[$(date -Is)] abs pid=${abs_pid} rank pid=${rank_pid}"

while kill -0 "$abs_pid" 2>/dev/null && kill -0 "$rank_pid" 2>/dev/null; do
  sleep 30
done

if pair_idle 0 1; then
  last_pid="$(launch last 0,1 29813 last)"
elif pair_idle 2 3; then
  last_pid="$(launch last 2,3 29813 last)"
else
  echo "[$(date -Is)] no free GPU pair for last; waiting"
  while ! pair_idle 0 1 && ! pair_idle 2 3; do
    sleep 30
  done
  if pair_idle 0 1; then
    last_pid="$(launch last 0,1 29813 last)"
  else
    last_pid="$(launch last 2,3 29813 last)"
  fi
fi
echo "[$(date -Is)] last pid=${last_pid}"

wait "$abs_pid" || echo "[$(date -Is)] abs exited $?"
wait "$rank_pid" || echo "[$(date -Is)] rank exited $?"
wait "$last_pid" || echo "[$(date -Is)] last exited $?"
echo "[$(date -Is)] 2s kvself rope_all prune jobs finished"
