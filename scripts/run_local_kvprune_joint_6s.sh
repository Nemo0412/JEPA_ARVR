#!/usr/bin/env bash
# Wait for in-flight kvprune joint 2s jobs, then launch 6s norope + rope_all.
# RoPE covers all Probe self-attn blocks (--only-block0 0).
# Start ckpt: /mnt/hdd/jepa/models/vjepa2-vitl/vitl.pt
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
OUT="${OUT:-/mnt/hdd/datasets/HD-EPIC/experiments/kvprune_joint_6s}"
LOG="${LOG:-/mnt/hdd/datasets/HD-EPIC/experiments/logs}"
CKPT="${CKPT:-/mnt/hdd/jepa/models/vjepa2-vitl/vitl.pt}"
mkdir -p "$OUT" "$LOG"

still_2s() {
  # Legacy 2s launches have no --horizon; ignore any --horizon 6 workers.
  pgrep -af 'run_local_kvprune_joint_2s\.py' 2>/dev/null \
    | grep -vE -- '--horizon(=| )6' \
    | grep -v 'run_local_kvprune_joint_6s' \
    | grep -v 'grep' \
    | grep -q .
}

echo "[$(date -Is)] waiting for joint_2s workers to exit..."
while still_2s; do
  sleep 60
done
sleep 30
echo "[$(date -Is)] launching 6s joint (norope + rope_all) from $CKPT"

CUDA_VISIBLE_DEVICES=0,1 MASTER_PORT=29701 "$PY" -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  scripts/run_local_kvprune_joint_2s.py \
  --horizon 6 --rope 0 --only-block0 0 \
  --checkpoint "$CKPT" --out-dir "$OUT" \
  --epochs 8 --batch-size 1 --num-workers 4 \
  >>"$LOG/kvprune_joint_6s_norope.log" 2>&1 &
echo "norope pid=$!"

sleep 2

CUDA_VISIBLE_DEVICES=2,3 MASTER_PORT=29702 "$PY" -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  scripts/run_local_kvprune_joint_2s.py \
  --horizon 6 --rope 1 --only-block0 0 \
  --checkpoint "$CKPT" --out-dir "$OUT" \
  --epochs 8 --batch-size 1 --num-workers 4 \
  >>"$LOG/kvprune_joint_6s_rope.log" 2>&1 &
echo "rope_all pid=$!"

echo "[$(date -Is)] logs: $LOG/kvprune_joint_6s_{norope,rope}.log"
echo "docs: docs/KVPRUNE_JOINT.md"
