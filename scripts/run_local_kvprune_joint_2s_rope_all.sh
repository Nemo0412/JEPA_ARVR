#!/usr/bin/env bash
# After the current 2s (norope / only_block0 RoPE) jobs finish, launch:
#   2s + RoPE on ALL Probe self-attn blocks, 4-GPU DDP.
# 6s is deferred — do not start from this script.
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
OUT="${OUT:-/mnt/hdd/datasets/HD-EPIC/experiments/kvprune_joint_2s}"
LOG="${LOG:-/mnt/hdd/datasets/HD-EPIC/experiments/logs}"
CKPT="${CKPT:-/mnt/hdd/jepa/models/vjepa2-vitl/vitl.pt}"
mkdir -p "$OUT" "$LOG"

# True while the *current* 2s ablation workers are still alive (no --only-block0 0).
still_legacy_2s() {
  pgrep -af 'run_local_kvprune_joint_2s\.py' 2>/dev/null \
    | grep -vE -- '--only-block0(=| )0' \
    | grep -v 'run_local_kvprune_joint_2s_rope_all' \
    | grep -v 'run_local_kvprune_joint_6s' \
    | grep -v 'grep' \
    | grep -q .
}

echo "[$(date -Is)] waiting for legacy joint_2s (norope / rope_blk0) to exit..."
while still_legacy_2s; do
  sleep 60
done
sleep 30
echo "[$(date -Is)] launching 2s rope_all on 4 GPUs from $CKPT"

CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29711 "$PY" -m torch.distributed.run \
  --standalone --nproc_per_node=4 \
  scripts/run_local_kvprune_joint_2s.py \
  --horizon 2 --rope 1 --only-block0 0 \
  --checkpoint "$CKPT" --out-dir "$OUT" \
  --epochs 8 --batch-size 1 --num-workers 2 \
  >>"$LOG/kvprune_joint_2s_rope_all.log" 2>&1 &
echo "rope_all pid=$!"

echo "[$(date -Is)] log: $LOG/kvprune_joint_2s_rope_all.log"
echo "out:  $OUT/joint_2s_rope_all/"
echo "docs: docs/KVPRUNE_JOINT.md"
