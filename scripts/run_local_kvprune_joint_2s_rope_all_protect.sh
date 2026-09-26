#!/usr/bin/env bash
# rope_all + Top-K history protect prune, 4-GPU DDP.
#   stream_steps=3  → T=128+3×34=230 (probe always sees packed 128)
#   protect_hist=2  → on prune, skip slots that were Top-K in either of the
#                     previous 2 score passes; walk scores low→high until 34
#                     frames dropped. K=34 frames (17 slots @ tubelet=2).
# Cold start: vitl.pt. Compare to .../joint_2s_rope_all/ (best 35.59%).
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

echo "[$(date -Is)] launching 2s rope_all + protect (K=34, hist=2, steps=3) on 4 GPUs from $CKPT"

CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29721 "$PY" -m torch.distributed.run \
  --standalone --nproc_per_node=4 \
  scripts/run_local_kvprune_joint_2s.py \
  --horizon 2 --rope 1 --only-block0 0 \
  --stream-steps 3 --protect-hist 2 --protect-k 34 \
  --checkpoint "$CKPT" --out-dir "$OUT" \
  --epochs 8 --batch-size 1 --num-workers 2 \
  >>"$LOG/kvprune_joint_2s_rope_all_protect.log" 2>&1 &
echo "protect pid=$!"

echo "[$(date -Is)] log: $LOG/kvprune_joint_2s_rope_all_protect.log"
echo "out:  $OUT/joint_2s_rope_all_protectk34_h2_s3/"
