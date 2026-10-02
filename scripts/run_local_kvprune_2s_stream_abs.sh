#!/usr/bin/env bash
# 2s whole-video KV-self, RoPE all, absolute time, abs-score protection.
#
# Same stream as the 2s rope_all script. Each slot keeps the max |probe-blk0
# score| over this pass and up to 2 previous passes (score-hist 3); the 34
# smallest of those are dropped. The first prune has no history, so it matches
# last. Horizon is 2s and is not added into RoPE. One GPU.
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
OUT="${OUT:-/mnt/hdd/datasets/HD-EPIC/experiments/kvprune_joint_2s_stream/abs}"
LOG="${LOG:-/mnt/hdd/datasets/HD-EPIC/experiments/logs/kvprune_joint_2s_stream_abs.log}"
CKPT="${CKPT:-/mnt/hdd/jepa/models/vjepa2-vitl/vitl.pt}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
MASTER_PORT="${MASTER_PORT:-29973}"
mkdir -p "$OUT" "$(dirname "$LOG")"

echo "[$(date -Is)] stream 2s abs gpu=$CUDA_VISIBLE_DEVICES out=$OUT"
exec "$PY" -m torch.distributed.run --standalone --nproc_per_node=1 \
  --master_port="$MASTER_PORT" \
  scripts/run_local_kvprune_6s_stream.py \
  --horizon 2 --rope 1 --only-block0 0 --rope-time abs \
  --prune-mode abs --score-hist 3 \
  --participant P01 \
  --checkpoint "$CKPT" --out-dir "$OUT" \
  --epochs 8 --patience 0 --resume 1 \
  >>"$LOG" 2>&1
