#!/usr/bin/env bash
# 2s whole-video KV-self, RoPE all, absolute time, ranking protection.
#
# Same stream as the 2s rope_all script. Drop order is the current probe-blk0
# score, but slots that were Top-4 frames in any of the previous 3 passes are
# kept. The first prune has an empty history. The 4th prune is the first one
# that sees all 3 prior Top-K sets. Horizon is 2s and is not added into RoPE.
# One GPU.
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
OUT="${OUT:-/mnt/hdd/datasets/HD-EPIC/experiments/kvprune_joint_2s_stream/ranking}"
LOG="${LOG:-/mnt/hdd/datasets/HD-EPIC/experiments/logs/kvprune_joint_2s_stream_ranking.log}"
CKPT="${CKPT:-/mnt/hdd/jepa/models/vjepa2-vitl/vitl.pt}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
MASTER_PORT="${MASTER_PORT:-29972}"
mkdir -p "$OUT" "$(dirname "$LOG")"

echo "[$(date -Is)] stream 2s ranking gpu=$CUDA_VISIBLE_DEVICES out=$OUT"
exec "$PY" -m torch.distributed.run --standalone --nproc_per_node=1 \
  --master_port="$MASTER_PORT" \
  scripts/run_local_kvprune_6s_stream.py \
  --horizon 2 --rope 1 --only-block0 0 --rope-time abs \
  --prune-mode rank --protect-hist 3 --protect-k 4 \
  --participant P01 \
  --checkpoint "$CKPT" --out-dir "$OUT" \
  --epochs 8 --patience 0 --resume 1 \
  >>"$LOG" 2>&1
