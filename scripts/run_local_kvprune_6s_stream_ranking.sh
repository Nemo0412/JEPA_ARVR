#!/usr/bin/env bash
# 6s whole-video KV-self, RoPE all, absolute time, ranking protection.
#
# Same stream as rope_all (t=0 to the end of each P01 video, probe sees 128,
# new 34 frames self-attend only). Drop order is the current probe-blk0 score,
# but slots that were Top-4 frames in any of the previous 3 passes are kept.
# The first prune has an empty history. The 4th prune is the first one that
# sees all 3 prior Top-K sets; protection stays on until the video ends.
# One GPU. Separate out dir from the clip-style ranking run.
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
OUT="${OUT:-/mnt/hdd/datasets/HD-EPIC/experiments/kvprune_joint_6s_stream/ranking}"
LOG="${LOG:-/mnt/hdd/datasets/HD-EPIC/experiments/logs/kvprune_joint_6s_stream_ranking.log}"
CKPT="${CKPT:-/mnt/hdd/jepa/models/vjepa2-vitl/vitl.pt}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
MASTER_PORT="${MASTER_PORT:-29962}"
mkdir -p "$OUT" "$(dirname "$LOG")"

echo "[$(date -Is)] stream ranking gpu=$CUDA_VISIBLE_DEVICES out=$OUT"
exec "$PY" -m torch.distributed.run --standalone --nproc_per_node=1 \
  --master_port="$MASTER_PORT" \
  scripts/run_local_kvprune_6s_stream.py \
  --horizon 6 --rope 1 --only-block0 0 --rope-time abs \
  --prune-mode rank --protect-hist 3 --protect-k 4 \
  --participant P01 \
  --checkpoint "$CKPT" --out-dir "$OUT" \
  --epochs 8 --patience 0 --resume 1 \
  >>"$LOG" 2>&1
