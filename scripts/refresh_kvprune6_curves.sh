#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/scratch/ll5914/conda_envs/SVD/bin/python}"
EXP=/scratch/ll5914/experiments/kvprune_joint_h6s
"$PY" "$ROOT/scripts/plot_kvprune_joint_curves.py" \
  --runs \
    norope:"$EXP/joint_6s_norope" \
    rope_all:"$EXP/joint_6s_rope_all" \
  --title 'kvprune joint 6s — norope / rope_all' \
  --out "$ROOT/kvprune_joint_6s_loss_curves.png" \
  --sync-dir "$ROOT/plots/kvprune_joint_6s" \
  --x-mode iter_latest \
  --run-global-batch 1 --align-batch 1
