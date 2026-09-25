#!/bin/bash
#SBATCH --job-name=kvprune_rope
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --partition=a100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:50:00
#SBATCH --output=/scratch/ll5914/logs/kvprune_rope_%j.out
#SBATCH --error=/scratch/ll5914/logs/kvprune_rope_%j.err

set -euo pipefail
export TMPDIR=/scratch/ll5914/tmp
mkdir -p "$TMPDIR" /scratch/ll5914/logs

# Stream KV + Probe-blk0-attn prune; probe FT ± abs-frame RoPE (blk0 only).
# Prune scores = Probe.blocks[0] self-attn received mass (recorded for next tick).
# Submit as a 3-partition race with low mem (prefer 96G):
#   for spec in 'a100_tandon|gpu:a100:1' 'h100_tandon|gpu:h100:1' 'h200_tandon|gpu:h200:1'; do
#     IFS='|' read -r part gres <<< "$spec"
#     sbatch --job-name=kvprune_rope --partition="$part" --gres="$gres" \
#       --mem=96G --cpus-per-task=8 --time=01:50:00 this.slurm
#   done

PYTHON="/scratch/ll5914/conda_envs/SVD/bin/python"
PROJECT_ROOT="/home/ll5914/Jepa_yifan/JEPA_ARVR"
VJEPA_ROOT="/home/ll5914/ARVR_Video/vjepa2"
export VJEPA_ROOT PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/scripts:${VJEPA_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

OUT="/scratch/ll5914/experiments/stream_kv_probe_blk0_prune_ft_rope"
DONE="${OUT}/DONE"
mkdir -p "$OUT"

if [[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_JOB_NAME:-}" ]]; then
  while read -r sib; do
    [[ -z "${sib}" || "${sib}" == "${SLURM_JOB_ID}" ]] && continue
    echo "Canceling sibling ${sib} (same name ${SLURM_JOB_NAME})"
    scancel --signal=KILL "${sib}" 2>/dev/null || true
  done < <(squeue -u "${USER}" -n "${SLURM_JOB_NAME}" -h -o '%i' 2>/dev/null || true)
fi

if [[ -f "${DONE}" ]]; then
  echo "Already done (${DONE}); exiting."
  exit 0
fi

echo "========================================================"
echo "Stream KV + Probe-blk0 prune  (± abs-frame RoPE FT)"
echo "Job=${SLURM_JOB_ID:-manual} partition=${SLURM_JOB_PARTITION:-} $(date)"
echo "host=$(hostname)"
echo "cache=128f new=34f keep=94; scores=Probe.blocks[0] self-attn mass"
echo "========================================================"
nvidia-smi -L || true

cd "$PROJECT_ROOT"
"$PYTHON" -m py_compile \
  app/hdepic_lora_action_anticipation/stream_kvcache_attn_prune.py \
  scripts/finetune_stream_kv_prune_probe_rope.py

"$PYTHON" scripts/finetune_stream_kv_prune_probe_rope.py \
  --out-dir "$OUT" \
  --copy-json "${PROJECT_ROOT}/configs/jepa_pe/stream_kv_probe_blk0_prune_ft_rope.json" \
  --copy-png "${PROJECT_ROOT}/plots/stream_kv_probe_blk0_prune_ft_rope.png" \
  --max-train 400 \
  --max-val 200 \
  --train-clips-per-video 8 \
  --val-clips-per-video 4 \
  --epochs 8 \
  --lr 1e-4

date > "${DONE}"
echo "Done $(date)"
"$PYTHON" - <<'PY'
import json
p="/scratch/ll5914/experiments/stream_kv_probe_blk0_prune_ft_rope/metrics.json"
d=json.load(open(p))
print("method", d.get("method"))
print("table", json.dumps(d["accuracy"]["table_action_top5"], indent=2))
print("delta_pp", json.dumps(d["accuracy"]["delta_pp"], indent=2))
PY
ls -lh "$OUT"/metrics.json "$OUT"/probe_ft_*.pt "$OUT"/*.png 2>/dev/null || true
