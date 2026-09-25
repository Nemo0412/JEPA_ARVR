#!/bin/bash
#SBATCH --job-name=probe_pe
#SBATCH --account=torch_pr_111_general
#SBATCH --partition=l40s_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/ll5914/logs/probe_posenc_kvcache_%j.out
#SBATCH --error=/scratch/ll5914/logs/probe_posenc_kvcache_%j.err

set -euo pipefail
export TMPDIR=/scratch/ll5914/tmp
mkdir -p "$TMPDIR" /scratch/ll5914/logs

PYTHON="/scratch/ll5914/conda_envs/SVD/bin/python"
PROJECT_ROOT="/home/ll5914/Jepa_yifan/JEPA_ARVR"
VJEPA_ROOT="/home/ll5914/ARVR_Video/vjepa2"
export VJEPA_ROOT PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/scripts:${VJEPA_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

OUT="/scratch/ll5914/experiments/probe_posenc_kvcache_128_34"
mkdir -p "$OUT"

echo "Probe PE  KV-cache 128 + new 34  rel_rank vs abs_stream vs noPE  job=${SLURM_JOB_ID:-manual}  $(date)"
nvidia-smi -L || true

cd "$PROJECT_ROOT"
"$PYTHON" -m py_compile \
  app/hdepic_lora_action_anticipation/stream_kvcache_attn_prune.py \
  scripts/eval_probe_posenc_kvcache_128_34.py

"$PYTHON" scripts/eval_probe_posenc_kvcache_128_34.py \
  --out-dir "$OUT" \
  --copy-json /home/ll5914/Jepa_yifan/probe_posenc_kvcache_128_34.json \
  --copy-png /home/ll5914/Jepa_yifan/probe_posenc_kvcache_128_34.png \
  --max-samples 108 \
  --clips-per-video 4 \
  --log-every 5

echo "Done $(date)"
python3 - <<'PY'
import json
p="/scratch/ll5914/experiments/probe_posenc_kvcache_128_34/metrics.json"
d=json.load(open(p))
print("method", d.get("method"))
if "accuracy" in d:
    print("accuracy", json.dumps(d["accuracy"]["table_action_top5"], indent=2))
    print("delta_pp", json.dumps(d["accuracy"]["delta_pp"], indent=2))
PY
ls -lh "$OUT"/probe_posenc_kvcache_128_34.png /home/ll5914/Jepa_yifan/probe_posenc_kvcache_128_34.png
