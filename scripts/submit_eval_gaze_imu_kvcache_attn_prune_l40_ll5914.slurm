#!/bin/bash
#SBATCH --job-name=gi_kvprune
#SBATCH --account=torch_pr_111_general
#SBATCH --partition=l40s_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/ll5914/logs/gaze_imu_kvprune_%j.out
#SBATCH --error=/scratch/ll5914/logs/gaze_imu_kvprune_%j.err

set -euo pipefail
export TMPDIR=/scratch/ll5914/tmp
mkdir -p "$TMPDIR" /scratch/ll5914/logs

PYTHON="/scratch/ll5914/conda_envs/SVD/bin/python"
PROJECT_ROOT="/home/ll5914/Jepa_yifan/JEPA_ARVR"
VJEPA_ROOT="/home/ll5914/ARVR_Video/vjepa2"
export VJEPA_ROOT PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/scripts:${VJEPA_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRI_MODAL_GAZE_MAP_CACHE=256
export GAZE_DECODER_CACHE_READER=1
export GAZE_DECODER_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

OUT="/scratch/ll5914/experiments/gaze_imu_kvcache_attn_prune"
mkdir -p "$OUT"

echo "Gaze+IMU stream KV-attn-prune  cache=128f new=34f  job=${SLURM_JOB_ID:-manual}  $(date)"
nvidia-smi -L || true

cd "$PROJECT_ROOT"
"$PYTHON" -m py_compile \
  app/hdepic_lora_action_anticipation/stream_kvcache_attn_prune.py \
  scripts/eval_gaze_imu_kvcache_attn_prune.py

"$PYTHON" scripts/eval_gaze_imu_kvcache_attn_prune.py \
  --out-dir "$OUT" \
  --copy-json /home/ll5914/Jepa_yifan/gaze_imu_kvcache_attn_prune.json \
  --copy-png /home/ll5914/Jepa_yifan/gaze_imu_kvcache_attn_prune.png \
  --stream-steps 1 \
  --max-samples 108 \
  --clips-per-video 4 \
  --warmup 3 \
  --repeats 10 \
  --log-every 5

echo "Done $(date)"
python3 - <<'PY'
import json
p="/scratch/ll5914/experiments/gaze_imu_kvcache_attn_prune/metrics.json"
d=json.load(open(p))
print("method", d.get("method"))
if "latency" in d:
    print("latency deltas", json.dumps(d["latency"]["delta_median_ms"], indent=2))
    for k,v in d["latency"]["results"].items():
        print(f"  {k}: e2e {v['e2e_ms']['median_ms']:.2f}")
if "accuracy" in d:
    print("accuracy", json.dumps(d["accuracy"]["table_action_top5"], indent=2))
    print("delta_pp", json.dumps(d["accuracy"]["delta_pp"], indent=2))
PY
