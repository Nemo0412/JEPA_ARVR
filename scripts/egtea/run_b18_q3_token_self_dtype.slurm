#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__b18_q3_self_dtype
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:02:00
set -euo pipefail
SHARE_GPU="${SHARE_GPU:-1}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"
TS_DTYPE_LOG=${DATA_ROOT}/outputs/attn_corner_sink/q3_token_self_return/dtype_${SLURM_JOB_ID}.gpu.csv
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits --loop=5 > "$TS_DTYPE_LOG" &
TS_DTYPE_PID=$!
trap 'kill "$TS_DTYPE_PID" 2>/dev/null || true; wait "$TS_DTYPE_PID" 2>/dev/null || true' EXIT
share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
python - <<'PY'
import torch, time, json
torch.set_num_threads(1)
q = torch.randn(1,12,256,32, device='cuda', dtype=torch.bfloat16)
k = torch.randn(1,12,16640,32, device='cuda', dtype=torch.bfloat16)
begin=time.time(); n=0
with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
    while time.time()-begin<25:
        logits=(q@k.transpose(-2,-1))*(32**-.5)
        positional=logits.softmax(-1)
        keyword=logits.softmax(dim=-1)
        if n==0:
            print(json.dumps(dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,q=str(q.dtype),logits=str(logits.dtype),positional_softmax=str(positional.dtype),legacy_keyword_softmax=str(keyword.dtype),equal=torch.equal(positional,keyword),max_abs=float((positional-keyword).abs().max()))),flush=True)
        torch.cuda.synchronize();n+=1
print(json.dumps(dict(start=begin,end=time.time(),iterations=n)),flush=True)
PY
EOF
