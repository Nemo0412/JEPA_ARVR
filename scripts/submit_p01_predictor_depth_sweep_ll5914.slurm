#!/bin/bash
#SBATCH --job-name=p01_pred_depth
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=384GB
#SBATCH --time=03:00:00
#SBATCH --array=0-5
#SBATCH --output=/scratch/ll5914/logs/p01_pred_depth_%A_%a.out
#SBATCH --error=/scratch/ll5914/logs/p01_pred_depth_%A_%a.err

set -euo pipefail

# HD-EPIC P01 video-only: discover how many predictor blocks to adapt via LoRA.
# Sweep predictor_lora.last_n_blocks in {1,2,4,6,8,12}.
# Encoder LoRA + probe frozen from stage-1 video best; only predictor LoRA trains.
# Metric for best/early-stop: val-action-top5.

DEPTHS=(1 2 4 6 8 12)
DEPTH="${DEPTHS[${SLURM_ARRAY_TASK_ID}]}"

PYTHON="/scratch/ll5914/conda_envs/SVD/bin/python"
PROJECT_ROOT="/home/ll5914/Jepa_yifan/JEPA_ARVR"
VJEPA_ROOT="/home/ll5914/ARVR_Video/vjepa2"

ANN_DIR="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/clip_split"
VIDEO_ROOT="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_videos"
HDEPIC_ANN_ROOT="/scratch/ll5914/datasets/HD-EPIC/hd-epic-annotations/narrations-and-action-segments"
CHECKPOINT="/scratch/ll5914/models/vjepa2/vitl.pt"
STAGE1_DIR="${STAGE1_DIR:-/scratch/ll5914/experiments/p01_video_enc_clip/action_anticipation_frozen/p01-video-enc-clip-vitl16-256-10ep}"
STAGE1_PROBE="${STAGE1_DIR}/best.pt"
STAGE1_ENCODER_LORA="${STAGE1_DIR}/encoder_lora_best.pt"

OUTPUT_ROOT="/scratch/ll5914/experiments/p01_predictor_depth_sweep"
OUTPUT_DIR="${OUTPUT_ROOT}/depth${DEPTH}"
CONFIG_PATH="/scratch/ll5914/logs/p01_predictor_depth_d${DEPTH}_config.yaml"
TAG="p01-pred-depth${DEPTH}-vitl16-256-8ep"

for f in "${STAGE1_PROBE}" "${STAGE1_ENCODER_LORA}"; do
    if [[ ! -f "${f}" ]]; then
        echo "ERROR: stage-1 checkpoint missing: ${f}" >&2
        exit 1
    fi
done

export MASTER_PORT=$((20000 + (${SLURM_JOB_ID:-$$} % 10000) + ${SLURM_ARRAY_TASK_ID:-0} * 17))
echo "========================================================"
echo "P01 predictor-depth sweep: last_n_blocks=${DEPTH}"
echo "Job=${SLURM_JOB_ID:-manual} array=${SLURM_ARRAY_TASK_ID:-0} MASTER_PORT=${MASTER_PORT}"
echo "Stage-1: ${STAGE1_DIR}"
echo "Out: ${OUTPUT_DIR}"
echo "Start: $(date)"
echo "========================================================"

export PYTHONPATH="${PROJECT_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"

"${PYTHON}" - <<PY
import sys, yaml
from pathlib import Path
sys.path.insert(0, '${PROJECT_ROOT}')
sys.path.insert(0, '${VJEPA_ROOT}')
template_path = Path('${VJEPA_ROOT}/configs/eval/vitl/ek100.yaml')
config_path   = Path('${CONFIG_PATH}')
depth = int('${DEPTH}')
with template_path.open() as f:
    cfg = yaml.safe_load(f)
cfg['nodes'] = 1
cfg['tasks_per_node'] = 1
cfg['cpus_per_task'] = 24
cfg['tag']  = '${TAG}'
cfg['eval_name'] = 'app.hdepic_lora_action_anticipation'
cfg['folder'] = '${OUTPUT_DIR}'
cfg['resume_checkpoint'] = True
cfg['val_only'] = False
data = cfg['experiment']['data']
data.update({
    'dataset': 'EK100', 'file_format': 1,
    'base_path': '${VIDEO_ROOT}',
    'dataset_train': '${ANN_DIR}/HD_EPIC_train_vjepa.csv',
    'dataset_val': '${ANN_DIR}/HD_EPIC_val_vjepa.csv',
    'num_workers': 10, 'val_num_workers': 6, 'clip_balanced': True,
    'resolution': 256, 'frames_per_clip': 32, 'frames_per_second': 8,
    'anticipation_time_sec': [1.0, 1.0],
    'train_anticipation_time_sec': [0.25, 1.75],
})
opt = cfg['experiment']['optimization']
# Probe frozen; multihead lr only used if somehow unfrozen. Predictor LoRA lr via lr_mult.
opt.update({'batch_size': 16, 'num_epochs': 8, 'use_bfloat16': True,
    'use_focal_loss': False, 'val_every_epochs': 1, 'best_metric': 'val-action-top5',
    'early_stopping_patience': 3, 'grad_clip': 1.0,
    'multihead_kwargs': [{'weight_decay': 0.0001, 'final_weight_decay': 0.0001,
        'lr': 0.0001, 'start_lr': 0.0, 'final_lr': 0.0, 'warmup': 1}]})
cfg['model_kwargs']['checkpoint'] = '${CHECKPOINT}'
cfg['experiment']['lora'] = {
    'enabled': True, 'rank': 8, 'alpha': 16.0, 'dropout': 0.05,
    'probe_train_mode': 'full',
    'train_heads': False,
    'load_probe_heads': True,
    'pretrained_probe': '${STAGE1_PROBE}',
    'align_reference_metrics': True, 'val_metric_scope': 'native',
    'val_metric_aggregation': 'metric_wise_max', 'class_space': 'train_only',
    'temporal_sampling': 'phd_reference', 'output_regularization': {'enabled': False},
    'hdepic_reference': {
        'annotations_pkl': '${HDEPIC_ANN_ROOT}/HD_EPIC_Narrations.pkl',
        'verb_classes_csv': '${HDEPIC_ANN_ROOT}/HD_EPIC_verb_classes.csv',
        'noun_classes_csv': '${HDEPIC_ANN_ROOT}/HD_EPIC_noun_classes.csv',
    },
    'encoder_lora': {
        'enabled': True, 'rank': 8, 'alpha': 16.0, 'dropout': 0.05,
        'last_n_blocks': 0, 'lr_mult': 0.5, 'weight_decay': 0.0001,
        'activation_checkpointing': True, 'target_suffixes': ['attn.qkv', 'attn.proj'],
        'load_checkpoint_path': '${STAGE1_ENCODER_LORA}',
        'warm_start_at_init': True, 'freeze': True,
    },
    'predictor_lora': {
        'enabled': True, 'rank': 8, 'alpha': 16.0, 'dropout': 0.05,
        'last_n_blocks': depth,
        'lr_mult': 0.5, 'weight_decay': 0.0001,
        'activation_checkpointing': True, 'target_suffixes': ['attn.qkv', 'attn.proj'],
    },
    'gaze': {'mode': 'none'},
}
config_path.parent.mkdir(parents=True, exist_ok=True)
Path('${OUTPUT_DIR}').mkdir(parents=True, exist_ok=True)
with config_path.open('w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f'Config: {config_path} depth={depth}')
PY

cd "${VJEPA_ROOT}"
"${PYTHON}" -m evals.main --fname "${CONFIG_PATH}" --devices cuda:0

# Write a small summary line for the sweep aggregator.
RUN_DIR="${OUTPUT_DIR}/action_anticipation_frozen/${TAG}"
SUMMARY="${OUTPUT_ROOT}/sweep_summary.csv"
mkdir -p "${OUTPUT_ROOT}"
if [[ ! -f "${SUMMARY}" ]]; then
    echo "depth,job_id,array_id,best_top5,last_top5,best_epoch,run_dir" > "${SUMMARY}"
fi
"${PYTHON}" - <<PY
import csv
from pathlib import Path
run = Path('${RUN_DIR}')
csv_path = run / 'topk_log_r0.csv'
best = last = float('nan')
best_ep = ''
if csv_path.exists():
    rows = list(csv.DictReader(csv_path.open()))
    vals = []
    for r in rows:
        try:
            vals.append((float(r['val-action-top5']), r.get('validation', '')))
        except Exception:
            pass
    if vals:
        best, best_ep = max(vals, key=lambda x: x[0])
        last = vals[-1][0]
summary = Path('${SUMMARY}')
with summary.open('a') as f:
    f.write(f"${DEPTH},${SLURM_ARRAY_JOB_ID:-},${SLURM_ARRAY_TASK_ID:-},{best:.5f},{last:.5f},{best_ep},{run}\n")
print(f'depth=${DEPTH} best_top5={best} last_top5={last} best_ep={best_ep}')
PY

echo "P01 predictor-depth depth=${DEPTH} complete: $(date)"
