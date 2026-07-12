#!/bin/bash
#SBATCH --job-name=p01_stage1_probe
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=384GB
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/ll5914/logs/p01_stage1_probe_%j.out
#SBATCH --error=/scratch/ll5914/logs/p01_stage1_probe_%j.err

set -euo pipefail

PYTHON="/scratch/ll5914/conda_envs/SVD/bin/python"
PROJECT_ROOT="/home/ll5914/Jepa_yifan/JEPA_ARVR"
VJEPA_ROOT="/home/ll5914/ARVR_Video/vjepa2"

ANN_DIR="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/stage2"
VIDEO_ROOT="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_videos"
HDEPIC_ANN_ROOT="/scratch/ll5914/datasets/HD-EPIC/hd-epic-annotations/narrations-and-action-segments"
CHECKPOINT="/scratch/ll5914/models/vjepa2/vitl.pt"
OUTPUT_DIR="/scratch/ll5914/experiments/p01_stage1_probe"
CONFIG_PATH="/scratch/ll5914/logs/p01_stage1_probe_config.yaml"
TAG="p01-stage1-probe-vitl16-256-10ep-p01fixed"

echo "========================================================"
echo "Stage 1: P01 fixed split, RGB-only, probe-only (2x H200)"
echo "Job  : ${SLURM_JOB_ID:-manual}  Node: $(hostname)"
echo "GPU  : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Start: $(date)"
echo "========================================================"

export PYTHONPATH="${PROJECT_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"

"${PYTHON}" -c "
import importlib, sys
sys.path.insert(0, '${PROJECT_ROOT}')
sys.path.insert(0, '${VJEPA_ROOT}')
importlib.import_module('app.hdepic_lora_action_anticipation.eval')
print('[stage1] app import OK')
"

"${PYTHON}" - <<PY
import sys, yaml
from pathlib import Path

sys.path.insert(0, '${PROJECT_ROOT}')
sys.path.insert(0, '${VJEPA_ROOT}')

template_path = Path('${VJEPA_ROOT}/configs/eval/vitl/ek100.yaml')
config_path   = Path('${CONFIG_PATH}')

with template_path.open() as f:
    cfg = yaml.safe_load(f)

cfg['nodes'] = 1
cfg['tasks_per_node'] = 2
cfg['cpus_per_task'] = 16
cfg['tag']  = '${TAG}'
cfg['eval_name'] = 'app.hdepic_lora_action_anticipation'
cfg['folder'] = '${OUTPUT_DIR}'
cfg['resume_checkpoint'] = True
cfg['val_only'] = False

data = cfg['experiment']['data']
data['dataset'] = 'EK100'
data['file_format'] = 1
data['base_path']     = '${VIDEO_ROOT}'
data['dataset_train'] = '${ANN_DIR}/HD_EPIC_train_vjepa.csv'
data['dataset_val']   = '${ANN_DIR}/HD_EPIC_val_vjepa.csv'
data['dataset_test']  = '${ANN_DIR}/HD_EPIC_test_vjepa.csv'
data['num_workers']   = 10
data['val_num_workers'] = 4
data['clip_balanced'] = True
data['resolution']    = 256
data['use_bfloat16']  = False

opt = cfg['experiment']['optimization']
opt['batch_size']    = 16
opt['num_epochs']    = 10
opt['use_bfloat16']  = True
opt['use_focal_loss'] = False
opt['val_every_epochs'] = 1
opt['best_metric']   = 'val-action-acc'
opt['early_stopping_patience'] = 3
opt['multihead_kwargs'] = [{
    'weight_decay': 0.0001,
    'final_weight_decay': 0.0001,
    'lr': 0.0001,
    'start_lr': 0.0,
    'final_lr': 0.0,
    'warmup': 2,
}]

cfg['model_kwargs']['checkpoint'] = '${CHECKPOINT}'

cfg['experiment']['lora'] = {
    'enabled': True,
    'rank': 8,
    'alpha': 16.0,
    'dropout': 0.05,
    'probe_train_mode': 'full',
    'train_heads': True,
    'pretrained_probe': '',
    'align_reference_metrics': True,
    'val_metric_scope': 'native',
    'val_metric_aggregation': 'metric_wise_max',
    'class_space': 'train_only',
    'temporal_sampling': 'phd_reference',
    'output_regularization': {'enabled': False},
    'hdepic_reference': {
        'annotations_pkl':   '${HDEPIC_ANN_ROOT}/HD_EPIC_Narrations.pkl',
        'verb_classes_csv':  '${HDEPIC_ANN_ROOT}/HD_EPIC_verb_classes.csv',
        'noun_classes_csv':  '${HDEPIC_ANN_ROOT}/HD_EPIC_noun_classes.csv',
    },
    'encoder_lora': {'enabled': False},
    'predictor_lora': {'enabled': False},
    'gaze': {'mode': 'none'},
}

cfg['experiment']['optimization']['grad_clip'] = 1.0

config_path.parent.mkdir(parents=True, exist_ok=True)
with config_path.open('w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f'[stage1] Config written: {config_path}')
PY

echo "[stage1] Launching training on 2x H200..."
cd "${VJEPA_ROOT}"
"${PYTHON}" -m evals.main \
    --fname "${CONFIG_PATH}" \
    --devices cuda:0 cuda:1 \
    --debugmode false

echo "Stage 1 complete: $(date)"
