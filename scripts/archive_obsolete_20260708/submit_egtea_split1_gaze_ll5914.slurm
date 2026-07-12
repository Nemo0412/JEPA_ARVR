#!/bin/bash
#SBATCH --job-name=egtea_s1_gaze
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=384GB
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/ll5914/logs/egtea_split1_gaze_%j.out
#SBATCH --error=/scratch/ll5914/logs/egtea_split1_gaze_%j.err

set -euo pipefail

# EGTEA Gaze+ split1: Video + Gaze (binary_input_adapter, 4ch), no pose.
# Train: training1.csv | Val/Test: validation1.csv (standard split1 benchmark)

PYTHON="/scratch/ll5914/conda_envs/SVD/bin/python"
PROJECT_ROOT="/home/ll5914/Jepa_yifan/JEPA_ARVR"
VJEPA_ROOT="/home/ll5914/ARVR_Video/vjepa2"

EGTEA_ROOT="/scratch/ll5914/datasets/EGTEA"
ANN_DIR="${EGTEA_ROOT}/hdepic_vjepa_annotations/split1"
VIDEO_ROOT="${EGTEA_ROOT}/egtea_vjepa_videos"
GAZE_EXTRACT="${EGTEA_ROOT}/gaze_extract"
CHECKPOINT="/scratch/ll5914/models/vjepa2/vitl.pt"
OUTPUT_DIR="/scratch/ll5914/experiments/egtea_split1_gaze"
CONFIG_PATH="/scratch/ll5914/logs/egtea_split1_gaze_config.yaml"
TAG="egtea-split1-gaze-vitl16-256-10ep"

echo "========================================================"
echo "EGTEA split1: Video + Gaze (no pose), binary_input_adapter"
echo "Job  : ${SLURM_JOB_ID:-manual}  Node: $(hostname)"
echo "Start: $(date)"
echo "========================================================"

# Verify prep completed
if [[ ! -f "${ANN_DIR}/EGTEA_train_vjepa.csv" ]]; then
    echo "ERROR: Run prepare_egtea_split1 first." >&2
    exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"

"${PYTHON}" -c "
import importlib, sys
sys.path.insert(0, '${PROJECT_ROOT}')
sys.path.insert(0, '${VJEPA_ROOT}')
importlib.import_module('app.hdepic_lora_action_anticipation.eval')
print('[egtea] import OK')
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
data['dataset_train'] = '${ANN_DIR}/EGTEA_train_vjepa.csv'
data['dataset_val']   = '${ANN_DIR}/EGTEA_val_vjepa.csv'
data['dataset_test']  = '${ANN_DIR}/EGTEA_test_vjepa.csv'
data['num_workers']   = 8
data['val_num_workers'] = 4
data['clip_balanced'] = True
data['resolution']    = 256
data['frames_per_clip'] = 32
data['frames_per_second'] = 8
data['anticipation_time_sec'] = [1.0, 1.0]
data['train_anticipation_time_sec'] = [0.25, 1.75]

opt = cfg['experiment']['optimization']
opt['batch_size']    = 16
opt['num_epochs']    = 10
opt['use_bfloat16']  = True
opt['use_focal_loss'] = False
opt['val_every_epochs'] = 1
opt['best_metric']   = 'val-action-acc'
opt['early_stopping_patience'] = 3
opt['grad_clip'] = 1.0
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
    'encoder_lora': {
        'enabled': True,
        'rank': 8,
        'alpha': 16.0,
        'dropout': 0.05,
        'last_n_blocks': 0,
        'lr_mult': 0.5,
        'weight_decay': 0.0001,
        'activation_checkpointing': True,
        'target_suffixes': ['attn.qkv', 'attn.proj'],
    },
    'predictor_lora': {'enabled': False},
    'gaze': {
        'mode': 'binary_input_adapter',
        'extract_root': '${GAZE_EXTRACT}',
        'slam_root': '${EGTEA_ROOT}',
        'binary_input_adapter': {
            'in_channels': 4,
            'temporal_kernel': 3,
            'hidden_dim': 8,
            'lr_mult': 1.0,
            'weight_decay': 0.0001,
            'activation_checkpointing': True,
            'disable_train_aug': True,
        },
        'pose_slam': {'enabled': False},
    },
}

config_path.parent.mkdir(parents=True, exist_ok=True)
with config_path.open('w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f'[egtea] Config written: {config_path}')
PY

echo "[egtea] Launching training on 2x H200..."
cd "${VJEPA_ROOT}"
"${PYTHON}" -m evals.main \
    --fname "${CONFIG_PATH}" \
    --devices cuda:0 cuda:1 \
    --debugmode false

echo "========================================================"
echo "EGTEA split1 complete: $(date)"
echo "Results: ${OUTPUT_DIR}/action_anticipation_frozen/${TAG}/"
echo "========================================================"
