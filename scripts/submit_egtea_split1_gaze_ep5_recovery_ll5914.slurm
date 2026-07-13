#!/bin/bash
#SBATCH --job-name=egtea_gaze_ep5
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=384GB
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/ll5914/logs/egtea_gaze_ep5_recovery_%j.out
#SBATCH --error=/scratch/ll5914/logs/egtea_gaze_ep5_recovery_%j.err

set -euo pipefail

# Re-train EGTEA gaze stage-1 for exactly 5 epochs into a fresh output dir.
# Job 12986096 peaked at epoch 5 (val-action-top3=54.51, top5~65.8) but resume
# job 13006291 reset the best tracker and overwrote best.pt at epoch 8. The ep5
# weights were not preserved in any archive.

PYTHON="/scratch/ll5914/conda_envs/SVD/bin/python"
PROJECT_ROOT="/home/ll5914/Jepa_yifan/JEPA_ARVR"
VJEPA_ROOT="/home/ll5914/ARVR_Video/vjepa2"
EGTEA_ROOT="/scratch/ll5914/datasets/EGTEA"

"${PYTHON}" "${PROJECT_ROOT}/scripts/prepare_egtea_split1.py" --egtea-root "${EGTEA_ROOT}"

ANN_DIR="${EGTEA_ROOT}/hdepic_vjepa_annotations/split1"
VIDEO_ROOT="${EGTEA_ROOT}/egtea_vjepa_videos"
GAZE_EXTRACT="${EGTEA_ROOT}/gaze_extract"
CHECKPOINT="/scratch/ll5914/models/vjepa2/vitl.pt"
OUTPUT_DIR="/scratch/ll5914/experiments/egtea_split1_gaze_ep5_recovery"
CONFIG_PATH="/scratch/ll5914/logs/egtea_split1_gaze_ep5_recovery_config.yaml"
TAG="egtea-split1-gaze-ep5-recovery-vitl16-256-5ep"

export PYTHONPATH="${PROJECT_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"

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
cfg['resume_checkpoint'] = False
cfg['val_only'] = False
data = cfg['experiment']['data']
data.update({
    'dataset': 'EK100', 'file_format': 1,
    'base_path': '${VIDEO_ROOT}',
    'dataset_train': '${ANN_DIR}/EGTEA_train_vjepa.csv',
    'dataset_val': '${ANN_DIR}/EGTEA_val_vjepa.csv',
    'dataset_test': '${ANN_DIR}/EGTEA_test_vjepa.csv',
    'num_workers': 12, 'val_num_workers': 8, 'clip_balanced': True,
    'resolution': 256, 'frames_per_clip': 32, 'frames_per_second': 8,
    'anticipation_time_sec': [1.0, 1.0],
    'train_anticipation_time_sec': [0.25, 1.75],
})
opt = cfg['experiment']['optimization']
opt.update({'batch_size': 16, 'num_epochs': 5, 'use_bfloat16': True,
    'use_focal_loss': False, 'val_every_epochs': 1, 'best_metric': 'val-action-top5',
    'early_stopping_patience': 0, 'grad_clip': 1.0,
    'multihead_kwargs': [{'weight_decay': 0.0001, 'final_weight_decay': 0.0001,
        'lr': 0.0001, 'start_lr': 0.0, 'final_lr': 0.0, 'warmup': 2}]})
cfg['model_kwargs']['checkpoint'] = '${CHECKPOINT}'
cfg['experiment']['lora'] = {
    'enabled': True, 'rank': 8, 'alpha': 16.0, 'dropout': 0.05,
    'probe_train_mode': 'full', 'train_heads': True, 'pretrained_probe': '',
    'align_reference_metrics': True, 'val_metric_scope': 'native',
    'val_metric_aggregation': 'metric_wise_max', 'class_space': 'train_only',
    'temporal_sampling': 'phd_reference', 'output_regularization': {'enabled': False},
    'encoder_lora': {'enabled': True, 'rank': 8, 'alpha': 16.0, 'dropout': 0.05,
        'last_n_blocks': 0, 'lr_mult': 0.5, 'weight_decay': 0.0001,
        'activation_checkpointing': True, 'target_suffixes': ['attn.qkv', 'attn.proj']},
    'predictor_lora': {'enabled': False},
    'gaze': {'mode': 'binary_input_adapter', 'extract_root': '${GAZE_EXTRACT}',
        'slam_root': '${EGTEA_ROOT}',
        'input_adapter': {'in_channels': 4, 'temporal_kernel': 1, 'hidden_dim': 8,
            'lr_mult': 1.0, 'weight_decay': 0.0001, 'activation_checkpointing': True,
            'disable_train_aug': True},
        'pose_slam': {'enabled': False}},
}
config_path.parent.mkdir(parents=True, exist_ok=True)
with config_path.open('w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f'Config: {config_path}')
PY

cd "${VJEPA_ROOT}"
"${PYTHON}" -m evals.main --fname "${CONFIG_PATH}" --devices cuda:0 cuda:1
RUN_DIR="${OUTPUT_DIR}/action_anticipation_frozen/${TAG}"
touch "${RUN_DIR}/.stage1_complete"
echo "EGTEA gaze ep5 recovery complete: $(date)"
