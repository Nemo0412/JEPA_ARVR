#!/bin/bash
#SBATCH --job-name=training
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4       # Launch 4 tasks on a single node
#SBATCH --gpus-per-task=2         # 2 GPUs per task
#SBATCH --gres=gpu:8
#SBATCH --time=2:00:00
#SBATCH --partition=advanced-gpu8
#SBATCH --cpus-per-task=32
#SBATCH --mem=1000GB

source ~/anaconda3/etc/profile.d/conda.sh
conda activate scalant


YOUR_ROOT_PATH=...
YOUR_FEATURE_DIR=...


# Array to handle GPU device IDs
declare -a gpu_ids=("0,1" "2,3" "4,5" "6,7")

for i in {0..3}; do
    CUDA_VISIBLE_DEVICES=${gpu_ids[$i]} python main.py \
      --cfg configs/ek100/default.yaml \
      --opts \
      DATA.DATA_ROOT_PATH ${YOUR_ROOT_PATH} \
      DATA.FEAT_DIR ${YOUR_FEATURE_DIR} &
done
wait
