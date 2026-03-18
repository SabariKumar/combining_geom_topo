#!/bin/bash
#SBATCH --job-name=protein_se3
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100-32:4
#SBATCH -p GPU-shared
#SBATCH --output=%g_%x_out.log
#SBATCH --error=%g_%x_err.log
#SBATCH --mail-type=END
#SBATCH --mail-user=sabarik@colostate.edu

# CLI args with defaults
BATCH_SIZE=${1:-20}
AMP=${2:-false}
NUM_EPOCHS=${3:-20}
LEARNING_RATE=${4:-0.0001}
WEIGHT_DECAY=${5:-0.01}
NUM_LAYERS=${6:-5}
NUM_DEGREES=${7:-2}
NUM_CHANNELS=${8:-64}

python -m torch.distributed.run --nnodes=1 --nproc_per_node=gpu --max_restarts 0 --module \
  runtime.topoformer.train_topoformer \
  --amp "$AMP" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$NUM_EPOCHS" \
  --lr "$LEARNING_RATE" \
  --min_lr 0.00001 \
  --weight_decay "$WEIGHT_DECAY" \
  --use_layer_norm \
  --norm \
  --save_ckpt_path results/20250605_baseline_model_topoformer.pth \
  --precompute_bases \
  --seed 42 \
  --wandb \
  --low_memory \
  --num_layers "$NUM_LAYERS" \
  --num_degrees "$NUM_DEGREES" \
  --num_heads 2
#  --num_channels "$NUM_CHANNELS"
