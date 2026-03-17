#!/usr/bin/env bash

# CLI args with defaults
BATCH_SIZE=${1:-20}
AMP=${2:-false}
NUM_EPOCHS=${3:-20}
LEARNING_RATE=${4:-0.0001}
WEIGHT_DECAY=${5:-0.01}
NUM_LAYERS=${6:-5}
NUM_DEGREES=${7:-2}
NUM_CHANNELS=${8:-64}

# choices: 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv',
#          'U0_atom', 'U_atom', 'H_atom', 'G_atom', 'A', 'B', 'C'

/usr/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=gpu --max_restarts 0 --module \
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
  