#!/bin/bash
#SBATCH --job-name=topoformer_%j
#SBATCH --output=topoformer_%j.out   # temporary; moved to OUTPUT_DIR below
#SBATCH --error=topoformer_%j.err    # temporary; moved to OUTPUT_DIR below
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --account=slc154
#SBATCH -t 48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=sabarik@colostate.edu

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"
JOB_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/results/${JOB_ID}}"
mkdir -p "${OUTPUT_DIR}"

# CLI args with defaults
BATCH_SIZE=${1:-20}
AMP=${2:-false}
NUM_EPOCHS=${3:-20}
LEARNING_RATE=${4:-0.0001}
WEIGHT_DECAY=${5:-0.01}
NUM_LAYERS=${6:-5}
NUM_DEGREES=${7:-2}
NUM_CHANNELS=${8:-64}

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

echo "Job ID: ${JOB_ID}"
echo "Node:   $(hostname)"
echo "Start:  $(date)"

# Activate pixi environment (set +u: pixi's shell-hook sources scripts that
# reference unset variables like ZSH_VERSION)
set +u
eval "$(pixi shell-hook --manifest-path "${REPO_DIR}/pixi.toml")"
set -u

nvidia-smi

# ---------------------------------------------------------------------------
# Resolve GPU count from CUDA_VISIBLE_DEVICES (set by SLURM for all GPU
# allocations regardless of whether --gpus or --gres=gpu:N was used).
# Falls back to SLURM_GPUS, then to 1.
# ---------------------------------------------------------------------------
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    N_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
else
    N_GPUS="${SLURM_GPUS:-1}"
fi
echo "GPUs:   ${N_GPUS} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"

# Use a job-specific port to avoid conflicts between concurrent jobs
MASTER_PORT=$((29500 + (SLURM_JOB_ID % 1000)))

# ---------------------------------------------------------------------------
# Run training
# ---------------------------------------------------------------------------
cd "${REPO_DIR}"

torchrun \
    --nproc_per_node="${N_GPUS}" \
    --master_addr="localhost" \
    --master_port="${MASTER_PORT}" \
    -m runtime.topoformer.train_topoformer \
    --amp "$AMP" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$NUM_EPOCHS" \
    --lr "$LEARNING_RATE" \
    --min_lr 0.00001 \
    --weight_decay "$WEIGHT_DECAY" \
    --use_layer_norm \
    --norm \
    --save_ckpt_path "${OUTPUT_DIR}/topoformer_model.pth" \
    --precompute_bases \
    --seed 42 \
    --wandb \
    --low_memory \
    --num_layers "$NUM_LAYERS" \
    --num_degrees "$NUM_DEGREES" \
    --num_heads 2
#   --num_channels "$NUM_CHANNELS"

echo "Finished: $(date)"
echo "Outputs:  ${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# Move SLURM log files into the output directory
# ---------------------------------------------------------------------------
if [ -n "${SLURM_JOB_ID:-}" ]; then
    for f in "topoformer_${SLURM_JOB_ID}.out" "topoformer_${SLURM_JOB_ID}.err"; do
        [ -f "${f}" ] && mv "${f}" "${OUTPUT_DIR}/"
    done
fi
