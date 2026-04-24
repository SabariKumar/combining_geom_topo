# Topoformer: Geometric and Topological Deep Learning for Protein Solubility

Topoformer combines **SE(3)-equivariant graph neural networks** with **topological data analysis (persistent homology)** to predict protein solubility from 3D structure. It also includes **TopoCoder**, a companion DeepSets model that learns to predict Betti numbers from persistence diagrams.

## Overview

Two models are provided:

| Model | Input | Architecture | Task |
|---|---|---|---|
| **Topoformer** | PDB structures + ESM-2 embeddings + Betti barcodes | SE(3)-Transformer with topological attention | Binary solubility classification |
| **TopoCoder** | Persistence diagrams (images or vectors) | DeepSets | Betti number regression |

### Key features
- SE(3)-equivariant message passing via [e3nn](https://e3nn.org/) fiber representations
- Persistent homology (Betti degrees 0–3) computed with [GUDHI](https://gudhi.inria.fr/), fused into transformer attention
- ESM-2 (1280-dim) + amino acid one-hot (25-dim) node features
- Distributed training with PyTorch DDP and optional AMP
- SLURM job scripts for Bridges2 and Expanse HPC clusters

---

## Installation

This project uses [Pixi](https://prefix.dev/) for reproducible environment management.

### 1. Install Pixi

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

### 2. Clone and enter the project

```bash
git clone <repo-url>
cd combining_geom_topo
```

### 3. Create the environment

```bash
pixi install
```

This resolves all dependencies from `pixi.lock`, including:
- PyTorch 2.4 with CUDA 12.4
- DGL (from `dglteam/label/th24_cu124`)
- e3nn, GUDHI, Graphein, fair-esm (ESM-2)
- Weights & Biases, scikit-learn, BioPython

> **Note:** The environment targets `linux-64` with CUDA 12.4. Ensure your system has a compatible NVIDIA driver.

### 4. Activate the environment

```bash
pixi shell
```

All subsequent commands assume the Pixi environment is active.

---

## Data

Data lives under `data/`:

```
data/
├── csvs/          # training_set.csv, test_set.csv  (columns: sid, solubility)
├── train/         # PDB files and cached .pt tensors for training proteins
├── test/          # PDB files and cached .pt tensors for test proteins
├── fastas/        # Protein sequences (JSON: {sid: sequence})
├── topocoder_labels/  # Precomputed Betti labels for TopoCoder
└── experimental/  # Additional experimental PDB structures
```

Solubility labels are binary: `1` = soluble, `0` = insoluble.

### Generating ESM-2 embeddings

ESM-2 embeddings must be precomputed before training Topoformer:

```bash
python data/gen_esm_embs.py
```

Embeddings are saved as `<sid>.pt` files alongside the PDB structures.

### Generating TopoCoder labels

```bash
python data/make_topocoder_labels.py
```

---

## Training

### Topoformer

#### Single GPU

```bash
bash runtime/topoformer/run_training_single_gpu.sh
```

Or directly with Python:

```bash
python -m torch.distributed.run --nproc_per_node=1 \
    -m runtime.topoformer.train_topoformer \
    --batch_size 20 \
    --epochs 150 \
    --learning_rate 0.0001 \
    --min_learning_rate 0.00001 \
    --weight_decay 0.01 \
    --num_layers 5 \
    --num_degrees 2 \
    --num_channels 64 \
    --num_heads 2 \
    --dropout 0.5 \
    --eq_dropout 0.2 \
    --use_layer_norm \
    --norm \
    --save_ckpt_path results/topoformer.pth \
    --wandb
```

#### Multi-GPU (single node)

```bash
bash runtime/topoformer/run_training.sh [BATCH_SIZE] [AMP] [EPOCHS] [LR] [WD] [LAYERS] [DEGREES] [CHANNELS]
# Example:
bash runtime/topoformer/run_training.sh 32 false 150 0.0001 0.01 5 2 64
```

#### SLURM (Expanse / Bridges2)

```bash
# Expanse
sbatch runtime/topoformer/run_training_expanse.sh 32 false 200 0.0001 0.01 5 2 64

# Bridges2
sbatch runtime/topoformer/run_training_slurm.sh
```

#### Key hyperparameters

| Argument | Default | Description |
|---|---|---|
| `--num_layers` | 5 | Number of SE(3)-equivariant transformer layers |
| `--num_degrees` | 2 | Number of irreducible representation degrees |
| `--num_channels` | 64 | Channels per degree |
| `--num_heads` | 2 | Attention heads |
| `--dropout` | 0.5 | Scalar feature dropout |
| `--eq_dropout` | 0.2 | Equivariant feature dropout |
| `--amp` | false | Automatic mixed precision |
| `--optimizer` | adam | `adam`, `sgd`, or `lamb` |
| `--accumulate_grad_batches` | 1 | Gradient accumulation steps |

Checkpoints are saved to `--save_ckpt_path` every `--ckpt_interval` epochs.

---

### TopoCoder

TopoCoder trains on persistence diagrams to predict Betti numbers:

```bash
python runtime/topocoder/train_topocoder.py \
    --betti_no 0 \
    --betti_no 1 \
    --betti_no 2 \
    --input_type pi \
    --train_pi_dir data/topocoder_labels \
    --checkpoint_dir results/topocoder \
    --model_save_dir results/topocoder \
    --device cuda:0 \
    --lr 5e-3 \
    --n_epochs 100 \
    --batch_size 1000 \
    --loss l1
```

**`--input_type`**: `pi` (persistence images) or `vec` (vector representations)  
**`--betti_no`**: pass multiple times to train on multiple Betti degrees simultaneously  
**`--loss`**: `l1` or `l2`

---

## Evaluation

Evaluate a trained Topoformer checkpoint on the test set:

```bash
python runtime/topoformer/evaluate_test_set.py \
    --ckpt_path results/topoformer.pth \
    --test_pdb_dir data/test \
    --test_csv data/csvs/test_set.csv \
    --barcode_dir data/test \
    --output_csv results/test_predictions.csv \
    --num_layers 5 \
    --num_degrees 2 \
    --num_channels 64 \
    --num_heads 2 \
    --batch_size 8
```

The output CSV contains columns: `sid`, `preds` (rounded), `preds_unrounded`, `targets`.

---

## Project Structure

```
combining_geom_topo/
├── pixi.toml                        # Environment configuration
├── data/                            # Datasets (PDB, CSVs, embeddings)
├── model/
│   ├── topoformer/                  # SE(3)-Transformer + BettiAttention
│   │   ├── transformer.py           # Main model definition
│   │   ├── fiber.py                 # Fiber / irrep representations
│   │   ├── layers/                  # Equivariant attention, conv, norm
│   │   └── runtime/                 # Argument parser, metrics, loggers
│   └── topocoder/                   # DeepSets Betti predictor
├── data_loading/
│   ├── topoformer/                  # ProteinDataModule (DGL graphs + ESM + barcodes)
│   └── topocoder/                   # Persistence diagram data loaders
├── runtime/
│   ├── topoformer/                  # Training & evaluation scripts + SLURM jobs
│   ├── topocoder/                   # TopoCoder training + embedding generation
│   └── baselines/                   # Random Forest baseline
└── results/                         # Checkpoints, logs, predictions
```

---

## Experiment Tracking

Training integrates with [Weights & Biases](https://wandb.ai). Pass `--wandb` to enable logging of hyperparameters, loss curves, and validation metrics.

---

## Citation

If you use this work, please cite the relevant papers for the SE(3)-Transformer, ESM-2, and GUDHI that underpin this codebase.
