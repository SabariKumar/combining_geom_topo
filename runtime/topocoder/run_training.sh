#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --account=skumar
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --job-name=topoformer
#SBATCH --output=slurm_outputs/topoformer_%j.out
#SBATCH --error=slurm_outputs/topoformer_%j.err
#SBATCH --partition=day-long-gpu
#SBATCH --gres=gpu:3
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=sabarik@colostate.edu

cd /nfs/home/skumar/ProteinSol/topoformer/
module load singularity
SINGULARITY_TMPDIR=/nfs/home/skumar/singularity_temp SINGULARITYENV_WANDB_API_KEY=046f107c2d4757c1423c36e02a3b22da8ed18c71 SINGULARITYENV_WANDB_PROJECT=topoformer singularity exec --nv -B /nfs/home/skumar/ProteinSol/  20241008_topoformer_fixedgrads.sif python ./train_ripsnet.py --betti_no 1 --betti_no 2 --betti_no 3 --input_type pi --train_pi_dir ../../data/soluprotgeom_expt/ --checkpoint_dir ../../results/ripsnet --model_save_dir ../../results/ripsnet --device cuda:0