#!/bin/bash
# ConvAE (3 levels, 4 latent channels) trained with MSE, 3 seeds. One wandb run per seed.
#SBATCH --job-name=conv_ae_mse
#SBATCH --partition=free-gpu32
#SBATCH --gres=gpu:L40S:1
#SBATCH --account=caoy44
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --tmp=100G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --requeue

source scripts/common.sh

for seed in 0 1 2; do
    fit_test "s$seed" configs/model/conv_ae.yaml --seed_everything "$seed"
done
