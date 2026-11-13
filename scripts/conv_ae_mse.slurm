#!/bin/bash
# ConvAE (3 levels, 4 latent channels) trained with MSE, 3 seeds. One wandb run per seed.
# Paid partition (43 SU/h with an L40S): free-gpu32 cancels preempted jobs, and a resubmitted
# job gets a new id, so it would not resume. About 2 h per seed.
#SBATCH --job-name=conv_ae_mse
#SBATCH --partition=gpu32
#SBATCH --gres=gpu:L40S:1
#SBATCH --account=caoy44
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --tmp=100G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

source scripts/common.sh

for seed in 0 1 2; do
    fit_test "s$seed" configs/model/conv_ae.yaml --seed_everything "$seed"
done
