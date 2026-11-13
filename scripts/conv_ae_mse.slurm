#!/bin/bash
# ConvAE (3 levels, 4 latent channels) trained with MSE, 3 seeds. One wandb run per seed.
# Paid partition, class GPU account (gpu32 needs a GPU32 account, which we do not have):
# free-gpu32 cancels preempted jobs, and a resubmitted job gets a new id, so it would not resume.
#SBATCH --job-name=conv_ae_mse
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A30:1   # torch 2.14 cu130 has no kernels for the V100s in this partition
#SBATCH --account=eecs195221_class_gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --tmp=100G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

source scripts/common.sh

for seed in 0 1 2; do
    fit_test "s$seed" configs/model/conv_ae.yaml --seed_everything "$seed"
done
