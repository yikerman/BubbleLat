#!/bin/bash
# ConvAE + MSE at 256x, 16x and 4x compression (1, 16 and 64 latent channels; conv_ae_mse is the
# 64x point with 4), 3 seeds each. One array task per latent size, one wandb run per seed.
# Paid partition and class GPU account, as in conv_ae_mse.slurm.
#SBATCH --job-name=conv_ae_ratio
#SBATCH --array=1,16,64
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A30:1   # torch 2.14 cu130 has no kernels for the V100s in this partition
#SBATCH --account=eecs195221_class_gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --tmp=100G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A_%a.out

source scripts/common.sh

c="$SLURM_ARRAY_TASK_ID"
for seed in 0 1 2; do
    fit_test "c$c-s$seed" configs/model/conv_ae.yaml --seed_everything "$seed" \
        --model.model.latent_channels "$c"
done
