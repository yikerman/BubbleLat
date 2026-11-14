#!/bin/bash
# beta-VAE on the ConvAE backbone at 64x compression (conv_ae_ratio task 3 is the beta = 0 point), 3 seeds
# per beta. One array task per beta, one wandb run per seed.
# Paid partition and class GPU account, as in conv_ae_ratio.slurm.
#SBATCH --job-name=conv_vae
#SBATCH --array=0-1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A30:1   # torch 2.14 cu130 has no kernels for the V100s in this partition
#SBATCH --account=eecs195221_class_gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --tmp=100G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A_%a.out

source scripts/common.sh

betas=(0.01 0.1)
beta="${betas[$SLURM_ARRAY_TASK_ID]}"
for seed in 0 1 2; do
    fit_test "b$beta-s$seed" configs/model/conv_vae.yaml --seed_everything "$seed" \
        --model.loss.init_args.beta "$beta"
done
