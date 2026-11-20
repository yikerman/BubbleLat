#!/bin/bash
# ViT backbone at 64x compression (patch 8, 4 latent channels: the latent shape of the 3-level
# ConvAE), the attention counterpart of conv_vae + conv_ae_ratio task 3. Task 0 is the plain ViTAE
# (the beta = 0 point), tasks 1-2 the beta-VAE. One array task per beta, one wandb run per seed.
# Paid partition and class GPU account, as in conv_ae_ratio.slurm.
#SBATCH --job-name=vit_vae
#SBATCH --array=0-2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A30:1   # torch 2.14 cu130 has no kernels for the V100s in this partition
#SBATCH --account=eecs195221_class_gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --tmp=100G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A_%a.out

source scripts/common.sh

betas=(0 0.01 0.1)
beta="${betas[$SLURM_ARRAY_TASK_ID]}"
for seed in 0 1 2; do
    if [[ "$beta" == 0 ]]; then
        fit_test "b0-s$seed" configs/model/vit_ae.yaml --seed_everything "$seed"
    else
        fit_test "b$beta-s$seed" configs/model/vit_vae.yaml --seed_everything "$seed" \
            --model.loss.init_args.beta "$beta"
    fi
done
