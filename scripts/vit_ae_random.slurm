#!/bin/bash
# Probe control: the 64x ViTAE (vit_vae task 0) at random initialization, never trained. Its probe
# scores are what the architecture alone reads out. One wandb run (test + probe) per seed.
#SBATCH --job-name=vit_ae_random
#SBATCH --partition=free-gpu32
#SBATCH --gres=gpu:1
#SBATCH --account=caoy44
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --tmp=100G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

source scripts/common.sh

for seed in 0 1 2; do
    evaluate configs/model/vit_ae.yaml --seed_everything "$seed"
done
