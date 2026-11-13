#!/bin/bash
# One-off: probe the checkpoints of conv_ae_mse job 59531412, which ran before fit_test probed.
# Same job name, so the scores land in the same wandb runs and group. Delete after it has run.
#SBATCH --job-name=conv_ae_mse
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
    id="conv_ae_mse-59531412-s$seed"
    probe "$id" configs/model/conv_ae.yaml --seed_everything "$seed" \
        --ckpt_path "outputs/bubblelat/$id/checkpoints/last.ckpt"
done
