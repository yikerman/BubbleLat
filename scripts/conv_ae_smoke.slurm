#!/bin/bash
# Test drive: ConvAE + MSE for a few hundred batches on the 128x128 cache. Any GPU type.
#SBATCH --job-name=conv_ae_smoke
#SBATCH --partition=free-gpu32
#SBATCH --gres=gpu:1
#SBATCH --account=caoy44
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --tmp=100G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

source scripts/common.sh

fit_test s0 configs/model/conv_ae.yaml \
    --data.stats_frames 16 \
    --trainer.max_epochs 2 --trainer.limit_train_batches 100 \
    --trainer.limit_val_batches 10 --trainer.limit_test_batches 10
