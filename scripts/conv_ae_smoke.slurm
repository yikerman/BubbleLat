#!/bin/bash
# Test drive: ConvAE + MSE on subcooled FC-72 only, a few hundred batches. Checks the whole
# pipeline on real data while keeping reads from shared storage small.
#SBATCH --job-name=conv_ae_smoke
#SBATCH --partition=free-gpu32
#SBATCH --gres=gpu:L40S:1
#SBATCH --account=caoy44
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --tmp=20G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

source scripts/common.sh

fit_test s0 configs/model/conv_ae.yaml \
    --data.data_dirs '[/share/crsp/lab/amowli/share/BubbleML_2/PoolBoiling-Subcooled-FC72-2D]' \
    --data.num_workers 4 --data.stats_frames 16 \
    --trainer.max_epochs 2 --trainer.limit_train_batches 100 \
    --trainer.limit_val_batches 10 --trainer.limit_test_batches 10
