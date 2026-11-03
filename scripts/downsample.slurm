#!/bin/bash
# Downsample baseline at several compression factors. No training: one wandb test run per factor.
#SBATCH --job-name=downsample
#SBATCH --partition=free-gpu32
#SBATCH --gres=gpu:L40S:1
#SBATCH --account=caoy44
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --tmp=20G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --requeue

source scripts/common.sh

for factor in 2 4 8 16; do
    evaluate configs/model/downsample.yaml --model.model.factor "$factor"
done
