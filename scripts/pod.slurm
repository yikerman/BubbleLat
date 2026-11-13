#!/bin/bash
# POD baseline at the latent sizes of Downsample factors 16, 8 and 4 (256x, 64x, 16x compression).
# No training: each rank refits the SVD on the CPU and logs one wandb test run.
#SBATCH --job-name=pod
#SBATCH --partition=free-gpu32
#SBATCH --gres=gpu:1
#SBATCH --account=caoy44
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --tmp=100G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --requeue

source scripts/common.sh

for rank in 256 1024 4096; do
    evaluate configs/model/pod.yaml --model.model.rank "$rank"
done
