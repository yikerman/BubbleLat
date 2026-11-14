#!/bin/bash
# ConvAE + MSE at 4x, 16x, 64x and 256x compression, 3 seeds each. One array task per ratio, one
# wandb run per seed.
# The ratio is set by the number of pooling levels (4^levels) with 4 latent channels throughout.
# Pooling further and adding latent channels reaches the same ratios with 1.3x to 4x the error,
# and 1 latent channel is worse still.
# Paid partition, class GPU account (gpu32 needs a GPU32 account, which we do not have):
# free-gpu32 cancels preempted jobs, and a resubmitted job gets a new id, so it would not resume.
#SBATCH --job-name=conv_ae_ratio
#SBATCH --array=1-4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A30:1   # torch 2.14 cu130 has no kernels for the V100s in this partition
#SBATCH --account=eecs195221_class_gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --tmp=100G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A_%a.out

source scripts/common.sh

levels="$SLURM_ARRAY_TASK_ID"
widths=(32 64 128 128)
channels="[$(IFS=,; echo "${widths[*]:0:$levels}")]"
for seed in 0 1 2; do
    fit_test "L$levels-s$seed" configs/model/conv_ae.yaml --seed_everything "$seed" \
        --model.model.channels "$channels" --model.model.latent_channels 4
done
