#!/bin/bash
# One-off: re-test the runs on the scatter so that their wandb summaries get the per-wavenumber
# spectrum, which the test step did not log when they ran. Delete after it has run.
#SBATCH --job-name=spectra
#SBATCH --partition=free-gpu32
#SBATCH --gres=gpu:1
#SBATCH --account=caoy44
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --tmp=100G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out

source scripts/common.sh
stage_cache

# retest ID MODEL_CONFIG [overrides...]: test into the existing wandb run ID, keeping its group.
retest() {
    local id="$1" model="$2"; shift 2
    local exp="${id%%-*}"
    bubblelat test "${CONFIGS[@]}" --config "$model" "${WANDB[@]}" \
        --trainer.logger.init_args.tags "[$exp]" --trainer.logger.init_args.group "$exp" \
        --trainer.logger.init_args.id "$id" --trainer.logger.init_args.resume allow "$@"
}

widths=(32 64 128 128)
for run in 59628671-L1 59628693-L2 59628706-L3 59628646-L4; do
    levels="${run##*L}"
    channels="[$(IFS=,; echo "${widths[*]:0:$levels}")]"
    for seed in 0 1 2; do
        id="conv_ae_ratio-$run-s$seed"
        retest "$id" configs/model/conv_ae.yaml --seed_everything "$seed" \
            --model.model.channels "$channels" --model.model.latent_channels 4 \
            --ckpt_path "outputs/bubblelat/$id/checkpoints/last.ckpt"
    done
done
for seed in 0 1 2; do
    retest "conv_ae_random-59541970-e$((seed + 1))" configs/model/conv_ae.yaml --seed_everything "$seed"
done
n=0
for factor in 2 4 8 16; do
    retest "downsample-59541969-e$((++n))" configs/model/downsample.yaml --model.model.factor "$factor"
done
n=0
for rank in 256 1024 4096; do
    retest "pod-59541968-e$((++n))" configs/model/pod.yaml --model.model.rank "$rank"
done
