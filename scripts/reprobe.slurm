#!/bin/bash
# Rerun `predict` on finished 64x runs, for the probes added after they ran (vapor map, MLP), and
# keep their latents in $LATENTS. Scores go to each run's own wandb run, under its own group and
# tags; nothing is retrained. Trained runs load last.ckpt; POD, Downsample and the random
# encoders are rebuilt from their config and seed, as `evaluate` built them.
#SBATCH --job-name=reprobe
# Free partition: inference only, and a preempted (cancelled) job is safe to resubmit, since every
# step rewrites the same latents and resumes the same wandb run.
#SBATCH --partition=free-gpu32
#SBATCH --gres=gpu:1
#SBATCH --account=caoy44
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --tmp=100G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A.out

source scripts/common.sh

# reprobe ID MODEL_CONFIG [overrides...]: `probe` run ID ("<exp>-<job>-<run>") under its own
# experiment's group and tags, not this job's. To redo only some experiments, e.g. after a failure
# part way: sbatch --export=ALL,ONLY="pod downsample" scripts/reprobe.slurm
reprobe() {
    local group="${1%%-[0-9]*}"
    [[ -z "${ONLY:-}" || " $ONLY " == *" $group "* ]] || return 0
    WANDB=(--trainer.logger.init_args.tags "[$group]" --trainer.logger.init_args.group "$group")
    probe "$@"
}
ckpt() { echo "outputs/bubblelat/$1/checkpoints/last.ckpt"; }

for s in 0 1 2; do
    id="conv_ae_ratio-59628706-L3-s$s"
    reprobe "$id" configs/model/conv_ae.yaml --seed_everything "$s" --ckpt_path "$(ckpt "$id")"
    id="vit_vae-59858583-b0-s$s"
    reprobe "$id" configs/model/vit_ae.yaml --seed_everything "$s" --ckpt_path "$(ckpt "$id")"
    id="vit_vae-59858611-b0.01-s$s"
    reprobe "$id" configs/model/vit_vae.yaml --seed_everything "$s" --ckpt_path "$(ckpt "$id")" \
        --model.loss.init_args.beta 0.01
    reprobe "conv_ae_random-59541970-e$((s + 1))" configs/model/conv_ae.yaml --seed_everything "$s"
    reprobe "vit_ae_random-59949004-e$((s + 1))" configs/model/vit_ae.yaml --seed_everything "$s"
done
id="vit_vae-59858581-b0.1-s0"
reprobe "$id" configs/model/vit_vae.yaml --seed_everything 0 --ckpt_path "$(ckpt "$id")" \
    --model.loss.init_args.beta 0.1
reprobe pod-59541968-e2 configs/model/pod.yaml --model.model.rank 1024
reprobe downsample-59541969-e3 configs/model/downsample.yaml --model.model.factor 8
