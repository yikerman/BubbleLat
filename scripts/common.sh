# Sourced by every scripts/*.slurm. Submit from the repo root: `sbatch scripts/<exp>.slurm`.
#
# The job name (#SBATCH --job-name) is the experiment name. Every wandb run the job creates
# gets it as both tag and group, so the runs of one experiment filter and group together.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

EXP="${SLURM_JOB_NAME:-local}"
JOB="${SLURM_JOB_ID:-local}"
echo "exp=$EXP job=$JOB host=$(hostname) commit=$(git rev-parse --short HEAD)"
nvidia-smi -L 2>/dev/null || echo "no GPU"

CONFIGS=(--config configs/base.yaml --config configs/hpc3.yaml)
WANDB=(--trainer.logger.init_args.tags "[$EXP]" --trainer.logger.init_args.group "$EXP"
       --trainer.logger.init_args.notes "slurm job $JOB, commit $(git rev-parse --short HEAD)")

# On a compute node, build the venv in node-local scratch ($TMPDIR, deleted when the job ends):
# a venv is many small files, which /pub and HOME handle badly. Needs `#SBATCH --tmp=...`.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    export UV_PROJECT_ENVIRONMENT="$TMPDIR/venv" UV_CACHE_DIR="$TMPDIR/uv-cache"
    export UV_PYTHON_INSTALL_DIR="$TMPDIR/uv-python"  # never load the interpreter from /pub
    export PYTHONPYCACHEPREFIX="$TMPDIR/pycache"
fi
uv sync --frozen

bubblelat() { uv run --frozen --no-sync bubblelat "$@"; }

# Copy the 128x128 cache into node-local scratch once per job and train from there.
CACHE=/pub/caoy44/bubblelat_cache_128
stage_cache() {
    [[ -n "${SLURM_JOB_ID:-}" && -z "${STAGED:-}" ]] || return 0
    echo "staging $CACHE -> $TMPDIR/cache"
    time cp -r "$CACHE" "$TMPDIR/cache"
    CONFIGS+=(--data.data_dirs "[$TMPDIR/cache]")
    STAGED=1
}

# evaluate MODEL_CONFIG [overrides...]: test a model without training (one wandb run).
evaluate() {
    local model="$1"; shift
    stage_cache
    bubblelat test "${CONFIGS[@]}" --config "$model" "${WANDB[@]}" "$@"
}

# fit_test RUN MODEL_CONFIG [overrides...]: train, then test last.ckpt, both in one wandb run.
# RUN must be unique per run within the job; the wandb id is "<exp>-<job>-<RUN>", so a
# requeued job resumes the same wandb run and continues training from last.ckpt.
fit_test() {
    local run="$1" model="$2"; shift 2
    stage_cache
    local id="$EXP-$JOB-$run"
    local ckpt="outputs/bubblelat/$id/checkpoints/last.ckpt"
    local ids=(--trainer.logger.init_args.id "$id" --trainer.logger.init_args.resume allow)
    local resume=()
    [[ -f "$ckpt" ]] && resume=(--ckpt_path "$ckpt")
    bubblelat fit "${CONFIGS[@]}" --config "$model" "${WANDB[@]}" "${ids[@]}" "$@" "${resume[@]}"
    bubblelat test "${CONFIGS[@]}" --config "$model" "${WANDB[@]}" "${ids[@]}" "$@" --ckpt_path "$ckpt"
}
