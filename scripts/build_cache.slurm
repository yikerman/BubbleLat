#!/bin/bash
# One-time: stream all BubbleML_2 pool-boiling WebDataset shards (in order, sequential reads)
# and write the 128x128 float32 cache of dfun, temperature, velx, vely: one .npy + .json per run.
# Holds the whole cache (~48 GB) in RAM until the end, writes it to $TMPDIR, then copies to /pub.
#SBATCH --job-name=build_cache
#SBATCH --partition=standard
#SBATCH --account=caoy44
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --tmp=100G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

source scripts/common.sh

SHARDS='/share/crsp/lab/amowli/share/BubbleML_2_poolboiling_wds/shard-{000000..000724}.tar'
uv run --frozen --no-sync python -m bubblelat.cache wds "$SHARDS" --downsample 4 \
    --workers 14 --out "$TMPDIR/cache"
du -sh "$TMPDIR/cache"
mkdir -p "$CACHE.partial" && cp -r "$TMPDIR/cache/." "$CACHE.partial/" && mv "$CACHE.partial" "$CACHE"
echo "done: $CACHE"
