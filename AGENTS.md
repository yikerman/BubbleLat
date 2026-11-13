# AGENTS.md

Solo EECS 221 (ML for Science, UCI, Fall 2026) final project: learned compression of BubbleML
single frames, then whether the latents mean anything. See `ROADMAP.md` for plan and papers.

## Principles
- KISS. Small readable modules, no abstraction until a second use exists.
- Single frame (C, H, W) only for now; temporal models come later.
- Splits are by run, never by frame (see `split_runs` in `bubblelat/data.py`). The data directory
  is fixed, so the split rule is fixed too: per directory, sorted by wall temperature.
- Metrics are computed in physical units; losses are computed on normalized fields.

## Layout
- `bubblelat/data.py` · `models.py` · `losses.py` · `metrics.py` · `module.py` · `cli.py`
- `bubblelat/probe.py`: per-frame labels and the ridge readout probe; `bubblelat predict` dumps
  latents and logs `probe/*` to the model's wandb run.
- `bubblelat/cache.py`: builds the training cache, one float32 `(T, C, H, W)` `.npy` + `.json` per run.
  Everything trains and evaluates at 128x128 (512 average-pooled by 4), in full precision.
- `configs/base.yaml` (local, sample data), `configs/hpc3.yaml` (cluster data), `configs/model/*.yaml`
- `scripts/*.slurm`: one experiment per script. The job name is the wandb tag and group;
  `scripts/common.sh` holds the shared helpers (`evaluate`, `fit_test`, `probe`).

## Extending
- Model: subclass `Autoencoder` (`encode`, `decode`), and add a `configs/model/<name>.yaml`.
- Loss: an `nn.Module` with `forward(out, x) -> {"loss": ..., ...}`.
- Experiment: copy a `scripts/*.slurm`, change `--job-name` and the loop.

## Commands
```bash
uv sync
uv run python -m bubblelat.cache hdf5 ../BubbleGen/samples --out data/samples   # local cache, once
WANDB_MODE=offline uv run bubblelat validate --config configs/base.yaml --config configs/model/downsample.yaml
uv run ruff check bubblelat
sbatch scripts/<exp>.slurm        # on HPC3, from the repo root
```

## Rules
- Commits and PRs: no `Co-Authored-By` or other AI attribution lines.
- wandb is logged in locally and on HPC3. Use `WANDB_MODE=offline` for local smoke tests.
- HPC3 (`caoy44@hpc3.rcic.uci.edu`; repo at `~/BubbleLat` (home; `/pub` is too slow), no venv there: every job
  runs `uv sync` into `$TMPDIR`, see `scripts/common.sh`). Ask before submitting jobs
  or writing files. Don't do heavy or random reads of `/share/crsp` (no `du`, no h5py scans on
  login nodes).
- Data: the 128x128 cache at `/pub/caoy44/bubblelat_cache_128` is the working dataset (94 runs in 5
  directories; no Subcooled-R515B); jobs copy it to `$TMPDIR`. It was built 2026-11-03 from the
  WebDataset `BubbleML_2_poolboiling_wds`, which has since been removed from the share, so it cannot
  be rebuilt that way: don't delete or overwrite it. The raw HDF5 runs are still under
  `/share/crsp/lab/amowli/share/BubbleML_2`.
- Deadlines: check-in Nov 15, presentation Dec 8, report Dec 11 (6 pages + reproducible README).
