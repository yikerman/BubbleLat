# BubbleLat

Learned compression of BubbleML single frames, and how meaningful the resulting latents are.

## Layout

| file | role |
| --- | --- |
| `bubblelat/data.py` | reads `.hdf5` + `.json` runs, fixed split by wall temperature, standardizes each field |
| `bubblelat/models.py` | `Autoencoder` interface (`encode`, `decode`), `Downsample` baseline, `ConvAE` |
| `bubblelat/losses.py` | `MSE` (a loss returns a dict whose `"loss"` entry is optimized) |
| `bubblelat/metrics.py` | MAE, RMSE, VRMSE per field, in physical units |
| `bubblelat/module.py` | Lightning module: training, val/test metrics, wandb tables and sample images |
| `bubblelat/cli.py` | `bubblelat` command (LightningCLI), names wandb runs |
| `configs/` | `base.yaml` (trainer + data), plus one file per model in `configs/model/` |

## Usage

```bash
uv sync
# baseline: nothing to train, so evaluate it directly
uv run bubblelat test --config configs/base.yaml --config configs/model/downsample.yaml
# train, then test the checkpoint with the config saved in the wandb run
uv run bubblelat fit  --config configs/base.yaml --config configs/model/conv_ae.yaml
uv run bubblelat test --config outputs/wandb/<run>/files/config.yaml \
    --ckpt_path outputs/bubblelat/<id>/checkpoints/<ckpt> --trainer.logger.init_args.name null
```

Override any argument from the command line, e.g. `--model.model.factor 8` or `--data.downsample 4`.
Use `WANDB_MODE=offline` for quick local runs.

wandb runs are named after the model, its arguments and the loss (for example
`convae-channels32.64.128-in_channels4-latent_channels4-mse`), and `job_type` is the subcommand.
Each run stores its full resolved config, the train/val/test run ids and the parameter count.

## Adding things

- **Model:** subclass `Autoencoder` in `models.py` and point `model.model.class_path` at it.
- **Loss:** an `nn.Module` with `forward(out, x) -> {"loss": ..., ...}`; set `model.loss.class_path`.
- **Data:** set `data.data_dirs` to the BubbleML 2 directories. The split is fixed: within each
  directory, runs sorted by wall temperature go to val at positions 1, 6, 11, ... and to test at
  3, 8, 13, ... (`data.holdout_every`, default 5); everything else, including both extremes, trains.
