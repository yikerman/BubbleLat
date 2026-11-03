"""Command line entry point: `bubblelat {fit,validate,test} --config ...`.

Models, losses and data are chosen in YAML by class_path, so a new model or loss needs
no change here. Unless the config names the wandb run, it is named after the model, its
arguments and the loss, e.g. "convae-latent_channels4-mse".
"""

from pathlib import Path

import torch
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch.loggers import WandbLogger

from .data import BubbleMLData
from .module import AutoencoderModule


def _short(spec) -> str:
    """'ConvAE' + {latent_channels: 4} -> 'convae-latent_channels4'."""
    if spec is None:
        return ""
    name = spec.class_path.rsplit(".", 1)[-1].lower()
    args = [f"{k}{'.'.join(map(str, v)) if isinstance(v, list) else v}"
            for k, v in sorted(vars(spec.init_args).items()) if v is not None]
    return "-".join([name, *args])


class WandbConfig(SaveConfigCallback):
    """Store the full resolved config with the wandb run (as run config and config.yaml)."""

    def save_config(self, trainer, pl_module, stage) -> None:
        if not isinstance(trainer.logger, WandbLogger):
            return
        run = trainer.logger.experiment
        run.config.update(self.config.as_dict(), allow_val_change=True)
        self.parser.save(self.config, Path(run.dir) / "config.yaml", skip_none=False,
                         overwrite=True, multifile=False)


class CLI(LightningCLI):
    def before_instantiate_classes(self) -> None:
        cfg = self.config[self.subcommand]
        logger = cfg.trainer.logger
        if not hasattr(logger, "init_args"):
            return
        if logger.init_args.get("name") is None:
            parts = [_short(cfg.model.model), _short(cfg.model.loss)]
            logger.init_args.name = "-".join(p for p in parts if p)
        logger.init_args.job_type = self.subcommand


def main() -> None:
    torch.set_float32_matmul_precision("high")
    CLI(AutoencoderModule, BubbleMLData, save_config_callback=WandbConfig,
        save_config_kwargs={"save_to_log_dir": False})


if __name__ == "__main__":
    main()
