"""Lightning module that trains and evaluates any `Autoencoder` with any loss.

wandb layout:
  train/loss                      every step
  val/<metric>/<field>            every validation epoch, in physical units
  test/<metric>/<field>           once, plus a "test/table" with one row per field
  <stage>/samples                 target | reconstruction | error for one fixed frame
"""

from collections import defaultdict

import lightning as L
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger
from torch import nn

from .metrics import METRICS, field_errors
from .models import Autoencoder


class AutoencoderModule(L.LightningModule):
    def __init__(self, model: Autoencoder, loss: nn.Module | None = None, lr: float = 1e-3,
                 weight_decay: float = 0.0):
        super().__init__()
        self.model, self.loss = model, loss
        self.lr, self.weight_decay = lr, weight_decay

    def setup(self, stage: str) -> None:
        dm = self.trainer.datamodule
        z = self.model.encode(torch.zeros(1, len(dm.fields), 64, 64))
        info = {"model/params": sum(p.numel() for p in self.model.parameters()),
                "model/compression_64px": len(dm.fields) * 64 * 64 / z.numel(), **dm.describe()}
        if self.logger:
            self.logger.log_hyperparams(info)

    def training_step(self, batch, _):
        losses = self.loss(self.model(batch["x"]), batch["x"])
        self.log_dict({f"train/{k}": v for k, v in losses.items()}, prog_bar=True,
                      batch_size=len(batch["x"]))
        return losses["loss"]

    # evaluation: accumulate sample-weighted sums, reduce once per epoch

    def on_validation_epoch_start(self):
        self._sums, self._count = defaultdict(float), 0

    on_test_epoch_start = on_validation_epoch_start

    def _eval_step(self, batch, batch_idx):
        dm, x = self.trainer.datamodule, batch["x"]
        out = self.model(x)
        pred, target = dm.denormalize(out["xhat"].float()), dm.denormalize(x.float())
        metrics = field_errors(pred, target, dm.fields)
        if self.loss is not None:
            metrics |= self.loss(out, x)
        for k, v in metrics.items():
            self._sums[k] += v.item() * len(x)
        self._count += len(x)
        if batch_idx == 0:
            self._sample = (target[0].cpu(), pred[0].cpu(), batch["run"][0], int(batch["t"][0]))

    validation_step = test_step = _eval_step

    def _eval_epoch_end(self, stage: str):
        means = {k: v / self._count for k, v in self._sums.items()}
        self.log_dict({f"{stage}/{k}": v for k, v in means.items()}, sync_dist=True)
        if isinstance(self.logger, WandbLogger) and not self.trainer.sanity_checking:
            self.logger.experiment.log({f"{stage}/samples": wandb.Image(self._sample_figure())})
            if stage == "test":
                fields = self.trainer.datamodule.fields
                rows = [[f] + [means[f"{m}/{f}"] for m in METRICS] for f in fields]
                self.logger.experiment.log({"test/table": wandb.Table(["field"] + METRICS, rows)})
            plt.close("all")

    def on_validation_epoch_end(self):
        self._eval_epoch_end("val")

    def on_test_epoch_end(self):
        self._eval_epoch_end("test")

    def _sample_figure(self):
        target, pred, run, t = self._sample
        fields = self.trainer.datamodule.fields
        fig, axes = plt.subplots(len(fields), 3, figsize=(9, 3 * len(fields)), squeeze=False,
                                 layout="constrained")
        for c, name in enumerate(fields):
            lo, hi = target[c].min().item(), target[c].max().item()
            err = (pred[c] - target[c]).abs().max().item()
            panels = [(target[c], "viridis", lo, hi, "target"), (pred[c], "viridis", lo, hi, "recon"),
                      (pred[c] - target[c], "RdBu_r", -err, err, "error")]
            for ax, (img, cmap, vmin, vmax, title) in zip(axes[c], panels):
                im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
                ax.set_title(f"{name} {title}")
                ax.set_axis_off()
                fig.colorbar(im, ax=ax, shrink=0.8)
        fig.suptitle(f"{run}  t={t}")
        return fig

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        if not params:  # e.g. the Downsample baseline: nothing to train
            return None
        return torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
