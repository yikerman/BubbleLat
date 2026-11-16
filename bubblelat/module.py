"""Lightning module that trains and evaluates any `Autoencoder` with any loss.

wandb layout:
  train/loss                      every step
  lr-AdamW                        learning rate (cosine decay to zero over the run)
  val/<metric>/<field>            every validation epoch, in physical units
  val/vapor/{iou,frac_err}        vapor mask (dfun > 0) agreement
  test/<metric>/<field>           once, plus a "test/table" with one row per field
  spectrum.<field>.{target,error} test, run summary: mean power per wavenumber shell
  <stage>/samples                 target | reconstruction | error for one fixed frame
  probe/<target>                  `predict`: ridge readout scores of the latent (see probe.py)

`predict` also writes the latents and their labels to <trainer.default_root_dir>/latents.npz.
"""

from collections import defaultdict
from pathlib import Path

import lightning as L
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger
from torch import nn

from .data import SPLITS, denormalize
from .metrics import METRICS, field_errors, radial_power, vapor_errors
from .models import Autoencoder
from .probe import frame_labels, ridge_probe


class AutoencoderModule(L.LightningModule):
    def __init__(self, model: Autoencoder, loss: nn.Module | None = None, lr: float = 1e-3,
                 weight_decay: float = 0.0):
        super().__init__()
        self.model, self.loss = model, loss
        self.lr, self.weight_decay = lr, weight_decay

    def setup(self, stage: str) -> None:
        dm = self.trainer.datamodule
        self.model.init_from_data(dm.train_frames)
        x = dm.train_frames(1)
        info = {"model/params": sum(p.numel() for p in self.model.parameters()),
                "model/compression": x.numel() / self.model.encode(x).numel(), **dm.describe()}
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
        self._power = 0.0  # (2, C, K): summed target and error power per wavenumber shell

    on_test_epoch_start = on_validation_epoch_start

    def _eval_step(self, batch, batch_idx):
        dm, x = self.trainer.datamodule, batch["x"]
        out = self.model(x)
        pred, target = denormalize(out["xhat"].float(), batch), denormalize(x.float(), batch)
        metrics = field_errors(pred, target, dm.fields)
        if "dfun" in dm.fields:
            i = dm.fields.index("dfun")
            metrics |= vapor_errors(pred[:, i], target[:, i])
        if self.loss is not None:
            metrics |= self.loss(out, x)
        for k, v in metrics.items():
            self._sums[k] += v.item() * len(x)
        self._power += torch.stack([radial_power(target), radial_power(pred - target)]).sum(1).cpu()
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
                power = (self._power / self._count).tolist()
                for c, f in enumerate(fields):  # wandb stores nested summaries as dotted keys anyway
                    self.logger.experiment.summary[f"spectrum.{f}.target"] = power[0][c]
                    self.logger.experiment.summary[f"spectrum.{f}.error"] = power[1][c]
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

    # predict: dump the latent and labels of every strided frame, then probe them

    def on_predict_epoch_start(self):
        self._dump = defaultdict(list)

    def predict_step(self, batch, _, dataloader_idx: int = 0):
        dm, x, split = self.trainer.datamodule, batch["x"], SPLITS[dataloader_idx]
        dfun = denormalize(x.float(), batch)[:, dm.fields.index("dfun")]
        rows = {"z": self.model.encode(x).float().cpu(), "t": batch["t"].cpu(),
                "wall_temp": batch["wall_temp"].cpu(), "run": batch["run"], "fluid": batch["fluid"],
                **frame_labels(dfun)}
        for k, v in rows.items():
            self._dump[f"{split}/{k}"].append(np.asarray(v))

    def on_predict_epoch_end(self):
        data = {k: np.concatenate(v) for k, v in self._dump.items()}
        path = Path(self.trainer.default_root_dir) / "latents.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **data)
        scores = ridge_probe(data)
        print(f"latents: {path}\n" + "\n".join(f"{k:24s} {v: .4f}" for k, v in scores.items()))
        if self.logger:
            self.logger.log_metrics(scores)

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        if not params:  # e.g. the Downsample baseline: nothing to train
            return None
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        # Cosine decay to zero over the whole run. At a constant rate the validation loss jumped
        # severalfold between late epochs, so a run's score depended on where it stopped.
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.trainer.estimated_stepping_batches)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
