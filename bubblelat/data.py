"""BubbleML data. One sample is one frame (C, H, W) of one simulation run.

Each run is a `<name>.hdf5` file holding one (T, H, W) dataset per field, plus a
`<name>.json` with its metadata. A run's id is "<parent dir>/<name>", for example
"PoolBoiling-Subcooled-FC72-2D/Twall-100". Splits are made by run, never by frame,
because neighboring frames are nearly identical.
"""

import json
from pathlib import Path

import h5py
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

FIELDS = ["dfun", "temperature", "velx", "vely"]


class Run:
    """One simulation file and its metadata."""

    def __init__(self, path: Path):
        self.path = path
        self.id = f"{path.parent.name}/{path.stem}"
        self.meta = json.loads(path.with_suffix(".json").read_text())
        self.fluid = self.meta.get("liquid", "unknown")
        self.wall_temp = float(self.meta["heater"]["wallTemp"])
        with h5py.File(path, "r") as f:
            self.n_frames = f["dfun"].shape[0]


def find_runs(data_dirs: list[str]) -> list[Run]:
    paths = sorted(p for d in data_dirs for p in Path(d).expanduser().glob("*.hdf5"))
    if not paths:
        raise FileNotFoundError(f"no .hdf5 files in {data_dirs}")
    return [Run(p) for p in paths]


def split_runs(runs: list[Run], every: int) -> dict[str, list[Run]]:
    """Fixed split by wall temperature, separately for each directory (fluid and regime).

    Runs are sorted by wall temperature; position i goes to val if i % every == 1, to test
    if i % every == 3, and to train otherwise. Held-out runs are spread over the temperature
    range, and the coldest and hottest runs are always in train, so val and test measure
    interpolation between training wall temperatures.
    """
    splits = {"train": [], "val": [], "test": []}
    for d in sorted({r.path.parent.name for r in runs}):
        group = sorted((r for r in runs if r.path.parent.name == d), key=lambda r: r.wall_temp)
        for i, run in enumerate(group):
            split = {1: "val", 3: "test"}.get(i % every, "train")
            splits[split].append(run)
    return splits


class FrameDataset(Dataset):
    """All frames after `burn_in` of the given runs, average-pooled by `downsample`."""

    def __init__(self, runs: list[Run], fields: list[str], burn_in: int, downsample: int,
                 mean: torch.Tensor | None = None, std: torch.Tensor | None = None):
        self.runs, self.fields, self.downsample = runs, fields, downsample
        self.mean, self.std = mean, std
        self.index = [(r, t) for r, run in enumerate(runs) for t in range(burn_in, run.n_frames)]
        self.files: dict[int, h5py.File] = {}  # opened lazily, so each worker has its own handles

    def __len__(self) -> int:
        return len(self.index)

    def read(self, r: int, t: int) -> torch.Tensor:
        if r not in self.files:
            self.files[r] = h5py.File(self.runs[r].path, "r")
        f = self.files[r]
        x = torch.from_numpy(np.stack([f[k][t] for k in self.fields]).astype(np.float32))
        if self.downsample > 1:
            x = F.avg_pool2d(x[None], self.downsample)[0]
        return x  # (C, H, W), physical units

    def __getitem__(self, i: int) -> dict:
        r, t = self.index[i]
        run = self.runs[r]
        x = self.read(r, t)
        if self.mean is not None:
            x = (x - self.mean[:, None, None]) / self.std[:, None, None]
        return {"x": x, "run": run.id, "t": t, "fluid": run.fluid, "wall_temp": run.wall_temp}


class BubbleMLData(L.LightningDataModule):
    """Splits runs into train/val/test and standardizes each field with training statistics.

    Args:
        data_dirs: directories holding the .hdf5/.json pairs.
        holdout_every: see `split_runs`. With 20 runs per directory, 5 gives 12/4/4.
        downsample: integer factor of spatial average pooling, applied when reading.
    """

    def __init__(self, data_dirs: list[str], holdout_every: int = 5, fields: list[str] = FIELDS, burn_in: int = 100, downsample: int = 1,
                 batch_size: int = 16, num_workers: int = 4, stats_frames: int = 64):
        super().__init__()
        self.data_dirs, self.fields = data_dirs, fields
        self.holdout_every = holdout_every
        self.burn_in, self.downsample = burn_in, downsample
        self.batch_size, self.num_workers = batch_size, num_workers
        self.stats_frames = stats_frames

    def setup(self, stage: str | None = None) -> None:
        if hasattr(self, "mean"):
            return
        self.splits = split_runs(find_runs(self.data_dirs), self.holdout_every)
        self.mean, self.std = self._stats(self.splits["train"])

    def _stats(self, runs: list[Run]) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-field mean and std over evenly spaced training frames."""
        raw = FrameDataset(runs, self.fields, self.burn_in, self.downsample)
        n = min(len(raw), self.stats_frames * len(runs))
        x = torch.stack([raw[i]["x"] for i in np.linspace(0, len(raw) - 1, n).astype(int)])
        return x.mean(dim=(0, 2, 3)), x.std(dim=(0, 2, 3))

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std.to(x)[:, None, None] + self.mean.to(x)[:, None, None]

    def describe(self) -> dict:
        """Split summary, logged to wandb so every run records exactly what it saw."""
        return {f"data/{s}_runs": [r.id for r in runs] for s, runs in self.splits.items()}

    def _loader(self, split: str, shuffle: bool = False) -> DataLoader:
        ds = FrameDataset(self.splits[split], self.fields, self.burn_in, self.downsample,
                          self.mean, self.std)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=self.num_workers, persistent_workers=self.num_workers > 0)

    def train_dataloader(self):
        return self._loader("train", shuffle=True)

    def val_dataloader(self):
        return self._loader("val")

    def test_dataloader(self):
        return self._loader("test")
