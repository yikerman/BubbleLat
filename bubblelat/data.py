"""BubbleML data. One sample is one frame (C, H, W) of one simulation run.

Reads the cache written by `bubblelat.cache`: each run is a `<group>/<stem>.npy` array of
shape (T, C, H, W), float32 and already at training resolution, plus a `<stem>.json` holding
its metadata, field names and simulation frame indices "t". A run's id is "<group>/<stem>",
e.g. "PoolBoiling-Subcooled-FC72-2D/Twall_100". Splits are made by run, never by frame,
because neighboring frames are nearly identical.
"""

import json
from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

FIELDS = ["dfun", "temperature", "velx", "vely"]


class Run:
    """One cached simulation run and its metadata."""

    def __init__(self, path: Path):
        self.path = path
        self.id = f"{path.parent.name}/{path.stem}"
        self.meta = json.loads(path.with_suffix(".json").read_text())
        self.fluid = self.meta.get("liquid", "unknown")
        self.wall_temp = float(self.meta["heater"]["wallTemp"])
        self.fields: list[str] = self.meta["fields"]
        self.t: list[int] = self.meta["t"]  # simulation frame index of each stored frame


def find_runs(roots: list[str]) -> list[Run]:
    """All runs (<group>/<stem>.npy) under the given cache roots."""
    paths = sorted(p for d in roots for p in Path(d).expanduser().glob("*/*.npy"))
    if not paths:
        raise FileNotFoundError(f"no <group>/<stem>.npy runs under {roots}")
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
    """Every stored frame with simulation index t >= `burn_in` of the given runs."""

    def __init__(self, runs: list[Run], fields: list[str], burn_in: int,
                 mean: torch.Tensor | None = None, std: torch.Tensor | None = None):
        self.runs, self.mean, self.std = runs, mean, std
        self.channels = [[run.fields.index(f) for f in fields] for run in runs]
        self.index = [(r, i) for r, run in enumerate(runs)
                      for i, t in enumerate(run.t) if t >= burn_in]
        self.arrays: dict[int, np.ndarray] = {}  # memory-mapped lazily, once per worker

    def __len__(self) -> int:
        return len(self.index)

    def read(self, r: int, i: int) -> torch.Tensor:
        if r not in self.arrays:
            self.arrays[r] = np.load(self.runs[r].path, mmap_mode="r")
        return torch.from_numpy(np.array(self.arrays[r][i, self.channels[r]]))  # (C, H, W)

    def __getitem__(self, k: int) -> dict:
        r, i = self.index[k]
        run = self.runs[r]
        t = run.t[i]
        x = self.read(r, i)
        if self.mean is not None:
            x = (x - self.mean[:, None, None]) / self.std[:, None, None]
        return {"x": x, "run": run.id, "t": t, "fluid": run.fluid, "wall_temp": run.wall_temp}


class BubbleMLData(L.LightningDataModule):
    """Splits runs into train/val/test and standardizes each field with training statistics.

    Args:
        data_dirs: cache roots written by `bubblelat.cache` (each holds <group>/<stem>.npy).
        holdout_every: see `split_runs`. With 20 runs per directory, 5 gives 12/4/4.
        burn_in: skip frames with simulation index below this (the initial transient).
    """

    def __init__(self, data_dirs: list[str], holdout_every: int = 5, fields: list[str] = FIELDS,
                 burn_in: int = 100, batch_size: int = 16, num_workers: int = 4,
                 stats_frames: int = 64):
        super().__init__()
        self.data_dirs, self.fields = data_dirs, fields
        self.holdout_every = holdout_every
        self.burn_in = burn_in
        self.batch_size, self.num_workers = batch_size, num_workers
        self.stats_frames = stats_frames

    def setup(self, stage: str | None = None) -> None:
        if hasattr(self, "mean"):
            return
        self.splits = split_runs(find_runs(self.data_dirs), self.holdout_every)
        self.mean, self.std = self._stats(self.splits["train"])

    def _stats(self, runs: list[Run]) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-field mean and std over evenly spaced training frames."""
        raw = FrameDataset(runs, self.fields, self.burn_in)
        n = min(len(raw), self.stats_frames * len(runs))
        x = torch.stack([raw[i]["x"] for i in np.linspace(0, len(raw) - 1, n).astype(int)])
        return x.mean(dim=(0, 2, 3)), x.std(dim=(0, 2, 3))

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std.to(x)[:, None, None] + self.mean.to(x)[:, None, None]

    def describe(self) -> dict:
        """Split summary, logged to wandb so every run records exactly what it saw."""
        return {f"data/{s}_runs": [r.id for r in runs] for s, runs in self.splits.items()}

    def _loader(self, split: str, shuffle: bool = False) -> DataLoader:
        ds = FrameDataset(self.splits[split], self.fields, self.burn_in, self.mean, self.std)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=self.num_workers, persistent_workers=self.num_workers > 0)

    def train_dataloader(self):
        return self._loader("train", shuffle=True)

    def val_dataloader(self):
        return self._loader("val")

    def test_dataloader(self):
        return self._loader("test")
