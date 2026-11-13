"""BubbleML data. One sample is one frame (C, H, W) of one simulation run.

Reads the cache written by `bubblelat.cache`: each run is a `<group>/<stem>.npy` array of
shape (T, C, H, W), float32 and already at training resolution, plus a `<stem>.json` holding
its metadata, field names and simulation frame indices "t". A run's id is "<group>/<stem>",
e.g. "PoolBoiling-Subcooled-FC72-2D/Twall_100". Splits are made by run, never by frame,
because neighboring frames are nearly identical.

Normalization is one affine map per run and field, x_norm = (x - shift) / scale. It composes
the run's own units (`Run.units`: temperature becomes (T - T_bulk) / (T_wall - T_bulk), which is
0..1 for every fluid) with the training mean and std of the resulting fields. Every sample
carries its "shift" and "scale", and `denormalize` inverts the map back to physical units.
"""

import json
from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

FIELDS = ["dfun", "temperature", "velx", "vely"]
SPLITS = ["train", "val", "test"]


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

    def units(self, fields: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-field (shift, scale) that make this run's fields comparable across runs.

        Temperature is stored in degrees, and the fluids sit hundreds of degrees apart while a
        frame varies by T_wall - T_bulk, so it is rescaled to that range. The other fields are
        already non-dimensional (identity).
        """
        bulk = float(self.meta["bulk_temp"])
        shift = torch.tensor([bulk if f == "temperature" else 0.0 for f in fields])
        scale = torch.tensor([self.wall_temp - bulk if f == "temperature" else 1.0 for f in fields])
        return shift, scale


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
    """Every `stride`-th stored frame with simulation index t >= `burn_in` of the given runs."""

    def __init__(self, runs: list[Run], fields: list[str], burn_in: int,
                 mean: torch.Tensor | float = 0.0, std: torch.Tensor | float = 1.0, stride: int = 1):
        self.runs = runs
        units = [run.units(fields) for run in runs]
        self.shift = [shift + scale * mean for shift, scale in units]  # per run, (C,)
        self.scale = [scale * std for _, scale in units]
        self.channels = [[run.fields.index(f) for f in fields] for run in runs]
        self.index = [(r, i) for r, run in enumerate(runs)
                      for i, t in enumerate(run.t) if t >= burn_in][::stride]
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
        shift, scale = self.shift[r], self.scale[r]
        x = (self.read(r, i) - shift[:, None, None]) / scale[:, None, None]
        return {"x": x, "shift": shift, "scale": scale, "run": run.id, "t": t,
                "fluid": run.fluid, "wall_temp": run.wall_temp}


def denormalize(x: torch.Tensor, batch: dict) -> torch.Tensor:
    """Normalized (B, C, H, W) fields back to physical units, with the batch's own maps."""
    return x * batch["scale"][:, :, None, None].to(x) + batch["shift"][:, :, None, None].to(x)


class BubbleMLData(L.LightningDataModule):
    """Splits runs into train/val/test and standardizes each field with training statistics
    (taken after each run's own `units`).

    Args:
        data_dirs: cache roots written by `bubblelat.cache` (each holds <group>/<stem>.npy).
        holdout_every: see `split_runs`. With 20 runs per directory, 5 gives 12/4/4.
        burn_in: skip frames with simulation index below this (the initial transient).
        predict_stride: `predict` dumps the latent of every n-th frame of each split
            (neighboring frames are nearly identical).
    """

    def __init__(self, data_dirs: list[str], holdout_every: int = 5, fields: list[str] = FIELDS,
                 burn_in: int = 100, batch_size: int = 16, num_workers: int = 4,
                 stats_frames: int = 64, predict_stride: int = 8):
        super().__init__()
        self.predict_stride = predict_stride
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

    @staticmethod
    def _spaced(ds: FrameDataset, n: int) -> torch.Tensor:
        """At most n evenly spaced frames of a dataset, (n, C, H, W)."""
        n = min(n, len(ds))
        return torch.stack([ds[i]["x"] for i in np.linspace(0, len(ds) - 1, n).astype(int)])

    def _stats(self, runs: list[Run]) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-field mean and std over evenly spaced training frames, in run units."""
        x = self._spaced(FrameDataset(runs, self.fields, self.burn_in), self.stats_frames * len(runs))
        return x.mean(dim=(0, 2, 3)), x.std(dim=(0, 2, 3))

    def train_frames(self, n: int) -> torch.Tensor:
        """At most n evenly spaced normalized training frames, for closed-form model fits."""
        train = FrameDataset(self.splits["train"], self.fields, self.burn_in, self.mean, self.std)
        return self._spaced(train, n)

    def describe(self) -> dict:
        """Split summary, logged to wandb so every run records exactly what it saw."""
        return {f"data/{s}_runs": [r.id for r in runs] for s, runs in self.splits.items()}

    def _loader(self, split: str, shuffle: bool = False, stride: int = 1) -> DataLoader:
        ds = FrameDataset(self.splits[split], self.fields, self.burn_in, self.mean, self.std, stride)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=self.num_workers, persistent_workers=self.num_workers > 0)

    def train_dataloader(self):
        return self._loader("train", shuffle=True)

    def val_dataloader(self):
        return self._loader("val")

    def test_dataloader(self):
        return self._loader("test")

    def predict_dataloader(self):
        return [self._loader(s, stride=self.predict_stride) for s in SPLITS]
