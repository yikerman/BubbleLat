"""Build the training cache: one float32 array of shape (T, C, H, W) per simulation run.

    <out>/<group>/<stem>.npy   frames, average-pooled by `downsample`
    <out>/<group>/<stem>.json  the run's metadata, plus "fields" and "t" (simulation frame index)

Sources:
    wds   pre-shuffled BubbleML_2 WebDataset shards (32-frame, 8-field, 512x512 clips), streamed
          once in order. Fast on shared storage.
    hdf5  <group>/<stem>.hdf5 + .json pairs, e.g. the local sample files.

    uv run python -m bubblelat.cache wds '/share/.../shard-{000000..000724}.tar' --out ... --downsample 4
    uv run python -m bubblelat.cache hdf5 ../BubbleGen/samples --out data/samples
"""

import argparse
import io
import json
from collections import defaultdict
from functools import partial
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

FIELDS = ["dfun", "temperature", "velx", "vely"]


def pool(clip: np.ndarray, downsample: int) -> np.ndarray:
    """(T, C, H, W) -> (T, C, H/d, W/d) by average pooling, in float32."""
    x = torch.from_numpy(np.ascontiguousarray(clip, dtype=np.float32))
    return (F.avg_pool2d(x, downsample) if downsample > 1 else x).numpy()


def decode(sample: dict, downsample: int) -> tuple:
    """One WebDataset sample -> (group, stem, t_start, frames (T, C, h, w), sidecar)."""
    import zstandard

    arr = np.load(io.BytesIO(zstandard.ZstdDecompressor().decompress(sample["npy.zst"])))
    meta = json.loads(sample["json"])
    idx = [meta["fields"].index(f) for f in FIELDS]
    frames = pool(arr[idx].transpose(1, 0, 2, 3), downsample)  # (C, T, H, W) -> (T, C, h, w)
    return meta["group"], meta["stem"], meta["t_start"], frames, meta["sidecar"]


def wds_clips(shards: str, downsample: int, workers: int):
    """Decode WebDataset shards in parallel; each worker streams whole shards in order."""
    import webdataset as wds

    ds = wds.WebDataset(shards, shardshuffle=False, empty_check=False)
    ds = ds.map(partial(decode, downsample=downsample))
    loader = DataLoader(ds, batch_size=None, num_workers=workers, prefetch_factor=1)
    yield from loader


def hdf5_clips(root: str, downsample: int):
    """Yield one whole-run "clip" per .hdf5 file under root."""
    for path in sorted(Path(root).expanduser().rglob("*.hdf5")):
        with h5py.File(path, "r") as f:
            arr = np.stack([f[k][:] for k in FIELDS], axis=1)  # (T, C, H, W)
        meta = json.loads(path.with_suffix(".json").read_text())
        yield path.parent.name, path.stem, 0, pool(arr, downsample), meta


def write_runs(clips, out: Path) -> None:
    """Gather clips per run (they arrive in any order), then write one array per run."""
    runs: dict[tuple[str, str], dict[int, np.ndarray]] = defaultdict(dict)
    meta: dict[tuple[str, str], dict] = {}
    for n, (group, stem, t_start, frames, sidecar) in enumerate(clips, 1):
        frames = frames.numpy() if isinstance(frames, torch.Tensor) else frames
        for i, frame in enumerate(frames):
            runs[group, stem][int(t_start) + i] = frame
        meta[group, stem] = sidecar
        if n % 100 == 0:
            print(f"{n} clips, {len(runs)} runs", flush=True)

    for (group, stem), frames in sorted(runs.items()):
        t = sorted(frames)
        (out / group).mkdir(parents=True, exist_ok=True)
        np.save(out / group / f"{stem}.npy", np.stack([frames[i] for i in t]))
        info = {**meta[group, stem], "fields": FIELDS, "t": t}
        (out / group / f"{stem}.json").write_text(json.dumps(info))
        print(f"{group}/{stem}: {len(t)} frames, t={t[0]}..{t[-1]}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("source", choices=["wds", "hdf5"])
    p.add_argument("input", help="wds: brace-glob of shard paths; hdf5: root directory")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--downsample", type=int, default=1)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    if a.source == "wds":
        clips = wds_clips(a.input, a.downsample, a.workers)
    else:
        clips = hdf5_clips(a.input, a.downsample)
    write_runs(clips, a.out)


if __name__ == "__main__":
    main()
