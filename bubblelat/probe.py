"""Readout probes: how linearly can a frozen latent predict physical quantities of its frame?

`bubblelat predict` writes one `latents.npz` per model, holding for each split "<split>/z" and
the per-frame labels of `frame_labels` plus run, t, fluid and wall_temp. `ridge_probe` fits
ridge regression on the training runs, picks the penalty on the validation runs and scores on
the test runs, so a probe is held out by run exactly like reconstruction (Alain & Bengio 2016).

    uv run python -m bubblelat.probe outputs/latents/<id>/latents.npz
"""

import sys

import numpy as np
import torch
from scipy import ndimage

from .data import SPLITS

ALPHAS = torch.logspace(-12, 0, 25)  # ridge penalty, relative to the largest squared singular value


def frame_labels(dfun: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-frame labels from the physical signed distance field (B, H, W); vapor is dfun > 0.

    interface_len counts the pixel edges between vapor and liquid; bubble_count is the number of
    4-connected vapor regions.
    """
    vapor = (dfun > 0).cpu()
    edges = (vapor[:, 1:] != vapor[:, :-1]).sum(dim=(1, 2)) + (vapor[:, :, 1:] != vapor[:, :, :-1]).sum(dim=(1, 2))
    count = torch.tensor([ndimage.label(v)[1] for v in vapor.numpy()])
    return {"vapor_frac": vapor.float().mean(dim=(1, 2)), "interface_len": edges.float(),
            "bubble_count": count.float()}


def targets(data: dict) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Per split, a (N, K) target matrix, and the names of its regression columns. The remaining
    columns are the one-hot fluid.

    wall_temp is standardized within its run directory (fluid and regime) with training
    statistics: across directories it would only measure which fluid a frame shows.
    Neither wall_temp nor the fluid is visible in degrees: `Run.units` maps every run's
    temperature to 0..1, so both readouts work only through the flow (bubble size and shape).
    """
    group = {s: np.array([r.split("/")[0] for r in data[f"{s}/run"]]) for s in SPLITS}
    wall = {s: torch.from_numpy(data[f"{s}/wall_temp"]).double() for s in SPLITS}
    for g in np.unique(group["train"]):
        ref = wall["train"][torch.from_numpy(group["train"] == g)]
        mean, std = ref.mean(), ref.std()
        for s in SPLITS:
            m = torch.from_numpy(group[s] == g)
            wall[s][m] = (wall[s][m] - mean) / std
    names = ["vapor_frac", "interface_len", "bubble_count", "wall_temp"]
    fluids = sorted(set(data["train/fluid"]))
    out = {}
    for s in SPLITS:
        cols = [torch.from_numpy(data[f"{s}/{n}"]).double() for n in names[:-1]] + [wall[s]]
        onehot = [torch.from_numpy(data[f"{s}/fluid"] == f).double() for f in fluids]
        out[s] = torch.stack(cols + onehot, dim=1)
    return out, names


def ridge_probe(data: dict) -> dict[str, float]:
    """Test R^2 for each regression target ("probe/<name>") and test accuracy of the fluid
    readout ("probe/fluid_acc"), with the penalty chosen per target on the validation runs.

    Features are only centered, and the penalty grid is relative to their scale, so the score
    does not depend on the overall scale of a latent.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z = {s: torch.from_numpy(data[f"{s}/z"]).flatten(1).double().to(device) for s in SPLITS}
    y, names = targets(data)
    y = {s: v.to(device) for s, v in y.items()}
    z_mean, y_mean = z["train"].mean(0), y["train"].mean(0)

    # One SVD of the training features gives the ridge solution for every penalty.
    u, s, vh = torch.linalg.svd(z["train"] - z_mean, full_matrices=False)
    uty = u.T @ (y["train"] - y_mean)

    def predict(alpha, split):  # (N, K)
        w = vh.T @ ((s / (s**2 + alpha * s[0] ** 2))[:, None] * uty)
        return (z[split] - z_mean) @ w + y_mean

    def r2(pred, true):  # (K,)
        return 1 - (pred - true).pow(2).sum(0) / (true - true.mean(0)).pow(2).sum(0)

    k = len(names)
    val = torch.stack([r2(predict(a, "val"), y["val"]) for a in ALPHAS])  # (A, K)
    best = val.argmax(0)
    test = torch.stack([predict(ALPHAS[best[j]], "test")[:, j] for j in range(y["test"].shape[1])], dim=1)
    scores = dict(zip((f"probe/{n}" for n in names), r2(test, y["test"])[:k].tolist()))
    hit = test[:, k:].argmax(1) == y["test"][:, k:].argmax(1)
    return scores | {"probe/fluid_acc": hit.double().mean().item()}


if __name__ == "__main__":
    for path in sys.argv[1:]:
        print(path)
        for name, score in ridge_probe(dict(np.load(path))).items():
            print(f"  {name:24s} {score: .4f}")
