"""Readout probes: how well, and how linearly, does a frozen latent predict its frame's physics?

`bubblelat predict` writes one `latents.npz` per model, holding for each split "<split>/z" and
the per-frame labels of `frame_labels` plus run, t, fluid and wall_temp. Every probe fits on
the training runs, picks its hyperparameters on the validation runs and scores on the test runs,
so a probe is held out by run exactly like reconstruction (Alain & Bengio 2016).

`ridge_probe` reads frame-level scalars, which every model keeps at every ratio. `mask_probe`
reads the vapor mask on a fixed 32x32 grid, so a latent cell must hold more sub-cell geometry
as compression grows. `mlp_probe` reads the same scalars as `ridge_probe` nonlinearly: the gap
between the two separates "not in the latent" from "in the latent, but not linearly".

    uv run python -m bubblelat.probe outputs/latents/<id>/latents.npz
"""

import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from .data import SPLITS
from .metrics import mask_iou

ALPHAS = torch.logspace(-12, 0, 25)  # ridge penalty, relative to the largest squared singular value
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAP = 32  # side of the vapor map grid: fixed for every model, so it does not shrink with the latent


def vapor_map(dfun: torch.Tensor) -> torch.Tensor:
    """Vapor fraction of each cell of a MAP x MAP grid, from the signed distance field (B, H, W)."""
    return F.adaptive_avg_pool2d((dfun > 0).float()[:, None], MAP)[:, 0]


def frame_labels(dfun: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-frame labels from the physical signed distance field (B, H, W); vapor is dfun > 0.

    interface_len counts the pixel edges between vapor and liquid; bubble_count is the number of
    4-connected vapor regions; vapor_map is the `vapor_map` (B, MAP, MAP).
    """
    vapor = (dfun > 0).cpu()
    edges = (vapor[:, 1:] != vapor[:, :-1]).sum(dim=(1, 2)) + (vapor[:, :, 1:] != vapor[:, :, :-1]).sum(dim=(1, 2))
    count = torch.tensor([ndimage.label(v)[1] for v in vapor.numpy()])
    return {"vapor_frac": vapor.float().mean(dim=(1, 2)), "interface_len": edges.float(),
            "bubble_count": count.float(), "vapor_map": vapor_map(dfun).cpu()}


def map_iou(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Per-frame IoU of two vapor maps (B, MAP, MAP), each cell counted as vapor above one half."""
    return mask_iou(pred > 0.5, true > 0.5)


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


def ridge_path(data: dict, y: dict[str, torch.Tensor]):
    """Ridge regression from the flattened latent to y[split] (N, K) on DEVICE, fitted on the training
    split for every penalty in ALPHAS. Returns predict(alpha, split) -> (N, K).

    Features are only centered, and the penalty grid is relative to their scale, so the score
    does not depend on the overall scale of a latent.
    """
    z = {s: torch.from_numpy(data[f"{s}/z"]).flatten(1).double().to(DEVICE) for s in SPLITS}
    z_mean, y_mean = z["train"].mean(0), y["train"].mean(0)

    # One SVD of the training features gives the ridge solution for every penalty.
    u, s, vh = torch.linalg.svd(z["train"] - z_mean, full_matrices=False)
    uty = u.T @ (y["train"] - y_mean)
    proj = {split: (z[split] - z_mean) @ vh.T for split in SPLITS}

    def predict(alpha, split):
        return proj[split] @ ((s / (s**2 + alpha * s[0] ** 2))[:, None] * uty) + y_mean

    return predict


def ridge_probe(data: dict) -> dict[str, float]:
    """Test R^2 for each regression target ("probe/<name>") and test accuracy of the fluid
    readout ("probe/fluid_acc"), with the penalty chosen per target on the validation runs."""
    y, names = targets(data)
    y = {s: v.to(DEVICE) for s, v in y.items()}
    predict = ridge_path(data, y)

    def r2(pred, true):  # (K,)
        return 1 - (pred - true).pow(2).sum(0) / (true - true.mean(0)).pow(2).sum(0)

    k = len(names)
    val = torch.stack([r2(predict(a, "val"), y["val"]) for a in ALPHAS])  # (A, K)
    best = val.argmax(0)
    test = torch.stack([predict(ALPHAS[best[j]], "test")[:, j] for j in range(y["test"].shape[1])], dim=1)
    scores = dict(zip((f"probe/{n}" for n in names), r2(test, y["test"])[:k].tolist()))
    hit = test[:, k:].argmax(1) == y["test"][:, k:].argmax(1)
    return scores | {"probe/fluid_acc": hit.double().mean().item()}


def mask_probe(data: dict) -> dict[str, float]:
    """Test IoU of the vapor map read linearly off the latent ("probe/vapor_map_iou"), with one
    penalty for all cells chosen by validation IoU. "probe/vapor_map_iou_decoder" is the same
    score for the model's own reconstruction: the decoder bounds what is in z, the probe asks
    how much of it is linear.
    """
    y = {s: torch.from_numpy(data[f"{s}/vapor_map"]).flatten(1).double().to(DEVICE) for s in SPLITS}
    predict = ridge_path(data, y)

    def iou(alpha, split):
        return map_iou(predict(alpha, split).view(-1, MAP, MAP), y[split].view(-1, MAP, MAP)).mean()

    best = torch.stack([iou(a, "val") for a in ALPHAS]).argmax()
    return {"probe/vapor_map_iou": iou(ALPHAS[best], "test").item(),
            "probe/vapor_map_iou_decoder": float(data["test/vapor_map_iou_decoder"].mean())}


MLP_DECAYS = [1e-4, 1e-3, 1e-2, 1e-1]  # AdamW weight decay, chosen on the validation runs
MLP_STEPS, MLP_EVAL_EVERY = 2000, 50


@torch.inference_mode(False)  # it trains, also when called from Lightning's predict loop
@torch.enable_grad()
def mlp_probe(data: dict, hidden: int = 256) -> dict[str, float]:
    """The `ridge_probe` targets read by a 1-hidden-layer MLP ("probe_mlp/<name>"), one MLP per
    target. Where it beats ridge, the latent holds the quantity but not linearly.

    Inputs are centered and divided by one RMS, so like ridge the score ignores a latent's scale
    but not its geometry. Full-batch AdamW; the weight decay and the stopping step are chosen
    together on the validation runs (early stopping), then scored once on the test runs.
    """
    z = {s: torch.from_numpy(data[f"{s}/z"]).flatten(1).float().to(DEVICE) for s in SPLITS}
    mean, scale = z["train"].mean(0), z["train"].var(0).mean().sqrt()
    z = {s: (v - mean) / scale for s, v in z.items()}
    y, names = targets(data)
    y = {s: v.float().to(DEVICE) for s, v in y.items()}
    k = len(names)

    def r2(pred, true):
        return 1 - (pred - true).pow(2).sum() / (true - true.mean()).pow(2).sum()

    def acc(logits, onehot):
        return (logits.argmax(1) == onehot.argmax(1)).float().mean()

    # (output columns, loss, score) per readout: each regression target, then the fluid.
    readouts = [(slice(j, j + 1), F.mse_loss, r2) for j in range(k)]
    readouts.append((slice(k, None), lambda p, t: F.cross_entropy(p, t.argmax(1)), acc))

    scores = []
    for cols, loss_fn, score_fn in readouts:
        target = {s: v[:, cols] for s, v in y.items()}
        t_mean, t_std = target["train"].mean(0), target["train"].std(0)
        if loss_fn is F.mse_loss:  # regress standardized targets; R^2 is invariant to it
            target = {s: (v - t_mean) / t_std for s, v in target.items()}
        best_val, best_test = -float("inf"), float("nan")
        for decay in MLP_DECAYS:
            torch.manual_seed(0)
            net = torch.nn.Sequential(torch.nn.Linear(z["train"].shape[1], hidden), torch.nn.GELU(),
                                      torch.nn.Linear(hidden, target["train"].shape[1])).to(DEVICE)
            # Zero output layer: step 0 is the constant prediction (R^2 about 0), so early
            # stopping falls back to it when every trained checkpoint overfits.
            torch.nn.init.zeros_(net[-1].weight)
            torch.nn.init.zeros_(net[-1].bias)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=decay)
            for step in range(MLP_STEPS + 1):
                if step % MLP_EVAL_EVERY == 0:
                    with torch.no_grad():
                        val = score_fn(net(z["val"]), target["val"]).item()
                        if val > best_val:
                            best_val = val
                            best_test = score_fn(net(z["test"]), target["test"]).item()
                opt.zero_grad()
                loss_fn(net(z["train"]), target["train"]).backward()
                opt.step()
        scores.append(best_test)
    return dict(zip([f"probe_mlp/{n}" for n in names] + ["probe_mlp/fluid_acc"], scores))


def probe(data: dict) -> dict[str, float]:
    return ridge_probe(data) | mask_probe(data) | mlp_probe(data)


if __name__ == "__main__":
    for path in sys.argv[1:]:
        print(path)
        for name, score in probe(dict(np.load(path))).items():
            print(f"  {name:28s} {score: .4f}")
