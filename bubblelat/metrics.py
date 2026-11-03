"""Evaluation metrics on physical-unit tensors of shape (B, C, H, W)."""

import torch

METRICS = ["mae", "rmse", "vrmse"]


def field_errors(pred: torch.Tensor, target: torch.Tensor, fields: list[str]) -> dict:
    """Batch-mean MAE, RMSE and VRMSE for each field, keyed "<metric>/<field>".

    VRMSE is the RMSE divided by the target frame's spatial std (The Well, Ohana et al. 2024):
    1.0 means "no better than predicting the frame's mean".
    """
    err = pred - target
    mse = err.pow(2).mean(dim=(2, 3))  # (B, C)
    var = target.var(dim=(2, 3), unbiased=False)
    out = {}
    for c, name in enumerate(fields):
        out[f"mae/{name}"] = err[:, c].abs().mean()
        out[f"rmse/{name}"] = mse[:, c].sqrt().mean()
        out[f"vrmse/{name}"] = (mse[:, c] / (var[:, c] + 1e-7)).sqrt().mean()
    return out
