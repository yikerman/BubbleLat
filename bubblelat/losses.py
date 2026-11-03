"""Training losses. A loss takes the model's output dict and the normalized target and
returns a dict of scalars: "loss" is what gets optimized, and any other entries are only logged.
"""

import torch
from torch import nn


class MSE(nn.Module):
    """Mean squared error on normalized fields, with optional per-field weights."""

    def __init__(self, weights: list[float] | None = None):
        super().__init__()
        self.weights = weights

    def forward(self, out: dict, x: torch.Tensor) -> dict[str, torch.Tensor]:
        per_field = (out["xhat"] - x).pow(2).mean(dim=(0, 2, 3))  # (C,)
        if self.weights is None:
            return {"loss": per_field.mean()}
        w = per_field.new_tensor(self.weights)
        return {"loss": (w * per_field).sum() / w.sum()}
