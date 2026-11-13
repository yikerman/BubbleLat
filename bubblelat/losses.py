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


class BetaVAE(nn.Module):
    """MSE + beta * KL(q(z|x) || N(0, I)) (Higgins et al. 2017), for models that return "mu" and
    "logvar". The KL is in nats per data element (summed over the latent, divided by C*H*W), the
    same normalization as the MSE; in these units beta = 2 is the ELBO of a unit-variance decoder.
    """

    def __init__(self, beta: float = 0.01, weights: list[float] | None = None):
        super().__init__()
        self.beta, self.mse = beta, MSE(weights)

    def forward(self, out: dict, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar = out["mu"], out["logvar"]
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).flatten(1).sum(1).mean() / x[0].numel()
        mse = self.mse(out, x)["loss"]
        return {"loss": mse + self.beta * kl, "mse": mse, "kl": kl}
