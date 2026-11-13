"""Autoencoders. Every model maps x (B, C, H, W) to a latent z and back to a reconstruction.

To add a model, subclass `Autoencoder` and implement `encode` and `decode`. Anything
extra a loss needs (for a VAE, mu and logvar) goes in the dict that `forward` returns.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn


class Autoencoder(nn.Module):
    def init_from_data(self, train_frames: Callable[[int], torch.Tensor]) -> None:
        """Closed-form initialization, called once before any other use. `train_frames(n)`
        returns n evenly spaced normalized training frames (n, C, H, W). Nothing by default."""

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encode(x)
        return {"z": z, "xhat": self.decode(z)}


class Downsample(Autoencoder):
    """Baseline with no parameters: average-pool by `factor`, then nearest-neighbor upsample."""

    def __init__(self, factor: int = 4):
        super().__init__()
        self.factor = factor

    def encode(self, x):
        return F.avg_pool2d(x, self.factor)

    def decode(self, z):
        return F.interpolate(z, scale_factor=self.factor, mode="nearest")


class POD(Autoencoder):
    """Proper orthogonal decomposition (PCA): project onto the top `rank` modes of `snapshots`
    training frames. The optimal linear autoencoder for MSE (Baldi & Hornik 1989), so it is the
    floor any learned model of the same latent size has to beat. Fitted in closed form, so run
    `test`, not `fit`.
    """

    def __init__(self, rank: int = 1024, snapshots: int = 8192):
        super().__init__()
        self.rank, self.snapshots = rank, snapshots
        self.register_buffer("mean", torch.empty(0))   # (D,)
        self.register_buffer("modes", torch.empty(0))  # (rank, D), orthonormal rows

    def init_from_data(self, train_frames):
        x = train_frames(self.snapshots)
        assert len(x) > self.rank, f"POD rank {self.rank} needs more than {len(x)} snapshots"
        self.shape = x.shape[1:]
        x = x.flatten(1)
        self.mean = x.mean(0)
        self.modes = torch.linalg.svd(x - self.mean, full_matrices=False).Vh[:self.rank]

    def encode(self, x):
        return (x.flatten(1) - self.mean) @ self.modes.T

    def decode(self, z):
        return (z @ self.modes + self.mean).unflatten(1, self.shape)


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.GroupNorm(8, c_out), nn.GELU(),
                         nn.Conv2d(c_out, c_out, 3, padding=1), nn.GroupNorm(8, c_out), nn.GELU())


class ConvAE(Autoencoder):
    """Plain CNN autoencoder. Each entry of `channels` halves the resolution once."""

    def __init__(self, in_channels: int = 4, channels: tuple[int, ...] = (32, 64, 128),
                 latent_channels: int = 4):
        super().__init__()
        channels = list(channels)
        enc, c = [], in_channels
        for c_out in channels:
            enc += [conv_block(c, c_out), nn.AvgPool2d(2)]
            c = c_out
        self.encoder = nn.Sequential(*enc, nn.Conv2d(c, latent_channels, 1))

        dec = [nn.Conv2d(latent_channels, channels[-1], 1)]
        for c_in, c_out in zip(channels[::-1], channels[::-1][1:] + [channels[0]]):
            dec += [nn.Upsample(scale_factor=2, mode="nearest"), conv_block(c_in, c_out)]
        self.decoder = nn.Sequential(*dec, nn.Conv2d(channels[0], in_channels, 1))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


class ConvVAE(ConvAE):
    """ConvAE whose encoder outputs a diagonal Gaussian q(z|x) per latent element. Same backbone
    and latent shape as ConvAE, so the two differ only in the loss (`BetaVAE`).

    The latent is the mean: `encode` returns it, and outside training the reconstruction decodes
    it, so metrics and probes see the same deterministic code. Training decodes a sample.
    """

    def __init__(self, in_channels: int = 4, channels: tuple[int, ...] = (32, 64, 128),
                 latent_channels: int = 4):
        super().__init__(in_channels, channels, latent_channels)
        self.encoder[-1] = nn.Conv2d(list(channels)[-1], 2 * latent_channels, 1)

    def moments(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x).chunk(2, dim=1)
        return mu, logvar

    def encode(self, x):
        return self.moments(x)[0]

    def forward(self, x):
        mu, logvar = self.moments(x)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if self.training else mu
        return {"z": z, "xhat": self.decode(z), "mu": mu, "logvar": logvar}
