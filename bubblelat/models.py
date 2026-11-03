"""Autoencoders. Every model maps x (B, C, H, W) to a latent z and back to a reconstruction.

To add a model, subclass `Autoencoder` and implement `encode` and `decode`. Anything
extra a loss needs (for a VAE, mu and logvar) goes in the dict that `forward` returns.
"""

import torch
import torch.nn.functional as F
from torch import nn


class Autoencoder(nn.Module):
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
