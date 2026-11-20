"""Autoencoders. Every model maps x (B, C, H, W) to a latent z and back to a reconstruction.

To add a model, subclass `Autoencoder` and implement `encode` and `decode`. Anything
extra a loss needs (for a VAE, mu and logvar) goes in the dict that `forward` returns.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from timm.layers.pos_embed_sincos import build_sincos2d_pos_embed
from timm.models.vision_transformer import Block
from torch import nn


class Autoencoder(nn.Module):
    # Values the encoder head emits per latent element: 1 for a point code, 2 for GaussianLatent.
    latent_moments = 1

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
        self.encoder = nn.Sequential(*enc, nn.Conv2d(c, self.latent_moments * latent_channels, 1))

        dec = [nn.Conv2d(latent_channels, channels[-1], 1)]
        for c_in, c_out in zip(channels[::-1], channels[::-1][1:] + [channels[0]]):
            dec += [nn.Upsample(scale_factor=2, mode="nearest"), conv_block(c_in, c_out)]
        self.decoder = nn.Sequential(*dec, nn.Conv2d(channels[0], in_channels, 1))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


class GaussianLatent(Autoencoder):
    """Mixin that turns a backbone into a VAE: its encoder emits a diagonal Gaussian q(z|x), the
    mean and log-variance stacked along dim 1, for each latent element. Same latent shape as the
    backbone, so the two differ only in the loss (`BetaVAE`). List it before the backbone.

    The latent is the mean: `encode` returns it, and outside training the reconstruction decodes
    it, so metrics and probes see the same deterministic code. Training decodes a sample.
    """

    latent_moments = 2

    def moments(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = super().encode(x).chunk(2, dim=1)
        return mu, logvar

    def encode(self, x):
        return self.moments(x)[0]

    def forward(self, x):
        mu, logvar = self.moments(x)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if self.training else mu
        return {"z": z, "xhat": self.decode(z), "mu": mu, "logvar": logvar}


class ConvVAE(GaussianLatent, ConvAE):
    """ConvAE with a Gaussian latent."""


def to_tokens(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(2).transpose(1, 2)  # (B, D, h, w) -> (B, h*w, D)


def from_tokens(t: torch.Tensor, grid: torch.Size) -> torch.Tensor:
    return t.transpose(1, 2).unflatten(2, grid)  # (B, h*w, D) -> (B, D, h, w)


class ViTAE(Autoencoder):
    """Vision transformer autoencoder: MAE (He et al. 2022) without the masking. The encoder embeds
    `patch_size` patches, runs `depth` pre-norm transformer blocks and projects each token to
    `latent_channels`; the decoder mirrors it and ends in a linear map back to the patch pixels.

    The latent is spatial, (latent_channels, H/p, W/p), the same shape as a ConvAE with log2(p)
    pooling levels, so the two families compare at equal latent size and the spatial probes read
    both alike. Position embeddings are fixed 2D sin-cos, built for whatever grid comes in.
    """

    def __init__(self, in_channels: int = 4, patch_size: int = 8, latent_channels: int = 4,
                 dim: int = 192, depth: int = 4, heads: int = 3):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(in_channels, dim, patch_size, stride=patch_size)
        self.encoder = nn.Sequential(*[Block(dim, heads, qkv_bias=True) for _ in range(depth)],
                                     nn.LayerNorm(dim),
                                     nn.Linear(dim, self.latent_moments * latent_channels))
        self.latent_embed = nn.Linear(latent_channels, dim)
        self.decoder = nn.Sequential(*[Block(dim, heads, qkv_bias=True) for _ in range(depth)],
                                     nn.LayerNorm(dim),
                                     nn.Linear(dim, in_channels * patch_size ** 2))

    def pos_embed(self, grid: torch.Size, like: torch.Tensor) -> torch.Tensor:
        return build_sincos2d_pos_embed(list(grid), like.shape[-1], device=like.device,
                                        dtype=like.dtype)

    def encode(self, x):
        h = self.patch_embed(x)
        t = to_tokens(h)
        return from_tokens(self.encoder(t + self.pos_embed(h.shape[2:], t)), h.shape[2:])

    def decode(self, z):
        t = self.latent_embed(to_tokens(z))
        t = self.decoder(t + self.pos_embed(z.shape[2:], t))
        return F.pixel_shuffle(from_tokens(t, z.shape[2:]), self.patch_size)


class ViTVAE(GaussianLatent, ViTAE):
    """ViTAE with a Gaussian latent."""
