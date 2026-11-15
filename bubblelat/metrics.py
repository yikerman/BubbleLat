"""Evaluation metrics on physical-unit tensors of shape (B, C, H, W)."""

import torch

BANDS = ["low", "mid", "high"]
METRICS = ["mae", "rmse", "vrmse"] + [f"spec_{b}" for b in BANDS]


def radial_power(x: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, C, K): power in each wavenumber shell j <= |k| < j + 1.

    |k| is in cycles per domain. Shell 0 is the mean, and the shells sum to the frame's mean
    square (Parseval), so the others sum to its variance.
    """
    h, w = x.shape[2:]
    ky = torch.fft.fftfreq(h, 1 / h, device=x.device)[:, None]
    kx = torch.fft.rfftfreq(w, 1 / w, device=x.device)[None, :]
    shell = (ky**2 + kx**2).sqrt().floor().long().flatten()
    # One-sided spectrum: every column counts twice except kx = 0 and kx = Nyquist.
    weight = torch.full((w // 2 + 1,), 2.0, device=x.device)
    weight[0] = 1.0
    if w % 2 == 0:
        weight[-1] = 1.0
    power = (torch.fft.rfft2(x, norm="forward").abs().pow(2) * weight).flatten(2)
    return power.new_zeros(*power.shape[:2], int(shell.max()) + 1).index_add_(2, shell, power)


def band_slices(h: int, w: int) -> list[slice]:
    """Shell ranges of three bands evenly spaced in log |k|, without the mean. The edges are
    Nyquist^(1/3) and Nyquist^(2/3) (4 and 16 at 128 px), as in The Well's binned spectral error.
    """
    nyquist = min(h, w) / 2
    lo, hi = round(nyquist ** (1 / 3)), round(nyquist ** (2 / 3))
    return [slice(1, lo), slice(lo, hi), slice(hi, None)]


def field_errors(pred: torch.Tensor, target: torch.Tensor, fields: list[str]) -> dict:
    """Batch-mean errors for each field, keyed "<metric>/<field>".

    VRMSE is the RMSE divided by the target frame's spatial std (The Well, Ohana et al. 2024):
    1.0 means "no better than predicting the frame's mean". spec_<band> is the same ratio
    restricted to one wavenumber band: sqrt(error power / target power) in that band.
    """
    err = pred - target
    mse = err.pow(2).mean(dim=(2, 3))  # (B, C)
    var = target.var(dim=(2, 3), unbiased=False)
    err_power, target_power = radial_power(err), radial_power(target)  # (B, C, K)
    spec = [(err_power[..., s].sum(-1) / (target_power[..., s].sum(-1) + 1e-7)).sqrt()
            for s in band_slices(*target.shape[2:])]

    out = {}
    for c, name in enumerate(fields):
        out[f"mae/{name}"] = err[:, c].abs().mean()
        out[f"rmse/{name}"] = mse[:, c].sqrt().mean()
        out[f"vrmse/{name}"] = (mse[:, c] / (var[:, c] + 1e-7)).sqrt().mean()
        for band, value in zip(BANDS, spec):
            out[f"spec_{band}/{name}"] = value[:, c].mean()
    return out


def vapor_errors(pred_dfun: torch.Tensor, target_dfun: torch.Tensor) -> dict:
    """Batch-mean vapor-mask errors from the signed distance field (B, H, W); vapor is dfun > 0.

    "vapor/iou" is the intersection over union of the two masks (1.0 when neither frame has
    vapor) and "vapor/frac_err" is the absolute error of the vapor area fraction.
    """
    p, t = pred_dfun > 0, target_dfun > 0
    inter, union = (p & t).sum(dim=(1, 2)), (p | t).sum(dim=(1, 2))
    iou = torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(inter, dtype=torch.float))
    frac_err = (p.float().mean(dim=(1, 2)) - t.float().mean(dim=(1, 2))).abs()
    return {"vapor/iou": iou.mean(), "vapor/frac_err": frac_err.mean()}
