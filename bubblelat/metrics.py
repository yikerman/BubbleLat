"""Evaluation metrics on physical-unit tensors of shape (B, C, H, W)."""

import torch

BANDS = ["low", "mid", "high"]
METRICS = ["mae", "rmse", "vrmse"] + [f"spec_{b}" for b in BANDS]


def band_masks(h: int, w: int, device=None) -> torch.Tensor:
    """(3, h, w // 2 + 1) masks over rfft2 wavenumbers: three bands evenly spaced in log |k|.

    |k| is in cycles per domain. The edges are Nyquist^(1/3) and Nyquist^(2/3) (4 and 16 at
    128 px), as in The Well's binned spectral error. The mean (k = 0) is in no band.
    """
    ky = torch.fft.fftfreq(h, 1 / h, device=device)[:, None]
    kx = torch.fft.rfftfreq(w, 1 / w, device=device)[None, :]
    k = (ky**2 + kx**2).sqrt()
    nyquist = min(h, w) / 2
    lo, hi = nyquist ** (1 / 3), nyquist ** (2 / 3)
    return torch.stack([(k > 0) & (k < lo), (k >= lo) & (k < hi), k >= hi])


def field_errors(pred: torch.Tensor, target: torch.Tensor, fields: list[str]) -> dict:
    """Batch-mean errors for each field, keyed "<metric>/<field>".

    VRMSE is the RMSE divided by the target frame's spatial std (The Well, Ohana et al. 2024):
    1.0 means "no better than predicting the frame's mean". spec_<band> is the same ratio
    restricted to one wavenumber band: sqrt(error power / target power) in that band.
    """
    err = pred - target
    mse = err.pow(2).mean(dim=(2, 3))  # (B, C)
    var = target.var(dim=(2, 3), unbiased=False)

    # One-sided spectrum: double every column except kx = 0 and kx = Nyquist so that band
    # powers sum to the frame's variance (Parseval).
    h, w = target.shape[2:]
    weight = torch.full((w // 2 + 1,), 2.0, device=target.device)
    weight[0] = 1.0
    if w % 2 == 0:
        weight[-1] = 1.0
    masks = band_masks(h, w, target.device) * weight  # (3, h, w/2+1)

    def band_power(x):  # (B, C, 3)
        power = torch.fft.rfft2(x, norm="forward").abs().pow(2)
        return torch.einsum("bchw,nhw->bcn", power, masks)

    spec = (band_power(err) / (band_power(target) + 1e-7)).sqrt()

    out = {}
    for c, name in enumerate(fields):
        out[f"mae/{name}"] = err[:, c].abs().mean()
        out[f"rmse/{name}"] = mse[:, c].sqrt().mean()
        out[f"vrmse/{name}"] = (mse[:, c] / (var[:, c] + 1e-7)).sqrt().mean()
        for n, band in enumerate(BANDS):
            out[f"spec_{band}/{name}"] = spec[:, c, n].mean()
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
