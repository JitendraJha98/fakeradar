"""Blur estimator v1: a small fully convolutional network.

It reads RGB and predicts, for every 32-pixel grid cell, the log of the local
blur size and how sure it is (a log variance). Training uses real sharp
photos with synthetic defocus (see synth.py) and a Gaussian likelihood, so a
cell with nothing to measure, such as blank sky, learns to report a large
uncertainty instead of a guess.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .patches import Points, _blocks

STRIDE = 32  # one prediction per 32 x 32 pixel cell
MAX_STD = 0.5  # cells whose predicted std of log blur exceeds this are dropped
LOGVAR_RANGE = (-7.0, 5.0)
FULL_WEIGHT_STD = 0.2  # a cell this certain gets weight 0.5; more certain tends to 1
_WIDTHS = (32, 48, 64, 96, 128, 160)


def _stage(cin: int, cout: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class BlurNet(nn.Module):
    """(B, 3, H, W) in [0, 1] -> (B, 2, H/32, W/32): mean log blur, log variance."""

    def __init__(self) -> None:
        super().__init__()
        stages = [_stage(3, _WIDTHS[0], 1)]
        stages += [_stage(a, b, 2) for a, b in zip(_WIDTHS[:-1], _WIDTHS[1:], strict=True)]
        self.body = nn.Sequential(*stages)
        self.head = nn.Conv2d(_WIDTHS[-1], 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.head(self.body(x - 0.5))
        return torch.cat([out[:, :1], out[:, 1:].clamp(*LOGVAR_RANGE)], dim=1)


def nll_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Gaussian negative log-likelihood of the patch label at every output cell."""
    mean, logvar = pred[:, 0], pred[:, 1]
    target = target.view(-1, 1, 1)
    return (0.5 * (logvar + (target - mean) ** 2 * torch.exp(-logvar))).mean()


def predict_cells(
    model: nn.Module, rgb: np.ndarray, device: str = "cpu"
) -> tuple[np.ndarray, np.ndarray]:
    """(log blur size, its predicted std) for each full 32-pixel cell of an image."""
    rgb = np.asarray(rgb, dtype=np.float32)
    h, w = (rgb.shape[0] // STRIDE) * STRIDE, (rgb.shape[1] // STRIDE) * STRIDE
    x = torch.from_numpy(np.ascontiguousarray(rgb[:h, :w])).permute(2, 0, 1)[None].to(device)
    model = model.to(device).eval()
    with torch.no_grad():
        out = model(x)[0].float().cpu().numpy()
    return out[0], np.exp(0.5 * out[1])


def net_points(
    model: nn.Module,
    rgb: np.ndarray,
    inv_depth: np.ndarray,
    device: str = "cpu",
    max_std: float = MAX_STD,
    max_depth_spread: float = 0.10,
) -> Points:
    """One (inverse depth, log blur) point per cell the network is sure about."""
    inv_depth = np.asarray(inv_depth, dtype=np.float32)
    if inv_depth.shape != rgb.shape[:2]:
        raise ValueError(f"depth map {inv_depth.shape} and image {rgb.shape[:2]} differ in shape")
    log_sigma, std = predict_cells(model, rgb, device)
    ny, nx = log_sigma.shape
    lo, hi = np.percentile(inv_depth, [2, 98])
    span = max(float(hi - lo), 1e-12)
    depth = _blocks(inv_depth, STRIDE)
    p10, p50, p90 = np.percentile(depth, [10, 50, 90], axis=2)
    keep = (std <= max_std) & ((p90 - p10) / span <= max_depth_spread)
    iy, ix = np.nonzero(keep)
    weight = 1.0 / (1.0 + (std[iy, ix] / FULL_WEIGHT_STD) ** 2)
    yx = np.stack([iy * STRIDE + STRIDE / 2, ix * STRIDE + STRIDE / 2], axis=1).reshape(-1, 2)
    return Points(p50[iy, ix].astype(np.float64), log_sigma[iy, ix].astype(np.float64), weight, yx)
