"""Controlled defocus renders: one lawful, three that break the lens law.

All four use the same renderer, the same sharp photo and the same depth map;
only the rule that turns depth into blur differs. Blur is applied in linear
light, as a lens does, by blending a stack of Gaussian-blurred layers.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

MODES = ("lawful", "step", "position", "missing_foreground")
STEP_MARGIN = 0.05  # how far behind the focus plane the step mode starts to blur


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(np.asarray(c, dtype=np.float32), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055).astype(np.float32)


def blur_field(inv_depth: np.ndarray, mode: str, d_focus: float, sigma_max: float) -> np.ndarray:
    """Blur size per pixel. Larger inverse depth means nearer the camera."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    inv_depth = np.asarray(inv_depth, dtype=np.float32)
    h = inv_depth.shape[0]
    if mode == "position":
        rows = np.abs(np.arange(h, dtype=np.float32) / h - 0.5) * 2.0
        return np.broadcast_to(sigma_max * rows[:, None], inv_depth.shape).astype(np.float32)
    if mode == "step":
        return np.where(inv_depth < d_focus - STEP_MARGIN, sigma_max, 0.0).astype(np.float32)

    u = np.abs(inv_depth - d_focus)
    scale = float(np.percentile(u, 95))
    lawful = sigma_max * u / scale if scale > 1e-12 else np.zeros_like(u)
    if mode == "missing_foreground":
        lawful = np.where(inv_depth <= d_focus, lawful, 0.0)
    return lawful.astype(np.float32)


def _gaussian_layer(lin: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 1e-6:
        return lin
    h, w = lin.shape[-2:]
    radius = min(int(math.ceil(3 * sigma)), h - 1, w - 1)
    if radius < 1:
        return lin
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=lin.device)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = (k / k.sum()).to(lin.dtype)
    c = lin.shape[1]
    out = F.pad(lin, (radius, radius, 0, 0), mode="reflect")
    out = F.conv2d(out, k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    out = F.pad(out, (0, 0, radius, radius), mode="reflect")
    return F.conv2d(out, k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


def render_defocus(
    rgb: np.ndarray, sigma_field: np.ndarray, n_levels: int = 12, device: str = "cpu"
) -> np.ndarray:
    """Blur an sRGB image (H, W, 3 in [0, 1]) by a per-pixel Gaussian size."""
    rgb = np.asarray(rgb, dtype=np.float32)
    sigma_field = np.asarray(sigma_field, dtype=np.float32)
    top = float(sigma_field.max())
    if top <= 1e-6:
        return rgb.copy()
    lin = torch.from_numpy(srgb_to_linear(rgb)).permute(2, 0, 1)[None].to(device)
    position = torch.from_numpy(sigma_field / top * (n_levels - 1))[None, None].to(device)
    out = torch.zeros_like(lin)
    for level in range(n_levels):
        weight = (1.0 - (position - level).abs()).clamp_min(0.0)  # blend the two nearest layers
        if float(weight.max()) > 0:
            out += weight * _gaussian_layer(lin, top * level / (n_levels - 1))
    return linear_to_srgb(out[0].permute(1, 2, 0).cpu().numpy())
