"""Blur estimator v0: multi-scale gradient ratio at edges.

After Zhuo and Sim (2011). An ideal step edge blurred by a Gaussian of size
sigma loses gradient strength in a known way when it is blurred again by a
known amount, so the ratio of the two gradients gives sigma. The ratio only
resolves a narrow range of blur sizes, so the image is analysed at several
scales and each pixel takes the finest scale that can resolve its blur.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

BAND = (0.6, 3.0)  # blur sizes, in pixels at a scale, that the ratio can resolve
EDGE_FLOOR = 0.015  # minimum gradient of the re-blurred image, intensity per pixel
MIN_SIDE = 12  # scales that shrink the image below this are skipped


@dataclass
class BlurMap:
    sigma: np.ndarray  # (H, W) float32, blur size in pixels, NaN where unknown
    conf: np.ndarray  # (H, W) float32 in [0, 1], 0 where unknown


def _gauss1d(sigma: float) -> torch.Tensor:
    radius = max(1, int(math.ceil(4 * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _blur(t: torch.Tensor, sigma: float) -> torch.Tensor:
    k = _gauss1d(sigma).to(t.device)
    r = k.numel() // 2
    t = F.conv2d(F.pad(t, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(t, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))


def _grad(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    p = F.pad(t, (1, 1, 1, 1), mode="replicate")
    return 0.5 * (p[..., 1:-1, 2:] - p[..., 1:-1, :-2]), 0.5 * (
        p[..., 2:, 1:-1] - p[..., :-2, 1:-1]
    )


@functools.lru_cache(maxsize=4)
def _ratio_table(sigma0: float) -> tuple[np.ndarray, np.ndarray]:
    """Gradient ratio of a sampled step edge for each blur size in BAND.

    Simulated with the same discrete filters the estimator uses, averaged over
    sub-pixel edge positions, so the inversion is exact for ideal edges.
    """
    sizes = np.geomspace(BAND[0], BAND[1], 64)
    x = torch.arange(-48, 49, dtype=torch.float32)
    k = _gauss1d(sigma0)
    r = k.numel() // 2
    ratios = []
    for size in sizes:
        vals = []
        for offset in (0.0, 0.25, 0.5, 0.75):
            edge = 0.5 * (1 + torch.erf((x - offset) / (float(size) * math.sqrt(2))))
            padded = F.pad(edge.view(1, 1, -1), (r, r), mode="replicate")
            again = F.conv1d(padded, k.view(1, 1, -1)).view(-1)
            g = 0.5 * (edge[2:] - edge[:-2])
            g1 = 0.5 * (again[2:] - again[:-2])
            i = int(torch.argmax(g1))
            vals.append(float(g[i] / g1[i]))
        ratios.append(float(np.mean(vals)))
    return sizes, np.minimum.accumulate(np.array(ratios))  # ratio falls as blur grows


def estimate_blur(
    gray: np.ndarray,
    scales: tuple[int, ...] = (1, 2, 4, 8, 16),
    sigma0: float = 1.0,
    device: str = "cpu",
) -> BlurMap:
    """Blur size at edge pixels of a grayscale image in [0, 1]."""
    gray = np.asarray(gray, dtype=np.float32)
    h, w = gray.shape
    sigma = np.full((h, w), np.nan, dtype=np.float32)
    conf = np.zeros((h, w), dtype=np.float32)
    sizes, ratios = _ratio_table(float(sigma0))
    r_lo, r_hi = float(ratios[-1]), float(ratios[0])
    full = torch.from_numpy(gray)[None, None].to(device)
    scales = tuple(sorted(scales))
    finest = scales[0]

    for s in scales:
        hs, ws = h // s, w // s
        if min(hs, ws) < MIN_SIDE:
            continue
        t = full[..., : hs * s, : ws * s]
        if s > 1:
            t = F.avg_pool2d(t, s)
        gx, gy = _grad(t)
        g = torch.sqrt(gx * gx + gy * gy)
        gx1, gy1 = _grad(_blur(t, sigma0))
        g1 = torch.sqrt(gx1 * gx1 + gy1 * gy1)
        threshold = max(EDGE_FLOOR, 3.0 * float(g1.median()))

        # edge pixels: ridge of the re-blurred gradient along its dominant axis
        p = F.pad(g1, (1, 1, 1, 1), mode="replicate")
        across_x = (g1 >= p[..., 1:-1, :-2]) & (g1 >= p[..., 1:-1, 2:])
        across_y = (g1 >= p[..., :-2, 1:-1]) & (g1 >= p[..., 2:, 1:-1])
        ridge = torch.where(gx1.abs() >= gy1.abs(), across_x, across_y)
        ratio = (g / g1.clamp_min(1e-12))[0, 0].cpu().numpy()
        edge = (ridge & (g1 > threshold))[0, 0].cpu().numpy()
        if s == finest:
            # sharper than the band can resolve: report the floor instead of nothing,
            # so in-focus regions still contribute points
            ratio = np.minimum(ratio, r_hi)
        else:
            # Below the band the ratio is ambiguous and reads too large. An edge a
            # finer scale already resolved is therefore never re-measured here.
            have = np.isfinite(sigma[: hs * s, : ws * s]).reshape(hs, s, ws, s).any(axis=(1, 3))
            near = np.pad(have, 1)
            covered = np.zeros_like(have)
            for dy in range(3):
                for dx in range(3):
                    covered |= near[dy : dy + hs, dx : dx + ws]
            edge &= ~covered
        ok = edge & (ratio >= r_lo) & (ratio <= r_hi)
        if not ok.any():
            continue

        at_scale = np.interp(ratio[ok], ratios[::-1], sizes[::-1])
        box = (1.0 - 1.0 / (s * s)) / 12.0  # blur added by averaging s x s pixels
        est = np.full((hs, ws), np.nan, dtype=np.float32)
        est[ok] = s * np.sqrt(np.maximum(at_scale**2 - box, 0.05**2))
        strength = np.zeros((hs, ws), dtype=np.float32)
        strength[ok] = np.clip(g1[0, 0].cpu().numpy()[ok] / (4.0 * threshold), 0.0, 1.0)

        up = np.repeat(np.repeat(est, s, axis=0), s, axis=1)
        up_conf = np.repeat(np.repeat(strength, s, axis=0), s, axis=1)
        region = sigma[: hs * s, : ws * s]
        fill = np.isnan(region) & np.isfinite(up)  # finer scales were filled first
        region[fill] = up[fill]
        conf[: hs * s, : ws * s][fill] = up_conf[fill]
    return BlurMap(sigma, conf)
