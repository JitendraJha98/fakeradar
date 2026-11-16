"""Grid cells -> (inverse depth, log blur) points, and the applicability rule."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .blur_classic import BlurMap

MIN_CELLS = 60  # confident cells needed before the law can be tested
MIN_BLUR_SPREAD = 2.0  # P90/P10 of blur size: depth of field must be visible
FULL_WEIGHT_EDGE_PX = 50  # a cell with this many blur estimates gets weight 1


@dataclass
class Points:
    d: np.ndarray  # (N,) median inverse depth of each cell
    b: np.ndarray  # (N,) median natural-log blur size of each cell
    w: np.ndarray  # (N,) weight in (0, 1]
    yx: np.ndarray  # (N, 2) cell centres in pixels


def _blocks(a: np.ndarray, cell: int) -> np.ndarray:
    """(H, W) -> (rows, cols, cell*cell), dropping partial cells at the border."""
    ny, nx = a.shape[0] // cell, a.shape[1] // cell
    a = a[: ny * cell, : nx * cell].reshape(ny, cell, nx, cell)
    return a.transpose(0, 2, 1, 3).reshape(ny, nx, cell * cell)


def collect_points(
    blur: BlurMap,
    inv_depth: np.ndarray,
    cell: int = 32,
    min_edge_px: int = 12,
    max_depth_spread: float = 0.10,
) -> Points:
    """One point per grid cell that has enough blur estimates and no depth edge."""
    inv_depth = np.asarray(inv_depth, dtype=np.float32)
    if inv_depth.shape != blur.sigma.shape:
        raise ValueError(
            f"depth map {inv_depth.shape} and blur map {blur.sigma.shape} differ in shape"
        )
    lo, hi = np.percentile(inv_depth, [2, 98])
    span = max(float(hi - lo), 1e-12)

    sigma = _blocks(blur.sigma, cell)
    depth = _blocks(inv_depth, cell)
    count = np.isfinite(sigma).sum(axis=2)
    p10, p50, p90 = np.percentile(depth, [10, 50, 90], axis=2)
    keep = (count >= min_edge_px) & ((p90 - p10) / span <= max_depth_spread)
    iy, ix = np.nonzero(keep)
    if iy.size == 0:
        empty = np.zeros(0)
        return Points(empty, empty.copy(), empty.copy(), np.zeros((0, 2)))

    b = np.nanmedian(np.log(sigma[iy, ix]), axis=1)
    w = np.minimum(1.0, count[iy, ix] / FULL_WEIGHT_EDGE_PX)
    yx = np.stack([iy * cell + cell / 2, ix * cell + cell / 2], axis=1)
    return Points(p50[iy, ix].astype(np.float64), b.astype(np.float64), w, yx)


def applicable(points: Points) -> bool:
    """True when depth of field is visible enough for the lens law to be tested."""
    if len(points.b) < MIN_CELLS:
        return False
    spread = float(np.exp(np.percentile(points.b, 90) - np.percentile(points.b, 10)))
    return spread >= MIN_BLUR_SPREAD
