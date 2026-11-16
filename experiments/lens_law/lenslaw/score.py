"""Image + inverse-depth map -> one lens-law record."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .blur_classic import estimate_blur
from .blur_net import net_points
from .fit import score_points
from .patches import applicable, collect_points

WORK_LONG_SIDE = 1536  # larger images are downscaled to this before scoring
_SIXTEEN_BIT_MODES = ("I;16", "I;16L", "I;16B", "I")


def load_rgb(source) -> np.ndarray:
    """Path or PIL image, any mode or bit depth -> float32 RGB in [0, 1]."""
    img = source if isinstance(source, Image.Image) else Image.open(Path(source))
    if img.mode in _SIXTEEN_BIT_MODES or img.mode == "F":
        arr = np.asarray(img, dtype=np.float32)
        top = 65535.0 if arr.max() > 255 else 255.0 if arr.max() > 1 else 1.0
        img = Image.fromarray(np.clip(arr / top * 255.0, 0, 255).astype(np.uint8), mode="L")
    img = img.convert("RGB")
    long_side = max(img.size)
    if long_side > WORK_LONG_SIDE:
        scale = WORK_LONG_SIDE / long_side
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def to_gray(rgb: np.ndarray) -> np.ndarray:
    return (rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)).astype(np.float32)


def _match_shape(inv_depth: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    inv_depth = np.asarray(inv_depth, dtype=np.float32)
    if inv_depth.shape == shape:
        return inv_depth
    t = torch.from_numpy(inv_depth)[None, None]
    t = torch.nn.functional.interpolate(t, size=shape, mode="bilinear", align_corners=False)
    return t[0, 0].numpy()


def score_arrays(
    rgb: np.ndarray, inv_depth: np.ndarray, device: str = "cpu", blur_model=None
) -> dict:
    """Lens-law numbers for one image. Blur comes from `rgb`, depth is given.

    Passing the depth map of a different, aligned image (for example the
    all-in-focus twin of a wide-aperture photo) is allowed and intended.
    With `blur_model`, blur comes from the learned network instead of edges.
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    depth = _match_shape(inv_depth, rgb.shape[:2])
    if blur_model is not None:  # estimator v1: the learned network
        points = net_points(blur_model, rgb, depth, device)
    else:  # estimator v0: gradient ratio at edges
        points = collect_points(estimate_blur(to_gray(rgb), device=device), depth)
    out = score_points(points.d, points.b, points.w)
    out["applicable"] = applicable(points)
    return out
