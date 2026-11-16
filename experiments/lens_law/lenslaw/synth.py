"""Training patches for the learned blur estimator.

A sharp real patch is blurred by a known amount, the way a lens would (in
linear light, Gaussian or disc-shaped), then pushed through what happens to
photos afterwards: sensor-like noise, downscaling, JPEG. The label is the log
of the blur size that results, in pixels of the final patch. No generated
image is involved anywhere.
"""

from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image

from .render import linear_to_srgb, srgb_to_linear

KINDS = ("gaussian", "disc")
BASE_SIGMA = 0.5  # blur size assigned to a sharp, well-focused source patch
SIGMA_RANGE = (0.3, 12.0)  # added blur, log-uniform
P_SHARP = 0.15


def defocus_kernel(sigma: float, kind: str) -> np.ndarray:
    """Normalised 2-D kernel whose spread per axis is `sigma` pixels."""
    if kind not in KINDS:
        raise ValueError(f"unknown kernel kind {kind!r}; expected one of {KINDS}")
    if kind == "gaussian":
        radius = max(1, math.ceil(3 * sigma))
        x = np.arange(-radius, radius + 1, dtype=np.float64)
        g = np.exp(-0.5 * (x / sigma) ** 2)
        kernel = np.outer(g, g)
    else:
        # A uniform disc of radius 2*sigma has the same second moment in the
        # continuum; on the pixel grid the radius is adjusted until it matches.
        disc = 2.0 * sigma
        for _ in range(5):
            radius = math.ceil(disc + 1)
            x = np.arange(-radius, radius + 1, dtype=np.float64)
            dist = np.hypot(*np.meshgrid(x, x))
            kernel = np.clip(disc + 0.5 - dist, 0.0, 1.0)  # soft edge instead of jagged
            moment = float((kernel.sum(axis=0) * x * x).sum() / kernel.sum())
            if moment <= 0:
                break
            disc *= math.sqrt(sigma**2 / moment)
    return (kernel / kernel.sum()).astype(np.float32)


def blur_linear(rgb: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve an sRGB image (H, W, 3 in [0, 1]) with `kernel` in linear light."""
    r = kernel.shape[0] // 2
    lin = np.pad(srgb_to_linear(rgb), ((r, r), (r, r), (0, 0)), mode="reflect")
    h, w = lin.shape[:2]
    shifted = np.zeros((h, w), dtype=np.float64)
    shifted[: 2 * r + 1, : 2 * r + 1] = kernel
    shifted = np.roll(shifted, (-r, -r), axis=(0, 1))  # kernel centre at the origin
    spectrum = np.fft.rfft2(shifted)[..., None]
    out = np.fft.irfft2(np.fft.rfft2(lin, axes=(0, 1)) * spectrum, s=(h, w), axes=(0, 1))
    return linear_to_srgb(out[r : h - r, r : w - r])


def label_for(sigma: float, scale: float) -> float:
    """Log blur size, in output pixels, of a source blurred by `sigma` then scaled."""
    return 0.5 * math.log(BASE_SIGMA**2 + (sigma * scale) ** 2)


def _to_pil(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8))


def degrade(
    img: np.ndarray,
    sigma: float,
    kind: str,
    scale: float,
    jpeg_quality: int | None,
    noise_std: float,
    out: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Blur, add noise, downscale and JPEG-encode `img`, then take the centre crop."""
    img = np.asarray(img, dtype=np.float32)
    if sigma > 0:
        img = blur_linear(img, defocus_kernel(sigma, kind))
    if noise_std > 0:  # sensor noise is added after the optics, so it stays sharp
        rng = rng or np.random.default_rng()
        img = np.clip(img + rng.normal(0.0, noise_std, img.shape).astype(np.float32), 0, 1)
    if scale != 1.0:
        size = (round(img.shape[1] * scale), round(img.shape[0] * scale))
        img = np.asarray(_to_pil(img).resize(size, Image.BICUBIC), dtype=np.float32) / 255.0
    if jpeg_quality is not None:
        buf = io.BytesIO()
        _to_pil(img).save(buf, format="JPEG", quality=int(jpeg_quality))
        buf.seek(0)
        img = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0
    h, w = img.shape[:2]
    if min(h, w) < out:
        raise ValueError(f"image {h}x{w} is smaller than the {out}px output after scaling")
    y, x = (h - out) // 2, (w - out) // 2
    return np.ascontiguousarray(img[y : y + out, x : x + out])


def make_training_patch(
    sharp: np.ndarray, rng: np.random.Generator, out: int = 96
) -> tuple[np.ndarray, float]:
    """One (patch, log blur size) pair from a sharp source crop."""
    if min(sharp.shape[:2]) < 2 * out + 48:
        raise ValueError(f"source crop must be at least {2 * out + 48}px for a {out}px patch")
    sigma = 0.0
    if rng.random() >= P_SHARP:
        sigma = float(np.exp(rng.uniform(math.log(SIGMA_RANGE[0]), math.log(SIGMA_RANGE[1]))))
    kind = KINDS[int(rng.integers(len(KINDS)))]
    pick = rng.random()
    scale = 0.5 if pick < 0.25 else float(rng.uniform(0.5, 1.0)) if pick < 0.5 else 1.0
    jpeg_quality = int(rng.integers(50, 96)) if rng.random() < 0.6 else None
    noise_std = float(rng.uniform(0.0, 0.02)) if rng.random() < 0.7 else 0.0
    patch = degrade(sharp, sigma, kind, scale, jpeg_quality, noise_std, out, rng)
    return patch, label_for(sigma, scale)
