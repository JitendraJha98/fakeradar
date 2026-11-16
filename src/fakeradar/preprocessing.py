"""Preprocessing: loading + native-resolution crop extraction.

The single most common mistake in AI-image detection pipelines is resizing
everything to 224x224 — which destroys exactly the high-frequency traces the
forensic branches detect. fakeradar therefore:

  * never resizes inputs for the gradient/frequency branches;
  * takes up to `max_crops` native-resolution crops (center + corners) and
    aggregates their scores, so large images are covered without resampling;
  * only upscales when the image is *smaller* than the crop size.

The semantic branch does its own resized preprocessing separately.
"""

from __future__ import annotations

import io
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF


def load_image(source) -> Image.Image:
    """Accept a path / Path / PIL image and return a PIL RGB image."""
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    with Image.open(Path(source)) as img:  # closes the file handle after decode
        return img.convert("RGB")


_JPEG_FORMATS = ("JPEG", "MPO")


def load_image_equalized(source, jpeg_quality: int | None) -> Image.Image:
    """`load_image`, plus one JPEG pass for images that never had one.

    Some training sets store real photos as JPEG and generated images as
    lossless PNG (GenImage does), so "has JPEG traces" becomes a shortcut for
    "real". With `jpeg_quality` set, an image whose file is not already a JPEG
    (judged by the file's header, not its name) is JPEG-encoded once at that
    quality, giving both classes the same compression history. JPEG sources
    are returned untouched; `None` is exactly `load_image`.
    """
    if jpeg_quality is None:
        return load_image(source)
    if isinstance(source, Image.Image):
        fmt, img = source.format, source.convert("RGB")
    else:
        with Image.open(Path(source)) as opened:
            fmt, img = opened.format, opened.convert("RGB")
    if fmt in _JPEG_FORMATS:
        return img
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(jpeg_quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def semantic_view(img: Image.Image, size: int) -> torch.Tensor:
    """PIL RGB image -> (1, 3, size, size) resized + center-cropped global view
    in [0,1] (semantic branches only — forensic branches never see this)."""
    img = TF.resize(img, size, antialias=True)
    img = TF.center_crop(img, [size, size])
    return TF.to_tensor(img).unsqueeze(0)


def crop_positions(w: int, h: int, size: int, max_crops: int) -> list[tuple[int, int]]:
    """Top-left corners for up to max_crops crops, in coverage-priority order:
    center, then the four corners, then the four edge midpoints (a 3x3 grid)."""
    x0, x1, xc = 0, w - size, (w - size) // 2
    y0, y1, yc = 0, h - size, (h - size) // 2
    grid = [
        (xc, yc),  # center
        (x0, y0),
        (x1, y0),
        (x0, y1),
        (x1, y1),  # corners
        (xc, y0),
        (xc, y1),
        (x0, yc),
        (x1, yc),  # edge midpoints
    ]
    positions: list[tuple[int, int]] = []
    for p in grid:
        if p not in positions:
            positions.append(p)
    return positions[: max(1, max_crops)]


def extract_crops(img: Image.Image, size: int = 256, max_crops: int = 4) -> torch.Tensor:
    """PIL image -> (N, 3, size, size) float tensor in [0,1], no resampling
    unless the image is smaller than `size`."""
    w, h = img.size
    if min(w, h) < size:
        img = TF.resize(img, size, antialias=True)  # upscale small images only
        w, h = img.size
    crops = [
        TF.to_tensor(img.crop((x, y, x + size, y + size)))
        for x, y in crop_positions(w, h, size, max_crops)
    ]
    return torch.stack(crops, dim=0)
