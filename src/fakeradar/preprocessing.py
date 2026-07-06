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

from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF


def load_image(source) -> Image.Image:
    """Accept a path / Path / PIL image and return a PIL RGB image."""
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    return Image.open(Path(source)).convert("RGB")


def crop_positions(w: int, h: int, size: int, max_crops: int) -> list[tuple[int, int]]:
    """Top-left corners for up to max_crops crops: center first, then corners."""
    cx, cy = (w - size) // 2, (h - size) // 2
    positions = [(cx, cy)]
    corners = [(0, 0), (w - size, 0), (0, h - size), (w - size, h - size)]
    for p in corners:
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
