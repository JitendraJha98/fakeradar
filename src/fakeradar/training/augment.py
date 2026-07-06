"""Robustness-first augmentation.

Most published detectors collapse the moment an image passes through a social
platform (JPEG re-encode + resize). fakeradar trains *through* that pipeline:
every training image is randomly pushed through compression / resampling /
blur / noise before cropping, so the model learns traces that survive
real-world transport, not lab-condition artifacts.
"""

from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image, ImageFilter


class RobustnessAugment:
    def __init__(
        self,
        p_jpeg: float = 0.5,
        jpeg_quality: tuple[int, int] = (30, 95),
        p_webp: float = 0.1,
        p_resize: float = 0.3,
        resize_scale: tuple[float, float] = (0.5, 1.5),
        p_blur: float = 0.15,
        p_noise: float = 0.1,
        p_hflip: float = 0.5,
    ):
        self.p_jpeg, self.jpeg_quality = p_jpeg, jpeg_quality
        self.p_webp = p_webp
        self.p_resize, self.resize_scale = p_resize, resize_scale
        self.p_blur, self.p_noise, self.p_hflip = p_blur, p_noise, p_hflip

    # each op takes and returns a PIL RGB image -------------------------------
    def _recompress(self, img: Image.Image, fmt: str, quality: int) -> Image.Image:
        buf = io.BytesIO()
        img.save(buf, format=fmt, quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p_hflip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < self.p_resize:
            s = random.uniform(*self.resize_scale)
            w, h = img.size
            img = img.resize((max(64, int(w * s)), max(64, int(h * s))), Image.BILINEAR)
        if random.random() < self.p_blur:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.4, 1.6)))
        if random.random() < self.p_noise:
            arr = np.asarray(img).astype(np.float32)
            arr += np.random.normal(0.0, random.uniform(2.0, 8.0), arr.shape)
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        if random.random() < self.p_jpeg:
            img = self._recompress(img, "JPEG", random.randint(*self.jpeg_quality))
        elif random.random() < self.p_webp:
            img = self._recompress(img, "WEBP", random.randint(50, 95))
        return img
