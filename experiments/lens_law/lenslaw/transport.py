"""The three transport conditions from the spec: clean, 0.75x + JPEG 85, 0.5x + JPEG 70."""

from __future__ import annotations

import io

from PIL import Image

CONDITIONS = ("clean", "s075_q85", "s050_q70")
_SCALE_AND_QUALITY = {"s075_q85": (0.75, 85), "s050_q70": (0.5, 70)}


def apply_condition(img: Image.Image, name: str) -> Image.Image:
    """Downscale and JPEG-encode an image the way a sharing platform would."""
    if name not in CONDITIONS:
        raise ValueError(f"unknown condition {name!r}; expected one of {CONDITIONS}")
    img = img.convert("RGB")
    if name == "clean":
        return img
    scale, quality = _SCALE_AND_QUALITY[name]
    size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    buf = io.BytesIO()
    img.resize(size, Image.BICUBIC).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")
