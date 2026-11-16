"""Inputs for the reference detector, and FODB file names.

The preprocessing reproduces the Community Forensics test transform for the
384-input model: resize the shorter side to 440, centre-crop 384, scale to
[0, 1], normalise with ImageNet statistics.
"""

from __future__ import annotations

import re

import torch
from PIL import Image
from torchvision.transforms import functional as TF

RESIZE, CROP = 440, 384
_MEAN, _STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
_FODB = re.compile(r"(D\d{2})_img_([A-Za-z]+)_(\d{4})\.jpe?g$", re.IGNORECASE)


def commfor_preprocess(img: Image.Image) -> torch.Tensor:
    """PIL image -> (3, 384, 384) tensor as the reference detector expects."""
    img = TF.center_crop(TF.resize(img.convert("RGB"), RESIZE), [CROP, CROP])
    return TF.normalize(TF.to_tensor(img), _MEAN, _STD)


def parse_fodb_name(name: str) -> tuple[str, str, str] | None:
    """(phone, platform, scene) from an FODB file path, or None if it is not a photo."""
    match = _FODB.search(name)
    if not match:
        return None
    return match.group(1).upper(), match.group(2).lower(), match.group(3)
