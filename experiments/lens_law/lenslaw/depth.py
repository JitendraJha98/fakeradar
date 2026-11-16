"""Monocular depth: Depth Anything V2 Small (Apache-2.0), relative inverse depth.

torch and transformers are imported inside the functions so the rest of the
package, and its tests, work without downloading a model.
"""

from __future__ import annotations

import numpy as np

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


def normalize_inverse_depth(raw: np.ndarray) -> np.ndarray:
    """Map the 2nd and 98th percentiles to 0 and 1. A constant map becomes zeros.

    The lens-law fit does not depend on this scale; it only keeps numbers tidy
    and makes focus quantiles comparable between images.
    """
    raw = np.asarray(raw, dtype=np.float32)
    lo, hi = np.percentile(raw, [2, 98])
    if hi - lo <= 1e-12:
        return np.zeros_like(raw)
    return ((raw - lo) / (hi - lo)).astype(np.float32)


def load_depth(device: str = "cuda"):
    """Return (processor, model) on `device`."""
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).to(device).eval()
    return processor, model


def inverse_depth(loaded, img) -> np.ndarray:
    """(H, W) normalised inverse depth for a PIL RGB image, at the image's size."""
    import torch

    processor, model = loaded
    device = next(model.parameters()).device
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        pred = model(**inputs).predicted_depth
    pred = torch.nn.functional.interpolate(
        pred[:, None].float(), size=(img.height, img.width), mode="bicubic", align_corners=False
    )
    return normalize_inverse_depth(pred[0, 0].cpu().numpy())
