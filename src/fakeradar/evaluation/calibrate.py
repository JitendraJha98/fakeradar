"""Post-hoc temperature calibration (Guo et al., 2017).

`fakeradar calibrate val.csv --checkpoint best.pt` fits the fusion head's
temperature on held-out logits and re-saves the checkpoint with the fitted
value, reporting ECE before/after. `training/train.py` runs this automatically
on the best checkpoint after training (config `run.calibrate`, default true).

Logits are collected with the *deployment* protocol — the mean fused logit
over the same native-res crop grid `Detector.predict` uses — so the fitted
temperature applies to exactly the quantity it rescales at inference time.
(Averaging K crop logits shrinks their spread; a temperature fit on single
center crops would over-soften multi-crop predictions.)
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..models import (
    FakeRadarModel,
    checkpoint_extras,
    model_from_checkpoint,
    read_checkpoint,
    save_checkpoint,
)
from ..training.datasets import ManifestDataset
from .metrics import expected_calibration_error


@torch.no_grad()
def collect_logits(
    model: FakeRadarModel,
    manifest: str | Path,
    num_workers: int = 0,
    device: str = "cpu",
    equalize_jpeg_quality: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-image fused logits (mean over the inference crop grid) + labels."""
    ds = ManifestDataset(
        manifest,
        model.config.crop_size,
        augment=False,
        semantic_preprocess=model.semantic_preprocessor(),
        max_crops=model.config.max_crops,
        equalize_jpeg_quality=equalize_jpeg_quality,
    )
    # batch_size=None: one image (K crops, K varies with image size) per step
    dl = DataLoader(ds, batch_size=None, num_workers=num_workers)
    model.to(device).eval()
    logits, labels = [], []
    for item in dl:
        semantic = item.get("semantic")
        semantic = semantic.unsqueeze(0).to(device) if semantic is not None else None
        out = model(item["crops"].to(device), semantic)
        logits.append(out["logit"].mean().cpu())
        labels.append(item["label"])
    return torch.stack(logits), torch.stack(labels)


def calibrate_model(
    model: FakeRadarModel, logits: torch.Tensor, labels: torch.Tensor
) -> dict[str, float]:
    """Fit the temperature in place; return {temperature, ece_before, ece_after, n}."""
    if len(torch.unique(labels)) < 2:
        raise ValueError("Calibration manifest must contain both real (0) and AI (1) rows.")
    before = torch.sigmoid(logits / model.fusion.temperature.cpu())
    ece_before = expected_calibration_error(labels.numpy(), before.numpy())
    t = model.fusion.fit_temperature(logits, labels)
    after = torch.sigmoid(logits / t)
    ece_after = expected_calibration_error(labels.numpy(), after.numpy())
    return {
        "temperature": t,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "n": int(labels.numel()),
    }


def calibrate_checkpoint(
    checkpoint: str | Path,
    manifest: str | Path,
    out: str | Path | None = None,
    num_workers: int = 0,
    device: str = "cpu",
    equalize_jpeg_quality: int | None = None,
) -> dict[str, float]:
    """Load a checkpoint, fit its temperature on `manifest`, save it back
    (in place unless `out` is given) and return the calibration stats.
    Other stored metadata (epoch, val metrics, ...) is preserved."""
    ckpt = read_checkpoint(checkpoint, map_location=device)
    model = model_from_checkpoint(ckpt)
    logits, labels = collect_logits(model, manifest, num_workers, device, equalize_jpeg_quality)
    stats = calibrate_model(model, logits, labels)
    save_checkpoint(
        model, out or checkpoint, extra={**checkpoint_extras(ckpt), "calibration": stats}
    )
    return stats
