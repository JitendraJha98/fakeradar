"""Post-hoc temperature calibration (Guo et al., 2017).

`fakeradar calibrate val.csv --checkpoint best.pt` fits the fusion head's
temperature on held-out logits and re-saves the checkpoint with the fitted
value, reporting ECE before/after. `training/train.py` runs this automatically
on the best checkpoint after training (config `run.calibrate`, default true).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..models import FakeRadarModel, load_checkpoint, save_checkpoint
from ..training.datasets import ManifestDataset
from .metrics import expected_calibration_error


@torch.no_grad()
def collect_logits(
    model: FakeRadarModel,
    manifest: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Center-crop fused logits + labels for every manifest row (no augmentation)."""
    semantic_pre = None
    for name in model.resized_branch_names:
        semantic_pre = model.branches[name].preprocess
        break
    ds = ManifestDataset(
        manifest, model.config.crop_size, augment=False, semantic_preprocess=semantic_pre
    )
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
    model.to(device).eval()
    logits, labels = [], []
    for batch in dl:
        semantic = batch.get("semantic")
        semantic = semantic.to(device) if semantic is not None else None
        out = model(batch["crop"].to(device), semantic)
        logits.append(out["logit"].cpu())
        labels.append(batch["label"])
    return torch.cat(logits), torch.cat(labels)


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
    batch_size: int = 32,
    num_workers: int = 0,
    device: str = "cpu",
) -> dict[str, float]:
    """Load a checkpoint, fit its temperature on `manifest`, save it back
    (in place unless `out` is given) and return the calibration stats."""
    model = load_checkpoint(checkpoint, map_location=device)
    logits, labels = collect_logits(model, manifest, batch_size, num_workers, device)
    stats = calibrate_model(model, logits, labels)
    save_checkpoint(model, out or checkpoint, extra={"calibration": stats})
    return stats
