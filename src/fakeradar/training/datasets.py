"""Datasets.

Everything trains from a simple CSV manifest:

    path,label[,generator]
    /data/real/cat.jpg,0,camera
    /data/genimage/sd14/0001.png,1,stable_diffusion_v1.4

`label`: 0 = real, 1 = AI-generated. `generator` (optional) enables
per-generator evaluation breakdowns. `scripts/prepare_genimage.py` builds
manifests for the GenImage layout; any dataset works if you can write a CSV.
"""

from __future__ import annotations

import csv
import random
import warnings
from collections.abc import Callable
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from ..preprocessing import extract_crops, load_image_equalized
from .augment import RobustnessAugment

_MAX_LOAD_ATTEMPTS = 10


def read_manifest(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = {"path", "label"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest {path} is missing column(s): {sorted(missing)}")
        for line, row in enumerate(reader, start=2):
            label = (row["label"] or "").strip()
            if label not in ("0", "1"):
                raise ValueError(
                    f"{path}:{line}: label must be 0 (real) or 1 (AI), got {row['label']!r}"
                )
            rows.append(
                {
                    "path": row["path"],
                    "label": int(label),
                    "generator": (row.get("generator") or "").strip() or "unknown",
                }
            )
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def require_both_classes(rows: list[dict], what: str) -> None:
    labels = {r["label"] for r in rows}
    if labels != {0, 1}:
        raise ValueError(
            f"{what} must contain both real (0) and AI (1) rows; found only {sorted(labels)}."
        )


class ManifestDataset(Dataset):
    """`augment=True` (training): transport-chain augmentation + one random
    crop; unreadable images are skipped with a warning so one corrupt file
    can't kill a multi-hour run. `augment=False` (eval): one center crop, or —
    with `max_crops` — the exact native-res crop grid `Detector.predict` uses
    (item key "crops", shape (K,3,S,S), K varies per image so use
    `batch_size=None`); unreadable images raise, since silently dropping
    rows would change reported metrics."""

    def __init__(
        self,
        manifest: str | Path,
        crop_size: int = 256,
        augment: bool = False,
        semantic_preprocess: Callable | None = None,
        max_crops: int | None = None,
        equalize_jpeg_quality: int | None = None,
    ):
        if augment and max_crops is not None:
            raise ValueError("max_crops (multi-crop eval) is incompatible with augment=True")
        self.rows = read_manifest(manifest)
        self.crop_size = crop_size
        self.aug = RobustnessAugment() if augment else None
        self.semantic_preprocess = semantic_preprocess
        self.max_crops = max_crops
        # one JPEG pass for lossless sources: both classes share a compression history
        self.equalize_jpeg_quality = equalize_jpeg_quality

    def __len__(self) -> int:
        return len(self.rows)

    def _random_crop(self, img: Image.Image) -> torch.Tensor:
        s = self.crop_size
        if min(img.size) < s:
            img = TF.resize(img, s, antialias=True)
        w, h = img.size
        x = random.randint(0, w - s) if self.aug else (w - s) // 2
        y = random.randint(0, h - s) if self.aug else (h - s) // 2
        return TF.to_tensor(img.crop((x, y, x + s, y + s)))

    def _load(self, idx: int) -> tuple[dict, Image.Image]:
        if self.aug is None:
            row = self.rows[idx]
            try:
                return row, load_image_equalized(row["path"], self.equalize_jpeg_quality)
            except (OSError, Image.DecompressionBombError) as e:
                raise OSError(f"Cannot read image {row['path']}: {e}") from e
        for _ in range(_MAX_LOAD_ATTEMPTS):
            row = self.rows[idx]
            try:
                return row, load_image_equalized(row["path"], self.equalize_jpeg_quality)
            except (OSError, Image.DecompressionBombError) as e:
                warnings.warn(
                    f"skipping unreadable training image {row['path']}: {e}", stacklevel=2
                )
                idx = random.randrange(len(self.rows))
        raise OSError(f"{_MAX_LOAD_ATTEMPTS} consecutive unreadable images — check the manifest.")

    def __getitem__(self, idx: int) -> dict:
        row, img = self._load(idx)
        if self.aug is not None:
            img = self.aug(img)
        item: dict = {"label": torch.tensor(row["label"], dtype=torch.float32)}
        if self.max_crops is not None:
            item["crops"] = extract_crops(img, self.crop_size, self.max_crops)
        else:
            item["crop"] = self._random_crop(img)
        if self.semantic_preprocess is not None:
            item["semantic"] = self.semantic_preprocess(img).squeeze(0)
        return item
