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
from collections.abc import Callable
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from .augment import RobustnessAugment


def read_manifest(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "path": row["path"],
                    "label": int(row["label"]),
                    "generator": row.get("generator", "unknown"),
                }
            )
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


class ManifestDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        crop_size: int = 256,
        augment: bool = False,
        semantic_preprocess: Callable | None = None,
    ):
        self.rows = read_manifest(manifest)
        self.crop_size = crop_size
        self.aug = RobustnessAugment() if augment else None
        self.semantic_preprocess = semantic_preprocess

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

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        img = Image.open(row["path"]).convert("RGB")
        if self.aug is not None:
            img = self.aug(img)
        item = {
            "crop": self._random_crop(img),
            "label": torch.tensor(row["label"], dtype=torch.float32),
        }
        if self.semantic_preprocess is not None:
            item["semantic"] = self.semantic_preprocess(img).squeeze(0)
        return item
