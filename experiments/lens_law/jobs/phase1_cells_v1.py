"""Kaggle entry (GPU): per-cell predictions of blur estimator v1 on development pairs.

For each iDFD dev scene it stores, per 32-pixel cell: measured and estimated
inverse depth, and the network's log blur and uncertainty for the defocused
and the all-in-focus shot. The same numbers for 150 sharp and 150
wide-aperture dev Unsplash photos are stored too, so the applicability rule
can be tuned. No images leave the notebook.
"""

import csv
import json
import os

import numpy as np
import torch
from lenslaw.blur_net import STRIDE, BlurNet, predict_cells
from lenslaw.depth import inverse_depth, load_depth, normalize_inverse_depth
from lenslaw.score import load_rgb
from lenslaw.splits import assign_split
from PIL import Image

OUT = "/kaggle/working"
BLUR_WEIGHTS = "blur_net.pt"
N_UNSPLASH = 150
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PAIR_COLUMNS = [
    "scene",
    "row",
    "col",
    "d_gt",
    "spread_gt",
    "d_mono_oof",
    "b_oof",
    "std_oof",
    "b_aif",
    "std_aif",
]
PHOTO_COLUMNS = ["photo", "is_wide", "row", "col", "d_mono", "spread_mono", "b", "std"]


def find_dir(*parts: str) -> str:
    tail = os.path.join(*parts)
    for base, _dirs, _files in os.walk("/kaggle/input"):
        if base.endswith(tail):
            return base
    raise FileNotFoundError(tail)


def find_file(name: str) -> str:
    for base, _dirs, files in os.walk("/kaggle/input"):
        if name in files:
            return os.path.join(base, name)
    raise FileNotFoundError(name)


def blocks(a: np.ndarray) -> np.ndarray:
    ny, nx = a.shape[0] // STRIDE, a.shape[1] // STRIDE
    a = a[: ny * STRIDE, : nx * STRIDE].reshape(ny, STRIDE, nx, STRIDE)
    return a.transpose(0, 2, 1, 3).reshape(ny * nx, STRIDE * STRIDE)


def to_pil(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8))


def depth_cells(inv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.percentile(inv, [2, 98])
    b = blocks(inv)
    spread = (np.percentile(b, 90, axis=1) - np.percentile(b, 10, axis=1)) / max(
        float(hi - lo), 1e-12
    )
    return np.median(b, axis=1), spread


model = BlurNet()
model.load_state_dict(torch.load(find_file(BLUR_WEIGHTS), map_location="cpu", weights_only=True))
model = model.to(DEVICE).eval()
loaded = load_depth(DEVICE)

# ---- iDFD development pairs ----
idfd = find_dir("Raw", "inpainting")
names = sorted(os.listdir(os.path.join(idfd, "All_In_Focus")))
names = [n for n in names if assign_split(os.path.splitext(n)[0].rsplit("_", 1)[0]) == "dev"]
pair_rows, scenes = [], []
for i, name in enumerate(names):
    stem = os.path.splitext(name)[0]
    aif = load_rgb(os.path.join(idfd, "All_In_Focus", name))
    oof = load_rgb(os.path.join(idfd, "Out_Of_Focus", name))
    gt = np.asarray(Image.open(os.path.join(idfd, "Depth", stem + ".png")), dtype=np.float32)
    valid = gt > 0
    d_gt, spread_gt = depth_cells(
        normalize_inverse_depth(1.0 / np.where(valid, gt, np.median(gt[valid])))
    )
    d_mono, _ = depth_cells(inverse_depth(loaded, to_pil(oof)))
    b_oof, s_oof = predict_cells(model, oof, DEVICE)
    b_aif, s_aif = predict_cells(model, aif, DEVICE)
    ny, nx = b_oof.shape
    iy, ix = np.divmod(np.arange(ny * nx), nx)
    pair_rows.append(
        np.stack(
            [
                np.full(ny * nx, i),
                iy,
                ix,
                d_gt,
                spread_gt,
                d_mono,
                b_oof.ravel(),
                s_oof.ravel(),
                b_aif.ravel(),
                s_aif.ravel(),
            ],
            axis=1,
        ).astype(np.float32)
    )
    scenes.append(stem)
    if i % 50 == 0:
        print(f"pairs {i}/{len(names)}", flush=True)

# ---- Unsplash development photos: sharp (f/8+) and wide-aperture ----
manifest_path = find_file("unsplash_manifest.csv")
unsplash_dir = os.path.join(os.path.dirname(manifest_path), "unsplash")
with open(manifest_path, encoding="utf-8") as f:
    manifest = [r for r in csv.DictReader(f) if r["split"] == "dev"]
photo_rows, photos = [], []
for kind, is_wide in (("sharp", 0), ("wide", 1)):
    for row in [r for r in manifest if r["set"] == kind][:N_UNSPLASH]:
        rgb = load_rgb(os.path.join(unsplash_dir, row["photo_id"] + ".jpg"))
        d, spread = depth_cells(inverse_depth(loaded, to_pil(rgb)))
        b, s = predict_cells(model, rgb, DEVICE)
        ny, nx = b.shape
        iy, ix = np.divmod(np.arange(ny * nx), nx)
        photo_rows.append(
            np.stack(
                [
                    np.full(ny * nx, len(photos)),
                    np.full(ny * nx, is_wide),
                    iy,
                    ix,
                    d,
                    spread,
                    b.ravel(),
                    s.ravel(),
                ],
                axis=1,
            ).astype(np.float32)
        )
        photos.append(row["photo_id"])
    print(f"unsplash {kind} done", flush=True)

np.savez_compressed(
    os.path.join(OUT, "cells_v1.npz"),
    pairs=np.concatenate(pair_rows),
    photos=np.concatenate(photo_rows),
)
with open(os.path.join(OUT, "cells_v1_meta.json"), "w") as f:
    json.dump(
        {
            "pair_columns": PAIR_COLUMNS,
            "photo_columns": PHOTO_COLUMNS,
            "scenes": scenes,
            "photos": photos,
        },
        f,
    )
print("pairs", len(scenes), "photos", len(photos))
