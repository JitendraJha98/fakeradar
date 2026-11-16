"""Kaggle entry (CPU): dump per-cell blur and depth for development-split pairs.

For each iDFD dev scene and two cell sizes it stores, per grid cell: measured
and estimated inverse depth, the number of blur estimates, and low/median
quantiles of log blur for the defocused and the all-in-focus shot. The arrays
are small, contain no images, and let the cell statistic be studied locally.
"""

import json
import os

import numpy as np
from lenslaw.blur_classic import estimate_blur
from lenslaw.depth import inverse_depth, load_depth, normalize_inverse_depth
from lenslaw.score import load_rgb, to_gray
from lenslaw.splits import assign_split
from PIL import Image

OUT = "/kaggle/working"
N_SCENES = 120
CELLS = (32, 64)
QUANTILES = (10, 25, 50)
COLUMNS = [
    "scene",
    "cell",
    "row",
    "col",
    "d_gt",
    "spread_gt",
    "d_mono_oof",
    "d_mono_aif",
    "n_oof",
    "n_aif",
    *[f"q{q}_oof" for q in QUANTILES],
    *[f"q{q}_aif" for q in QUANTILES],
]


def find_dir(*parts: str) -> str:
    tail = os.path.join(*parts)
    for base, _dirs, _files in os.walk("/kaggle/input"):
        if base.endswith(tail):
            return base
    raise FileNotFoundError(tail)


def blocks(a: np.ndarray, cell: int) -> np.ndarray:
    ny, nx = a.shape[0] // cell, a.shape[1] // cell
    a = a[: ny * cell, : nx * cell].reshape(ny, cell, nx, cell)
    return a.transpose(0, 2, 1, 3).reshape(ny * nx, cell * cell)


def log_quantiles(sigma: np.ndarray, cell: int) -> tuple[np.ndarray, list[np.ndarray]]:
    logs = np.log(blocks(sigma, cell))
    count = np.isfinite(logs).sum(axis=1)
    safe = np.where(count[:, None] > 0, logs, 0.0)
    with np.errstate(all="ignore"):
        return count, [np.nanpercentile(safe, q, axis=1) for q in QUANTILES]


def to_pil(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8))


idfd = find_dir("Raw", "inpainting")
names = sorted(os.listdir(os.path.join(idfd, "All_In_Focus")))
names = [n for n in names if assign_split(os.path.splitext(n)[0].rsplit("_", 1)[0]) == "dev"]
names = names[:: max(1, len(names) // N_SCENES)][:N_SCENES]
loaded = load_depth("cpu")

rows, scenes = [], []
for i, name in enumerate(names):
    stem = os.path.splitext(name)[0]
    aif = load_rgb(os.path.join(idfd, "All_In_Focus", name))
    oof = load_rgb(os.path.join(idfd, "Out_Of_Focus", name))
    gt = np.asarray(Image.open(os.path.join(idfd, "Depth", stem + ".png")), dtype=np.float32)
    valid = gt > 0
    d_gt = normalize_inverse_depth(1.0 / np.where(valid, gt, np.median(gt[valid])))
    d_oof = inverse_depth(loaded, to_pil(oof))
    d_aif = inverse_depth(loaded, to_pil(aif))
    s_oof = estimate_blur(to_gray(oof)).sigma
    s_aif = estimate_blur(to_gray(aif)).sigma
    scenes.append(
        {
            "scene": stem,
            "median_sigma_oof": float(np.nanmedian(s_oof)),
            "median_sigma_aif": float(np.nanmedian(s_aif)),
            "gt_invalid": float(1 - valid.mean()),
        }
    )
    for cell in CELLS:
        g = blocks(d_gt, cell)
        n_oof, q_oof = log_quantiles(s_oof, cell)
        n_aif, q_aif = log_quantiles(s_aif, cell)
        ny, nx = d_gt.shape[0] // cell, d_gt.shape[1] // cell
        iy, ix = np.divmod(np.arange(ny * nx), nx)
        table = np.stack(
            [
                np.full(ny * nx, i),
                np.full(ny * nx, cell),
                iy,
                ix,
                np.median(g, axis=1),
                np.percentile(g, 90, axis=1) - np.percentile(g, 10, axis=1),
                np.median(blocks(d_oof, cell), axis=1),
                np.median(blocks(d_aif, cell), axis=1),
                n_oof,
                n_aif,
                *q_oof,
                *q_aif,
            ],
            axis=1,
        )
        rows.append(table[(n_oof > 0) | (n_aif > 0)].astype(np.float32))
    if i % 20 == 0:
        print(f"{i}/{len(names)} {stem}", flush=True)

cells = np.concatenate(rows)
np.savez_compressed(os.path.join(OUT, "cells.npz"), cells=cells)
with open(os.path.join(OUT, "cells_meta.json"), "w") as f:
    json.dump({"columns": COLUMNS, "scenes": scenes}, f, indent=1)
print("cells:", cells.shape, "scenes:", len(scenes))
