"""Kaggle entry (CPU): why does the blur estimator not show a V on real pairs?

Development-split iDFD scenes only. For each scene it compares the blur map
of the defocused shot with that of its all-in-focus twin, cell by cell, using
the measured (Kinect) depth, and tries several ways of summarising a cell.
Writes diag.json and a few figures.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from lenslaw.blur_classic import estimate_blur
from lenslaw.depth import normalize_inverse_depth
from lenslaw.fit import fit_thin_lens
from lenslaw.score import load_rgb, to_gray
from lenslaw.splits import assign_split
from PIL import Image

OUT = "/kaggle/working"
CELL = 32
N_SCENES = 48
N_FIGURES = 8
QUANTILES = (10, 25, 50)


def find_dir(*parts: str) -> str:
    tail = os.path.join(*parts)
    for base, _dirs, _files in os.walk("/kaggle/input"):
        if base.endswith(tail):
            return base
    raise FileNotFoundError(tail)


def blocks(a: np.ndarray) -> np.ndarray:
    ny, nx = a.shape[0] // CELL, a.shape[1] // CELL
    a = a[: ny * CELL, : nx * CELL].reshape(ny, CELL, nx, CELL)
    return a.transpose(0, 2, 1, 3).reshape(ny * nx, CELL * CELL)


def cell_stats(sigma: np.ndarray) -> dict:
    """Per cell: count of estimates and low/median quantiles of log blur."""
    logs = np.log(blocks(sigma))
    count = np.isfinite(logs).sum(axis=1)
    out = {"count": count}
    with np.errstate(all="ignore"):
        for q in QUANTILES:
            out[f"q{q}"] = np.nanpercentile(np.where(count[:, None] > 0, logs, 0.0), q, axis=1)
    return out


def r2_of(d, b, ok) -> float:
    if ok.sum() < 40:
        return float("nan")
    return fit_thin_lens(d[ok], b[ok], np.ones(int(ok.sum()))).r2


idfd = find_dir("Raw", "inpainting")
names = sorted(os.listdir(os.path.join(idfd, "All_In_Focus")))
names = [n for n in names if assign_split(os.path.splitext(n)[0].rsplit("_", 1)[0]) == "dev"]
names = names[:: max(1, len(names) // N_SCENES)][:N_SCENES]

scenes = []
for i, name in enumerate(names):
    stem = os.path.splitext(name)[0]
    aif = load_rgb(os.path.join(idfd, "All_In_Focus", name))
    oof = load_rgb(os.path.join(idfd, "Out_Of_Focus", name))
    gt = np.asarray(Image.open(os.path.join(idfd, "Depth", stem + ".png")), dtype=np.float32)
    valid = gt > 0
    inv = normalize_inverse_depth(1.0 / np.where(valid, gt, np.median(gt[valid])))
    s_aif = estimate_blur(to_gray(aif)).sigma
    s_oof = estimate_blur(to_gray(oof)).sigma
    c_aif, c_oof = cell_stats(s_aif), cell_stats(s_oof)
    depth_cells = blocks(inv)
    d = np.median(depth_cells, axis=1)
    flat = (np.percentile(depth_cells, 90, axis=1) - np.percentile(depth_cells, 10, axis=1)) <= 0.10
    both = flat & (c_aif["count"] >= 12) & (c_oof["count"] >= 12)

    row = {
        "scene": stem,
        "cells_both": int(both.sum()),
        "median_sigma_aif": float(np.nanmedian(s_aif)),
        "median_sigma_oof": float(np.nanmedian(s_oof)),
        "edge_px_aif": int(np.isfinite(s_aif).sum()),
        "edge_px_oof": int(np.isfinite(s_oof).sum()),
        "depth_spread": float(np.percentile(d, 90) - np.percentile(d, 10)),
    }
    for q in QUANTILES:
        b_aif, b_oof = c_aif[f"q{q}"], c_oof[f"q{q}"]
        row[f"r2_oof_q{q}"] = r2_of(d, b_oof, both)
        row[f"r2_aif_q{q}"] = r2_of(d, b_aif, both)
        # blur added by the wide aperture, with each cell's own edge softness removed
        added = np.sqrt(np.maximum(np.exp(2 * b_oof) - np.exp(2 * b_aif), 0.05**2))
        row[f"r2_added_q{q}"] = r2_of(d, np.log(added), both)
        row[f"spread_oof_q{q}"] = float(
            np.exp(np.percentile(b_oof[both], 90) - np.percentile(b_oof[both], 10))
            if both.sum() > 10
            else np.nan
        )
        row[f"spread_aif_q{q}"] = float(
            np.exp(np.percentile(b_aif[both], 90) - np.percentile(b_aif[both], 10))
            if both.sum() > 10
            else np.nan
        )
        row[f"median_ratio_oof_over_aif_q{q}"] = float(
            np.exp(np.median(b_oof[both] - b_aif[both])) if both.sum() > 10 else np.nan
        )
    scenes.append(row)

    if i % max(1, len(names) // N_FIGURES) == 0:
        fig, ax = plt.subplots(2, 4, figsize=(22, 11))
        ax[0, 0].imshow(aif)
        ax[0, 0].set_title(f"{stem}: all-in-focus")
        ax[0, 1].imshow(oof)
        ax[0, 1].set_title("out-of-focus (wide aperture)")
        ax[0, 2].imshow(inv, cmap="viridis")
        ax[0, 2].set_title("measured inverse depth (near = bright)")
        crop = slice(400, 656)
        ax[0, 3].imshow(np.concatenate([aif[crop, crop], oof[crop, crop]], axis=1))
        ax[0, 3].set_title("centre crop: sharp | defocused")
        for k, (label, s) in enumerate((("all-in-focus", s_aif), ("out-of-focus", s_oof))):
            dense = np.full_like(s, np.nan)
            ys, xs = np.nonzero(np.isfinite(s))
            dense[ys, xs] = s[ys, xs]
            im = ax[1, k].imshow(
                np.log2(dense), cmap="magma", vmin=-1, vmax=4, interpolation="nearest"
            )
            ax[1, k].set_title(f"log2 blur at edges: {label}")
            plt.colorbar(im, ax=ax[1, k], fraction=0.046)
        ax[1, 2].scatter(d[both], c_aif["q25"][both], s=8, label="all-in-focus", alpha=0.6)
        ax[1, 2].scatter(d[both], c_oof["q25"][both], s=8, label="out-of-focus", alpha=0.6)
        ax[1, 2].set_xlabel("inverse depth (cell median)")
        ax[1, 2].set_ylabel("log blur (cell 25th percentile)")
        ax[1, 2].legend()
        ax[1, 2].set_title("blur vs depth, per cell")
        added = np.sqrt(np.maximum(np.exp(2 * c_oof["q25"]) - np.exp(2 * c_aif["q25"]), 0.05**2))
        ax[1, 3].scatter(d[both], np.log(added[both]), s=8, color="k", alpha=0.6)
        ax[1, 3].set_xlabel("inverse depth (cell median)")
        ax[1, 3].set_ylabel("log added blur")
        ax[1, 3].set_title("blur added by the wide aperture, per cell")
        for a in ax[0]:
            a.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"diag_{stem}.png"), dpi=60)
        plt.close(fig)


def med(key: str) -> float:
    vals = [s[key] for s in scenes if np.isfinite(s[key])]
    return float(np.median(vals)) if vals else float("nan")


summary = {k: med(k) for k in scenes[0] if k != "scene"}
print(json.dumps(summary, indent=1))
with open(os.path.join(OUT, "diag.json"), "w") as f:
    json.dump({"summary": summary, "scenes": scenes}, f, indent=1)
