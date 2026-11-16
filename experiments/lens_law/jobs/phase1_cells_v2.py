"""Kaggle entry (GPU): bounded check of the two remaining levers, dev photos only.

For 150 sharp and 150 wide-aperture development Unsplash photos it stores, per
32-pixel cell: the blur network's reading at three image scales, and inverse
depth from three depth models (the shipped small one, a larger one of the
same family, and a metric model of a different family). No images leave the
notebook.
"""

import csv
import json
import os

import numpy as np
import torch
from lenslaw.blur_net import STRIDE, BlurNet, predict_cells
from lenslaw.depth import normalize_inverse_depth
from lenslaw.score import load_rgb
from PIL import Image

OUT = "/kaggle/working"
N_UNSPLASH = 150
SCALES = (1, 2, 4)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEPTH_MODELS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
    "pro": "apple/DepthPro-hf",
}
COLUMNS = ["photo", "is_wide", "row", "col"]
COLUMNS += [f"{k}_{name}" for name in DEPTH_MODELS for k in ("d", "spread")]
COLUMNS += [f"{k}{s}" for s in SCALES for k in ("b", "std")]


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


def load_depth_model(model_id: str):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    processor = AutoImageProcessor.from_pretrained(model_id)
    return processor, AutoModelForDepthEstimation.from_pretrained(model_id).to(DEVICE).eval()


def inverse_depth_from(loaded, img: Image.Image, metric: bool) -> np.ndarray:
    processor, model = loaded
    inputs = processor(images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        pred = model(**inputs).predicted_depth
    pred = pred if pred.dim() == 3 else pred[:, 0]
    pred = (
        torch.nn.functional.interpolate(
            pred[:, None].float(), size=(img.height, img.width), mode="bicubic", align_corners=False
        )[0, 0]
        .cpu()
        .numpy()
    )
    if metric:  # metric depth in metres -> inverse depth
        pred = 1.0 / np.clip(pred, 1e-3, None)
    return normalize_inverse_depth(pred)


depth_models, depth_errors = {}, {}
for name, model_id in DEPTH_MODELS.items():
    try:
        depth_models[name] = load_depth_model(model_id)
    except Exception as e:  # noqa: BLE001
        depth_errors[name] = f"{type(e).__name__}: {e}"[:300]
print("depth models:", list(depth_models), "errors:", depth_errors, flush=True)

net = BlurNet()
net.load_state_dict(torch.load(find_file("blur_net.pt"), map_location="cpu", weights_only=True))
net = net.to(DEVICE).eval()

manifest_path = find_file("unsplash_manifest.csv")
unsplash_dir = os.path.join(os.path.dirname(manifest_path), "unsplash")
with open(manifest_path, encoding="utf-8") as f:
    manifest = [r for r in csv.DictReader(f) if r["split"] == "dev"]

rows, photos = [], []
for kind, is_wide in (("sharp", 0), ("wide", 1)):
    for row in [r for r in manifest if r["set"] == kind][:N_UNSPLASH]:
        rgb = load_rgb(os.path.join(unsplash_dir, row["photo_id"] + ".jpg"))
        pil = to_pil(rgb)
        ny, nx = rgb.shape[0] // STRIDE, rgb.shape[1] // STRIDE
        iy, ix = np.divmod(np.arange(ny * nx), nx)
        cols = [np.full(ny * nx, len(photos)), np.full(ny * nx, is_wide), iy, ix]
        for name in DEPTH_MODELS:
            try:
                inv = inverse_depth_from(depth_models[name], pil, metric=name == "pro")
                cols += list(depth_cells(inv))
            except Exception as e:  # noqa: BLE001
                depth_errors.setdefault(name, f"{type(e).__name__}: {e}"[:300])
                cols += [np.full(ny * nx, np.nan), np.full(ny * nx, np.nan)]
        for s in SCALES:
            small = pil if s == 1 else pil.resize((pil.width // s, pil.height // s), Image.LANCZOS)
            b, std = predict_cells(net, np.asarray(small, dtype=np.float32) / 255.0, DEVICE)
            grid_b = np.full((ny, nx), np.nan, dtype=np.float32)
            grid_s = np.full((ny, nx), np.nan, dtype=np.float32)
            up_b = np.repeat(np.repeat(b, s, axis=0), s, axis=1)[:ny, :nx]
            up_s = np.repeat(np.repeat(std, s, axis=0), s, axis=1)[:ny, :nx]
            grid_b[: up_b.shape[0], : up_b.shape[1]] = up_b + np.log(s)  # blur in full-size pixels
            grid_s[: up_s.shape[0], : up_s.shape[1]] = up_s
            cols += [grid_b.ravel(), grid_s.ravel()]
        rows.append(np.stack(cols, axis=1).astype(np.float32))
        photos.append(row["photo_id"])
    print(f"{kind} done", flush=True)

np.savez_compressed(os.path.join(OUT, "cells_v2.npz"), cells=np.concatenate(rows))
with open(os.path.join(OUT, "cells_v2_meta.json"), "w") as f:
    json.dump({"columns": COLUMNS, "photos": photos, "depth_errors": depth_errors}, f)
print("photos", len(photos), "depth errors", depth_errors)
