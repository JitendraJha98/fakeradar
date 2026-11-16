"""Kaggle entry (GPU): Phase 1 scoring for Gate M.

Inputs: the iDFD dataset (real all-in-focus / defocused pairs with measured
depth) and the output of the phase1-data job (Unsplash photos + manifest).
Output: records.jsonl (one line per image, variant and condition), the split
files, and score_report.json.

Set LIMIT to a small number for a dry run; 0 scores everything.
"""

import collections
import csv
import json
import os
import time
import zlib

import numpy as np
import torch
from lenslaw.blur_net import BlurNet
from lenslaw.depth import inverse_depth, load_depth, normalize_inverse_depth
from lenslaw.render import MODES, blur_field, render_defocus
from lenslaw.score import load_rgb, score_arrays, to_gray
from lenslaw.splits import assign_split
from lenslaw.transport import CONDITIONS, apply_condition
from PIL import Image

LIMIT = 0
PARTS = ("pairs", "real_wide", "render")  # which sections this job scores
SHARD = (0, 1)  # (k, n): this job scores items whose index is k modulo n
# Which splits this job scores. Test stays unscored until the method is frozen.
SPLITS = ("dev", "test")
SIGMA_MAX = 4.0
# Weights file of the learned blur estimator among the inputs; empty = estimator v0.
BLUR_WEIGHTS = ""
OUT = "/kaggle/working"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


def to_pil(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8))


def small(a: np.ndarray, n: int = 64) -> np.ndarray:
    t = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))[None, None]
    return torch.nn.functional.adaptive_avg_pool2d(t, n).flatten().numpy()


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = small(a), small(b)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


idfd = find_dir("Raw", "inpainting")
manifest_path = find_file("unsplash_manifest.csv")
unsplash_dir = os.path.join(os.path.dirname(manifest_path), "unsplash")
with open(manifest_path, encoding="utf-8") as f:
    manifest = list(csv.DictReader(f))

loaded = load_depth(DEVICE)
blur_model = None
if BLUR_WEIGHTS:
    blur_model = BlurNet()
    state = torch.load(find_file(BLUR_WEIGHTS), map_location="cpu", weights_only=True)
    blur_model.load_state_dict(state)
    blur_model = blur_model.to(DEVICE).eval()
counts = collections.Counter()
errors = []
t_start = time.time()
fh = open(os.path.join(OUT, "records.jsonl"), "w", encoding="utf-8")  # noqa: SIM115


def depth_of(rgb: np.ndarray) -> np.ndarray:
    return inverse_depth(loaded, to_pil(rgb))


def record(meta: dict, rgb: np.ndarray, depth: np.ndarray, extra: dict | None = None) -> None:
    out = dict(meta)
    out.update(score_arrays(rgb, depth, device=DEVICE, blur_model=blur_model))
    out["height"], out["width"] = rgb.shape[:2]
    if extra:
        out.update(extra)
    fh.write(json.dumps(out) + "\n")
    fh.flush()
    counts[(meta["source"], meta["variant"], meta["condition"])] += 1


def guarded(label: str, fn, *args) -> None:
    try:
        fn(*args)
    except Exception as e:  # noqa: BLE001
        counts["errors"] += 1
        if len(errors) < 20:
            errors.append(f"{label}: {type(e).__name__}: {e}")


def progress(label: str, i: int, n: int) -> None:
    if i % 100 == 0:
        print(f"[{int(time.time() - t_start)}s] {label} {i}/{n}", flush=True)


# ------------------------------------------------------------ aperture pairs
def score_pair(name: str) -> None:
    stem = os.path.splitext(name)[0]
    aif = load_rgb(os.path.join(idfd, "All_In_Focus", name))
    oof = load_rgb(os.path.join(idfd, "Out_Of_Focus", name))
    if aif.shape != oof.shape:
        raise ValueError(f"pair shapes differ: {aif.shape} vs {oof.shape}")
    d_aif, d_oof = depth_of(aif), depth_of(oof)
    gt = np.asarray(Image.open(os.path.join(idfd, "Depth", stem + ".png")), dtype=np.float32)
    valid = gt > 0
    fill = float(np.median(gt[valid])) if valid.any() else 1.0
    d_gt = normalize_inverse_depth(1.0 / np.where(valid, gt, fill))
    meta = {
        "id": stem,
        "source": "pairs",
        "split": assign_split(stem.rsplit("_", 1)[0]),
        "scene": stem,
        "condition": "clean",
    }
    diag = {
        "align_corr": corr(to_gray(aif), to_gray(oof)),
        "depth_corr_narrow_gt": corr(d_aif, d_gt),
        "depth_corr_wide_gt": corr(d_oof, d_gt),
        "depth_corr_wide_narrow": corr(d_oof, d_aif),
        "gt_invalid": float(1.0 - valid.mean()),
    }
    record({**meta, "variant": "wide_depth_narrow"}, oof, d_aif, diag)
    record({**meta, "variant": "wide_depth_self"}, oof, d_oof)
    record({**meta, "variant": "wide_depth_gt"}, oof, d_gt)
    record({**meta, "variant": "narrow_depth_self"}, aif, d_aif)


names = sorted(os.listdir(os.path.join(idfd, "All_In_Focus")))
names = [n for n in names if assign_split(os.path.splitext(n)[0].rsplit("_", 1)[0]) in SPLITS]
names = names[:: max(1, len(names) // LIMIT)][:LIMIT] if LIMIT else names
names = names[SHARD[0] :: SHARD[1]] if "pairs" in PARTS else []
for i, name in enumerate(names):
    progress("pairs", i, len(names))
    guarded(name, score_pair, name)


# ------------------------------------------------- real wide-aperture photos
def score_photo(row: dict) -> None:
    img = Image.open(os.path.join(unsplash_dir, row["photo_id"] + ".jpg")).convert("RGB")
    for cond in CONDITIONS:
        rgb = load_rgb(apply_condition(img, cond))
        meta = {
            "id": row["photo_id"],
            "source": "real_wide",
            "variant": "photo",
            "split": row["split"],
            "scene": row["photo_id"],
            "condition": cond,
        }
        record(meta, rgb, depth_of(rgb), {"aperture": row["aperture"]})


wide = [r for r in manifest if r["set"] == "wide" and r["split"] in SPLITS]
wide = wide[:LIMIT] if LIMIT else wide
wide = wide[SHARD[0] :: SHARD[1]] if "real_wide" in PARTS else []
for i, row in enumerate(wide):
    progress("real_wide", i, len(wide))
    guarded(row["photo_id"], score_photo, row)


# ------------------------------------- controlled renders from sharp photos
def score_renders(row: dict) -> None:
    base = load_rgb(os.path.join(unsplash_dir, row["photo_id"] + ".jpg"))
    d_base = depth_of(base)
    rng = np.random.default_rng(zlib.crc32(row["photo_id"].encode()))
    focus_q = float(rng.uniform(0.3, 0.7))
    d_focus = float(np.quantile(d_base, focus_q))
    for mode in MODES:
        field = blur_field(d_base, mode, d_focus, SIGMA_MAX)
        rendered = to_pil(render_defocus(base, field, device=DEVICE))
        for cond in CONDITIONS:
            rgb = load_rgb(apply_condition(rendered, cond))
            meta = {
                "id": f"{row['photo_id']}:{mode}",
                "source": "render",
                "variant": mode,
                "split": row["split"],
                "scene": row["photo_id"],
                "condition": cond,
            }
            record(meta, rgb, depth_of(rgb), {"focus_q": focus_q})


sharp = [r for r in manifest if r["set"] == "sharp" and r["split"] in SPLITS]
sharp = sharp[:LIMIT] if LIMIT else sharp
sharp = sharp[SHARD[0] :: SHARD[1]] if "render" in PARTS else []
for i, row in enumerate(sharp):
    progress("render", i, len(sharp))
    guarded(row["photo_id"], score_renders, row)

fh.close()

# ------------------------------------------------------- splits and report
with open(os.path.join(OUT, "splits_pairs.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["scene", "split"])
    for name in sorted(os.listdir(os.path.join(idfd, "All_In_Focus"))):
        stem = os.path.splitext(name)[0]
        w.writerow([stem, assign_split(stem.rsplit("_", 1)[0])])
with open(os.path.join(OUT, "splits_unsplash.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["photo_id", "set", "split"])
    for row in manifest:
        w.writerow([row["photo_id"], row["set"], row["split"]])

report = {
    "device": DEVICE,
    "estimator": "v1" if BLUR_WEIGHTS else "v0",
    "limit": LIMIT,
    "parts": list(PARTS),
    "shard": list(SHARD),
    "splits": list(SPLITS),
    "seconds": int(time.time() - t_start),
    "pairs": len(names),
    "real_wide": len(wide),
    "sharp": len(sharp),
    "records": {"/".join(k): v for k, v in counts.items() if k != "errors"},
    "n_errors": counts["errors"],
    "errors": errors,
}
print(json.dumps(report, indent=1))
with open(os.path.join(OUT, "score_report.json"), "w") as f:
    json.dump(report, f, indent=1)
