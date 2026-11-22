"""Kaggle entry (GPU): Community Forensics on the same test images as fakeradar.

Re-evaluation, not retraining: the public weights (OwensLab/commfor-model-384,
MIT) were trained on thousands of generators, so this is a reference point,
not a same-training-data comparison. Uses the exact test and robustness
manifests saved by the fakeradar seed-42 run (kernel phase4-train-fast).
Writes benchmark_*.md, robustness_equalized.md and raw score CSVs in the same
format as fakeradar's harness.
"""

import csv
import io
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision.transforms import functional as TF

subprocess.run(
    [
        "git",
        "clone",
        "--depth",
        "1",
        "https://github.com/JeongsooP/Community-Forensics",
        "/tmp/commfor",
    ],
    check=True,
)
sys.path.insert(0, "/tmp/commfor")
from models import ViTClassifier  # noqa: E402

from fakeradar.evaluation.metrics import summarize  # noqa: E402
from fakeradar.preprocessing import load_image_equalized  # noqa: E402

OUT = "/kaggle/working"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EQUALIZE = 96
model = ViTClassifier.from_pretrained("OwensLab/commfor-model-384", device=DEVICE).to(DEVICE).eval()


def prep(img: Image.Image) -> torch.Tensor:
    img = TF.center_crop(TF.resize(img.convert("RGB"), 440), [384, 384])
    return TF.normalize(TF.to_tensor(img), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


@torch.no_grad()
def score(images: list[Image.Image]) -> list[float]:
    x = torch.stack([prep(im) for im in images]).to(DEVICE)
    return torch.sigmoid(model(x).flatten().float()).cpu().tolist()


def find(name: str) -> str:
    for base, _dirs, files in os.walk("/kaggle/input"):
        if name in files and "manifests" in base:
            return os.path.join(base, name)
    raise FileNotFoundError(name)


def read(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [
            {"path": r["path"], "label": int(r["label"]), "generator": r["generator"]}
            for r in csv.DictReader(f)
        ]


def fmt(v: float) -> str:
    return "n/a" if math.isnan(v) else f"{v:.4f}"


def benchmark(rows: list[dict], probs: list[float], tag: str) -> None:
    real = [p for p, r in zip(probs, rows, strict=True) if r["label"] == 0]
    by_gen = defaultdict(list)
    for r, p in zip(rows, probs, strict=True):
        if r["label"] == 1:
            by_gen[r["generator"]].append(p)
    lines = [
        "| generator | n_fake | AUROC | AP | acc@0.5 | TPR@5%FPR | ECE | abstain |",
        "|---|---|---|---|---|---|---|---|",
    ]
    aurocs = []
    for gen, fake in sorted(by_gen.items()):
        m = summarize([1] * len(fake) + [0] * len(real), fake + real)
        aurocs.append(m["auroc"])
        lines.append(
            f"| {gen} | {len(fake)} | "
            + " | ".join(fmt(m[k]) for k in ("auroc", "ap", "acc@0.5", "tpr@5%fpr", "ece"))
            + " | n/a |"
        )
    m = summarize([r["label"] for r in rows], probs)
    lines.append(
        f"| ALL | {len(rows) - len(real)} | "
        + " | ".join(fmt(m[k]) for k in ("auroc", "ap", "acc@0.5", "tpr@5%fpr", "ece"))
        + " | n/a |"
    )
    lines.append(f"| MEAN (8 gen) |  | {fmt(float(np.mean(aurocs)))} |  |  |  |  |  |")
    note = (
        "as stored" if tag == "raw" else f"non-JPEG images JPEG-encoded once at quality {EQUALIZE}"
    )
    lines.append(f"\nCommunity Forensics (OwensLab/commfor-model-384), re-evaluated; {note}.")
    with open(os.path.join(OUT, f"benchmark_{tag}.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT, f"benchmark_{tag}_scores.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "generator", "prob_ai"])
        w.writerows(
            [r["path"], r["label"], r["generator"], f"{p:.6f}"]
            for r, p in zip(rows, probs, strict=True)
        )


def jpeg(im, q):
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def scale(im, s):
    return im.resize((max(64, int(im.width * s)), max(64, int(im.height * s))), Image.BILINEAR)


PERTURB = {
    "clean": lambda im: im,
    "jpeg90": lambda im: jpeg(im, 90),
    "jpeg70": lambda im: jpeg(im, 70),
    "jpeg50": lambda im: jpeg(im, 50),
    "jpeg30": lambda im: jpeg(im, 30),
    "down0.75": lambda im: scale(im, 0.75),
    "down0.5": lambda im: scale(im, 0.5),
    "blur1.0": lambda im: im.filter(ImageFilter.GaussianBlur(1.0)),
}

report = {"device": DEVICE, "seconds": {}}
test = read(find("test.csv"))
for tag, quality in (("raw", None), ("equalized", EQUALIZE)):
    t0 = time.time()
    probs = []
    for i in range(0, len(test), 32):
        probs += score([load_image_equalized(r["path"], quality) for r in test[i : i + 32]])
    benchmark(test, probs, tag)
    report["seconds"][tag] = int(time.time() - t0)
    print(tag, "done", report["seconds"][tag], "s", flush=True)

robust = read(find("robust.csv"))
labels = [r["label"] for r in robust]
lines = ["| perturbation | AUROC | acc@0.5 | TPR@5%FPR | abstain |", "|---|---|---|---|---|"]
for name, fn in PERTURB.items():
    probs = []
    for i in range(0, len(robust), 32):
        probs += score([fn(load_image_equalized(r["path"], EQUALIZE)) for r in robust[i : i + 32]])
    m = summarize(labels, probs)
    lines.append(
        f"| {name} | {fmt(m['auroc'])} | {fmt(m['acc@0.5'])} | {fmt(m['tpr@5%fpr'])} | n/a |"
    )
with open(os.path.join(OUT, "robustness_equalized.md"), "w") as f:
    f.write("\n".join(lines) + "\n")
with open(os.path.join(OUT, "baseline_report.json"), "w") as f:
    json.dump(report, f, indent=1)
print(open(os.path.join(OUT, "benchmark_equalized.md")).read())
