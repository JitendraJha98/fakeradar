"""Benchmark + robustness harness.

`fakeradar benchmark manifest.csv --checkpoint best.pt`
    -> per-generator AUROC/AP/acc table (the cross-generator generalization
       view that actually matters), exported as markdown for the leaderboard.

`fakeradar robustness manifest.csv --checkpoint best.pt`
    -> the same model re-evaluated under JPEG q∈{90,70,50,30}, downscale
       {0.75, 0.5}, and blur — the "will it survive WhatsApp" table.
"""

from __future__ import annotations

import io
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageFilter

from ..detector import Detector
from ..training.datasets import read_manifest
from .metrics import summarize

PERTURBATIONS = {
    "clean": lambda im: im,
    "jpeg90": lambda im: _jpeg(im, 90),
    "jpeg70": lambda im: _jpeg(im, 70),
    "jpeg50": lambda im: _jpeg(im, 50),
    "jpeg30": lambda im: _jpeg(im, 30),
    "down0.75": lambda im: _scale(im, 0.75),
    "down0.5": lambda im: _scale(im, 0.5),
    "blur1.0": lambda im: im.filter(ImageFilter.GaussianBlur(1.0)),
}


def _jpeg(im: Image.Image, q: int) -> Image.Image:
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _scale(im: Image.Image, s: float) -> Image.Image:
    w, h = im.size
    return im.resize((max(64, int(w * s)), max(64, int(h * s))), Image.BILINEAR)


def _to_markdown(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def run_benchmark(detector: Detector, manifest: str | Path, out_md: str | Path | None = None) -> str:
    rows = read_manifest(manifest)
    by_gen: dict[str, tuple[list[float], list[int]]] = defaultdict(lambda: ([], []))
    all_scores, all_labels = [], []

    for row in rows:
        prob = detector.predict(row["path"]).prob_ai
        all_scores.append(prob)
        all_labels.append(row["label"])
        if row["label"] == 1:
            by_gen[row["generator"]][0].append(prob)
            by_gen[row["generator"]][1].append(1)

    reals = [(s, y) for s, y in zip(all_scores, all_labels) if y == 0]
    table = []
    for gen, (scores, labels) in sorted(by_gen.items()):
        s = scores + [r[0] for r in reals]
        y = labels + [r[1] for r in reals]
        m = summarize(y, s)
        table.append([gen, len(scores), f"{m['auroc']:.4f}", f"{m['ap']:.4f}", f"{m['acc@0.5']:.4f}"])
    overall = summarize(all_labels, all_scores)
    table.append(["ALL", len(all_scores), f"{overall['auroc']:.4f}", f"{overall['ap']:.4f}", f"{overall['acc@0.5']:.4f}"])

    md = _to_markdown(["generator", "n_fake", "AUROC", "AP", "acc@0.5"], table)
    if out_md:
        Path(out_md).write_text(md + "\n")
    return md


def run_robustness(detector: Detector, manifest: str | Path, out_md: str | Path | None = None) -> str:
    rows = read_manifest(manifest)
    table = []
    for name, fn in PERTURBATIONS.items():
        scores, labels = [], []
        for row in rows:
            img = fn(Image.open(row["path"]).convert("RGB"))
            scores.append(detector.predict(img).prob_ai)
            labels.append(row["label"])
        m = summarize(labels, scores)
        table.append([name, f"{m['auroc']:.4f}", f"{m['acc@0.5']:.4f}", f"{m['tpr@5%fpr']:.4f}"])

    md = _to_markdown(["perturbation", "AUROC", "acc@0.5", "TPR@5%FPR"], table)
    if out_md:
        Path(out_md).write_text(md + "\n")
    return md
