"""Benchmark + robustness harness.

`fakeradar benchmark manifest.csv --checkpoint best.pt`
    -> per-generator AUROC/AP/acc/TPR@5%FPR/ECE table (the cross-generator
       generalization view that actually matters) plus the mean over
       generators, exported as markdown for the leaderboard.

`fakeradar robustness manifest.csv --checkpoint best.pt`
    -> the same model re-evaluated under JPEG q∈{90,70,50,30}, downscale
       {0.75, 0.5}, and blur — the "will it survive WhatsApp" table.

Both can write the raw per-image scores (`scores_csv`) so every published
number is reproducible and auditable (paper/INTEGRITY_CHECKLIST.md).

Per-generator rows score that generator's AI images against *all* real images
in the manifest (the pooled real set).
"""

from __future__ import annotations

import csv
import io
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageFilter

from ..detector import Detector
from ..preprocessing import load_image
from ..training.datasets import read_manifest
from .metrics import abstention_rate, summarize

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


def _fmt(v: float) -> str:
    return "n/a" if math.isnan(v) else f"{v:.4f}"


def _abstainer(detector: Detector):
    cfg = detector.config
    return lambda s: abstention_rate(s, cfg.threshold_real, cfg.threshold_fake)


def _abstain_note(detector: Detector) -> str:
    cfg = detector.config
    return f"abstain = share of `uncertain` verdicts, P(AI) in ({cfg.threshold_real}, {cfg.threshold_fake})."


def _write_scores(path: str | Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def run_benchmark(
    detector: Detector,
    manifest: str | Path,
    out_md: str | Path | None = None,
    scores_csv: str | Path | None = None,
) -> str:
    rows = read_manifest(manifest)
    probs = [detector.predict(row["path"]).prob_ai for row in rows]
    labels = [row["label"] for row in rows]

    real_probs = [p for p, y in zip(probs, labels, strict=True) if y == 0]
    by_gen: dict[str, list[float]] = defaultdict(list)
    for row, p in zip(rows, probs, strict=True):
        if row["label"] == 1:
            by_gen[row["generator"]].append(p)

    abstain = _abstainer(detector)

    def metric_row(name: str, n_fake: int, y: list[int], s: list[float]) -> tuple[list, float]:
        m = summarize(y, s)
        keys = ("auroc", "ap", "acc@0.5", "tpr@5%fpr", "ece")
        return [name, n_fake, *(_fmt(m[k]) for k in keys), _fmt(abstain(s))], m["auroc"]

    table, gen_aurocs = [], []
    for gen, fake_probs in sorted(by_gen.items()):
        y = [1] * len(fake_probs) + [0] * len(real_probs)
        row, auroc = metric_row(gen, len(fake_probs), y, fake_probs + real_probs)
        table.append(row)
        gen_aurocs.append(auroc)
    table.append(metric_row("ALL", sum(labels), labels, probs)[0])
    valid = [a for a in gen_aurocs if not math.isnan(a)]
    if len(by_gen) > 1 and valid:
        mean = _fmt(sum(valid) / len(valid))
        table.append([f"MEAN ({len(valid)} gen)", "", mean, "", "", "", "", ""])

    headers = ["generator", "n_fake", "AUROC", "AP", "acc@0.5", "TPR@5%FPR", "ECE", "abstain"]
    md = _to_markdown(headers, table)
    md += f"\n\n{len(real_probs)} real images pooled against each generator. {_abstain_note(detector)}"
    if out_md:
        Path(out_md).write_text(md + "\n", encoding="utf-8")
    if scores_csv:
        _write_scores(
            scores_csv,
            ["path", "label", "generator", "prob_ai"],
            [
                [r["path"], r["label"], r["generator"], f"{p:.6f}"]
                for r, p in zip(rows, probs, strict=True)
            ],
        )
    return md


def run_robustness(
    detector: Detector,
    manifest: str | Path,
    out_md: str | Path | None = None,
    scores_csv: str | Path | None = None,
) -> str:
    rows = read_manifest(manifest)
    labels = [row["label"] for row in rows]
    probs: dict[str, list[float]] = {name: [] for name in PERTURBATIONS}
    for row in rows:
        img = load_image(row["path"])  # decode once, perturb many
        for name, fn in PERTURBATIONS.items():
            probs[name].append(detector.predict(fn(img)).prob_ai)

    abstain = _abstainer(detector)
    table = []
    for name in PERTURBATIONS:
        m = summarize(labels, probs[name])
        table.append(
            [
                name,
                _fmt(m["auroc"]),
                _fmt(m["acc@0.5"]),
                _fmt(m["tpr@5%fpr"]),
                _fmt(abstain(probs[name])),
            ]
        )

    md = _to_markdown(["perturbation", "AUROC", "acc@0.5", "TPR@5%FPR", "abstain"], table)
    md += f"\n\n{_abstain_note(detector)}"
    if out_md:
        Path(out_md).write_text(md + "\n", encoding="utf-8")
    if scores_csv:
        _write_scores(
            scores_csv,
            ["path", "label", "generator", "perturbation", "prob_ai"],
            [
                [row["path"], row["label"], row["generator"], name, f"{probs[name][i]:.6f}"]
                for name in PERTURBATIONS
                for i, row in enumerate(rows)
            ],
        )
    return md
