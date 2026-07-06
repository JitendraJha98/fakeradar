#!/usr/bin/env python3
"""Build fakeradar CSV manifests from a GenImage-style directory tree.

Expected layout (per GenImage release):
    ROOT/<generator>/<split>/{ai,nature}/*.png|jpg
Usage:
    python scripts/prepare_genimage.py /data/GenImage --generators stable_diffusion_v_1_4 \
        --out data/manifests
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def rows_for(root: Path, generator: str, split: str):
    for label_dir, label in (("nature", 0), ("ai", 1)):
        base = root / generator / split / label_dir
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if p.suffix.lower() in EXTS:
                yield [str(p), label, generator if label else "real"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--generators", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("data/manifests"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for split, name in (("train", "train"), ("val", "val")):
        out = args.out / f"genimage_{'_'.join(args.generators)}_{name}.csv"
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "label", "generator"])
            n = 0
            for gen in args.generators:
                for row in rows_for(args.root, gen, split):
                    w.writerow(row)
                    n += 1
        print(f"{out}: {n} rows")


if __name__ == "__main__":
    main()
