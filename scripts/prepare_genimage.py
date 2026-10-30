#!/usr/bin/env python3
"""Build fakeradar CSV manifests from a GenImage download.

Thin wrapper around `fakeradar prepare-genimage` (logic lives in
fakeradar.training.genimage so pip installs get it too). Usage:

    python scripts/prepare_genimage.py /data/GenImage \
        --generators stable_diffusion_v_1_4 --test-generators all --out data/manifests
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fakeradar.training.genimage import build_manifests


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--generators", nargs="+", required=True, help="training generator(s)")
    ap.add_argument("--test-generators", nargs="+", default=None, help="or 'all'")
    ap.add_argument("--out", type=Path, default=Path("data/manifests"))
    ap.add_argument("--name", default=None)
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    result = build_manifests(
        args.root,
        args.generators,
        args.test_generators,
        args.out,
        args.name,
        args.val_fraction,
        args.seed,
    )
    for path, n in result.values():
        print(f"{path}: {n} rows")


if __name__ == "__main__":
    main()
