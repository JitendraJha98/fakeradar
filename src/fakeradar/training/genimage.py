"""Build fakeradar CSV manifests from a GenImage download.

GenImage ships each generator as ``ROOT/<generator>/[<subset>/]{train,val}/{ai,nature}/``
(the released archives add a ``<subset>`` level such as ``imagenet_ai_0419_sdv4``;
both layouts are accepted).

Protocol (docs/benchmarks.md): train on one generator, test on all. GenImage's
``val`` split is the *test* set, so model selection + temperature calibration
use a stratified hold-out carved from the training generator's ``train`` split
instead — otherwise the reported test numbers would be tuned on the test set.

Outputs ``<name>_train.csv``, ``<name>_val.csv`` (hold-out) and
``<name>_test.csv`` (GenImage ``val`` of every test generator), with absolute
paths so the manifests work from any working directory.
"""

from __future__ import annotations

import csv
import random
from collections.abc import Iterable
from pathlib import Path

from ..constants import SUPPORTED_EXTENSIONS

LABEL_DIRS = (("nature", 0), ("ai", 1))


def find_split_dir(root: Path, generator: str, split: str) -> Path | None:
    """`ROOT/<gen>/<split>` or `ROOT/<gen>/<subset>/<split>` (first match)."""
    gen_dir = root / generator
    if (gen_dir / split).is_dir():
        return gen_dir / split
    if gen_dir.is_dir():
        for sub in sorted(p for p in gen_dir.iterdir() if p.is_dir()):
            if (sub / split).is_dir():
                return sub / split
    return None


def discover_generators(root: Path, split: str = "val") -> list[str]:
    """Every generator directory under `root` that has `split`."""
    return sorted(
        p.name for p in root.iterdir() if p.is_dir() and find_split_dir(root, p.name, split)
    )


def rows_for(root: Path, generator: str, split: str) -> list[list]:
    split_dir = find_split_dir(root, generator, split)
    if split_dir is None:
        raise FileNotFoundError(
            f"No {split!r} split for generator {generator!r} under {root} "
            f"(expected {root / generator}/[<subset>/]{split}/{{ai,nature}})."
        )
    rows: list[list] = []
    for label_dir, label in LABEL_DIRS:
        base = split_dir / label_dir
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if p.suffix.lower() in SUPPORTED_EXTENSIONS:
                rows.append([str(p.resolve()), label, generator if label else "real"])
    if {r[1] for r in rows} != {0, 1}:
        raise ValueError(f"{split_dir} must contain both nature/ and ai/ images.")
    return rows


def stratified_holdout(
    rows: list[list], fraction: float, seed: int
) -> tuple[list[list], list[list]]:
    """Split rows into (keep, holdout) with `fraction` of each label held out."""
    rng = random.Random(seed)
    keep, hold = [], []
    for label in (0, 1):
        group = [r for r in rows if r[1] == label]
        rng.shuffle(group)
        n = max(1, round(len(group) * fraction))
        hold += group[:n]
        keep += group[n:]
    return keep, hold


def _write(path: Path, rows: Iterable[list]) -> int:
    n = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "generator"])
        for row in rows:
            w.writerow(row)
            n += 1
    return n


def build_manifests(
    root: str | Path,
    train_generators: list[str],
    test_generators: list[str] | None = None,
    out_dir: str | Path = "data/manifests",
    name: str | None = None,
    val_fraction: float = 0.05,
    seed: int = 0,
) -> dict[str, tuple[Path, int]]:
    """Write train/val/test manifests; returns {split: (path, n_rows)}.

    `test_generators=None` tests on the training generators; `["all"]` tests
    on every generator found under `root`.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    root, out_dir = Path(root), Path(out_dir)
    if test_generators is None:
        test_generators = list(train_generators)
    elif test_generators == ["all"]:
        test_generators = discover_generators(root, "val")
    name = name or "genimage_" + "_".join(train_generators)

    train_rows = [r for g in train_generators for r in rows_for(root, g, "train")]
    train_rows, val_rows = stratified_holdout(train_rows, val_fraction, seed)
    test_rows = [r for g in test_generators for r in rows_for(root, g, "val")]

    out_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for split, rows in (("train", train_rows), ("val", val_rows), ("test", test_rows)):
        path = out_dir / f"{name}_{split}.csv"
        result[split] = (path, _write(path, rows))
    return result
