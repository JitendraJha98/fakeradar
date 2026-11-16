"""Load scoring records and summarise them for the Gate M memo.

python -m lenslaw.report runs/lens_law/phase1-score/records.jsonl          # dev only
python -m lenslaw.report runs/lens_law/phase1-score/records.jsonl --test   # once, at the end
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .gate_m import gate_m

PAIR_DIAGNOSTICS = (
    "align_corr",
    "depth_corr_narrow_gt",
    "depth_corr_wide_gt",
    "depth_corr_wide_narrow",
    "gt_invalid",
)


def load_records(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _median(values) -> float:
    values = [v for v in values if isinstance(v, int | float) and math.isfinite(v)]
    return float(np.median(values)) if values else float("nan")


def diagnostics(records: list[dict], split: str) -> dict[str, dict]:
    """Per source/variant/condition: count, share applicable, median fit quality."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r["split"] == split:
            groups[f"{r['source']}/{r['variant']}/{r['condition']}"].append(r)
    out = {}
    for key in sorted(groups):
        rows = groups[key]
        usable = [r for r in rows if r.get("applicable")]
        out[key] = {
            "n": len(rows),
            "applicable_share": len(usable) / len(rows),
            "median_r2_lens": _median(r["r2_lens"] for r in usable),
            "median_resid_lens": _median(r["resid_lens"] for r in usable),
            "median_blur_spread": _median(r["blur_spread"] for r in rows),
            "median_cells": _median(r["n"] for r in rows),
        }
    return out


def pair_diagnostics(records: list[dict], split: str) -> dict[str, float]:
    rows = [r for r in records if r["split"] == split and r.get("variant") == "wide_depth_narrow"]
    return {k: _median(r.get(k) for r in rows) for k in PAIR_DIAGNOSTICS}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gate M report from scoring records.")
    parser.add_argument("records", type=Path)
    parser.add_argument("--test", action="store_true", help="also evaluate the test split")
    parser.add_argument("--out", type=Path, default=None, help="write the report as JSON")
    args = parser.parse_args(argv)

    records = load_records(args.records)
    report = {"n_records": len(records)}
    for split in ("dev", "test") if args.test else ("dev",):
        report[split] = {
            "gate": gate_m(records, split),
            "diagnostics": diagnostics(records, split),
            "pairs": pair_diagnostics(records, split),
        }
    text = json.dumps(report, indent=1)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(text.encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
