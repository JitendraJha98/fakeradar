"""Gate B report: the fixed audit plus plain comparisons.

    python -m fprcal.report ../../runs/fpr_guarantee/fodb_records.jsonl \
        --out GATE_B_REPORT.md --detector "Community Forensics (commfor-model-384)"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .conformal import calibrate, flag
from .gate_b import GATE_B, gate_b


def load_records(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def audit_rates(
    records: list[dict], mode: str, alpha: float | None = None, threshold: float | None = None
) -> dict[str, float]:
    """Share of audit-phone photos flagged, per platform.

    mode "fixed": score above `threshold` (0 is the detector's own 0.5 cut).
    mode "pooled": one threshold from all calibration photos, ignoring platform.
    mode "stratified": one threshold per platform, as Gate B uses.
    The held-out platform never sets a threshold in the calibrated modes.
    """
    cal = [r for r in records if r["role"] == "cal" and r["platform"] != GATE_B["held_out"]]
    audit = [r for r in records if r["role"] == "audit"]
    scores = np.array([r["score"] for r in audit])
    platforms = np.array([r["platform"] for r in audit])
    if mode == "fixed":
        flagged = scores > threshold
    else:
        cal_scores = np.array([r["score"] for r in cal])
        cal_strata = np.array([r["platform"] if mode == "stratified" else "all" for r in cal])
        fitted = calibrate(cal_scores, cal_strata, alpha)
        flagged = flag(
            fitted, scores, platforms if mode == "stratified" else np.full(len(audit), "all")
        )
    return {str(p): float(flagged[platforms == p].mean()) for p in sorted(set(platforms.tolist()))}


def _pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def render_report(records: list[dict], result: dict, detector: str) -> str:
    """Markdown report for an already-computed Gate B result."""
    phones = {
        r["role"]: sorted({x["phone"] for x in records if x["role"] == r["role"]}) for r in records
    }
    platforms = sorted({r["platform"] for r in records})
    lines = [
        "# Gate B report: does the calibrated false-accusation rate hold on unseen phones?",
        "",
        f"**Verdict: Gate B {'PASSED' if result['passed'] else 'FAILED'}** "
        f"({result['n_failed']} of {len(result['checks'])} checks missed; the rule allows "
        f"{GATE_B['max_failed']} and no rate above {GATE_B['hard_ratio']:g}x its level).",
        "",
        f"- Detector: {detector}. Higher score means more likely generated.",
        f"- Photos: {len(records):,} real photos, {len(phones.get('cal', []))} calibration phones, "
        f"{len(phones.get('audit', []))} audit phones, versions: {', '.join(platforms)}.",
        f"- Held out of calibration: {GATE_B['held_out']}.",
        "- Thresholds come from calibration phones only; rates below are on audit phones only.",
        "",
        "## The ten checks",
        "",
        "| Version | Level | Measured rate | 95% interval | Check |",
        "|---|---|---|---|---|",
    ]
    for key, c in result["checks"].items():
        platform, level = key.split("@")
        verdict = "pass" if c["passed"] else "MISS"
        if c["over_hard_limit"]:
            verdict += " (above hard limit)"
        lines.append(
            f"| {platform} | {_pct(float(level))} | {_pct(c['rate'])} | "
            f"{_pct(c['low'])} to {_pct(c['high'])} | {verdict} |"
        )
    held = result["held_out"]
    lines += [
        "",
        f"## Held-out version ({held['platform']}), thresholded with the pooled calibration",
        "",
        "| Level | Measured rate |",
        "|---|---|",
        *[f"| {_pct(float(level))} | {_pct(rate)} |" for level, rate in held["rates"].items()],
        "",
        "## Comparisons (audit phones)",
        "",
        "Share of real photos flagged per version. The detector's own cut is a score of 0",
        "(probability 0.5). Temperature scaling is not listed because it rescales scores",
        "without moving that cut, so its decisions are identical to the fixed cut.",
        "",
        "| Version | Fixed cut (0.5) | Pooled 5% | Per-version 5% | Pooled 1% | Per-version 1% |",
        "|---|---|---|---|---|---|",
    ]
    fixed = audit_rates(records, "fixed", threshold=0.0)
    table = {
        (mode, alpha): audit_rates(records, mode, alpha=alpha)
        for mode in ("pooled", "stratified")
        for alpha in GATE_B["levels"]
    }
    for p in platforms:
        lines.append(
            f"| {p} | {_pct(fixed[p])} | {_pct(table['pooled', 0.05][p])} | "
            f"{_pct(table['stratified', 0.05][p])} | {_pct(table['pooled', 0.01][p])} | "
            f"{_pct(table['stratified', 0.01][p])} |"
        )
    lines += [
        "",
        "## Pass rule (fixed before any score existed)",
        "",
        f"A check passes if the measured rate is at most {GATE_B['ratio']:g}x the level, or if the "
        "low end of its 95% interval is at or below the level. Intervals resample calibration and "
        f"audit phones together ({GATE_B['bootstrap']:,} draws). The gate passes if at most "
        f"{GATE_B['max_failed']} check misses and no rate exceeds {GATE_B['hard_ratio']:g}x its level.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Gate B once and write the report.")
    parser.add_argument("records", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--detector", default="reference detector")
    args = parser.parse_args(argv)
    records = load_records(args.records)
    result = gate_b(records)
    args.out.write_bytes(render_report(records, result, args.detector).encode("utf-8"))
    args.out.with_suffix(".json").write_bytes(json.dumps(result, indent=1).encode("utf-8"))
    print(f"Gate B {'PASSED' if result['passed'] else 'FAILED'}; report in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
