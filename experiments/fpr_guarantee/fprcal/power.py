"""Detection rate of generated images at the calibrated real-photo thresholds."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .conformal import calibrate
from .gate_b import GATE_B


def detection_rates(
    real_records: list[dict], generated: list[dict], alpha: float
) -> dict[str, dict]:
    """Share of generated images flagged, per generator/condition and per threshold.

    Thresholds come from calibration phones only (never audit phones, never the
    held-out platform): one per platform version plus the pooled one. Each
    column answers "if this image were shared as that version, would it be
    flagged at this level?".
    """
    cal = [r for r in real_records if r["role"] == "cal" and r["platform"] != GATE_B["held_out"]]
    fitted = calibrate(
        np.array([r["score"] for r in cal]), np.array([r["platform"] for r in cal]), alpha
    )
    limits = dict(fitted.thresholds, pooled=fitted.pooled)
    groups: dict[str, list[float]] = defaultdict(list)
    for r in generated:
        groups[f"{r['generator']}/{r['condition']}"].append(float(r["score"]))
    out = {}
    for key in sorted(groups):
        scores = np.array(groups[key])
        out[key] = {name: float((scores > limit).mean()) for name, limit in limits.items()}
        out[key]["n"] = int(scores.size)
    return out
