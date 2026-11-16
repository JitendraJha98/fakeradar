"""Split-conformal thresholds from real photos only.

If a new real photo is exchangeable with the calibration photos of its
stratum, the chance that its score exceeds the threshold is at most `alpha`.
Nothing about generated images enters, so the bound cannot be broken by a
new generator; only a change in the real photos can break it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def _finite(scores) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64).ravel()
    bad = int((~np.isfinite(scores)).sum())
    if bad:
        raise ValueError(f"{bad} scores are NaN or infinite")
    return scores


def conformal_threshold(scores, alpha: float) -> float:
    """Score above which a photo is flagged. Infinite if there are too few photos."""
    scores = np.sort(_finite(scores))
    rank = math.ceil((scores.size + 1) * (1 - alpha))
    return float("inf") if rank > scores.size else float(scores[rank - 1])


@dataclass
class Calibration:
    thresholds: dict[str, float]
    pooled: float
    alpha: float
    fallback: set[str] = field(default_factory=set)  # strata too small for their own threshold


def calibrate(scores, strata, alpha: float, min_count: int | None = None) -> Calibration:
    """One threshold per stratum with enough photos; the pooled one otherwise."""
    scores = _finite(scores)
    strata = np.asarray(strata).ravel()
    if strata.size != scores.size:
        raise ValueError("scores and strata differ in length")
    if min_count is None:
        min_count = math.ceil(1 / alpha) - 1  # fewer photos cannot give this level at all
    cal = Calibration({}, conformal_threshold(scores, alpha), alpha)
    for name in sorted(set(strata.tolist())):
        own = scores[strata == name]
        if own.size >= min_count:
            cal.thresholds[str(name)] = conformal_threshold(own, alpha)
        else:
            cal.fallback.add(str(name))
    return cal


def flag(cal: Calibration, scores, strata) -> np.ndarray:
    """True where a photo's score is strictly above its stratum's threshold."""
    scores = _finite(scores)
    strata = np.asarray(strata).ravel()
    limits = np.array([cal.thresholds.get(str(s), cal.pooled) for s in strata.tolist()])
    return scores > limits
