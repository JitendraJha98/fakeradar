"""A bounded false-accusation rate from real photos only (split conformal).

`fakeradar calibrate-fpr REAL.csv --checkpoint best.pt --alpha 0.01`

A threshold is set so that a new real photo, exchangeable with the calibration
photos of its stratum, is flagged with probability at most `alpha`. Only real
photos enter, so a new generator cannot break the bound; a change in the real
photos (new cameras, a platform that re-encodes differently) can, which is why
thresholds are kept per stratum and should be refreshed with fresh real
photos. In a manifest the stratum is the `generator` column: for real photos
put the transport state there (for example `orig`, `whatsapp`).

Audited on 23,105 phone photos shared through five platforms; see
`experiments/fpr_guarantee/GATE_B_REPORT.md`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _finite(scores) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64).ravel()
    bad = int((~np.isfinite(scores)).sum())
    if bad:
        raise ValueError(f"{bad} scores are NaN or infinite")
    return scores


def conformal_threshold(scores, alpha: float) -> float:
    """Score above which a photo is flagged; infinite if there are too few photos."""
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
    """True where a score is strictly above its stratum's threshold (pooled if unknown)."""
    scores = _finite(scores)
    strata = np.asarray(strata).ravel()
    limits = np.array([cal.thresholds.get(str(s), cal.pooled) for s in strata.tolist()])
    return scores > limits


def save_calibration(cal: Calibration, path: str | Path, n: int) -> None:
    data = {
        "alpha": cal.alpha,
        "n": int(n),
        "pooled": cal.pooled,
        "thresholds": cal.thresholds,
        "fallback": sorted(cal.fallback),
    }
    Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_calibration(path: str | Path) -> Calibration:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Calibration(
        {str(k): float(v) for k, v in data["thresholds"].items()},
        float(data["pooled"]),
        float(data["alpha"]),
        set(data.get("fallback", [])),
    )


def calibrate_real_manifest(
    detector, manifest: str | Path, alpha: float, equalize_jpeg_quality: int | None = None
) -> tuple[Calibration, int]:
    """Score a manifest of real photos with `detector` and calibrate on P(AI).

    Every row must be labelled real (0); its `generator` column is the stratum.
    """
    from ..preprocessing import load_image_equalized
    from ..training.datasets import read_manifest

    rows = read_manifest(manifest)
    generated = sum(r["label"] != 0 for r in rows)
    if generated:
        raise ValueError(
            f"{manifest} has {generated} rows not labelled real (0); "
            "the guarantee is calibrated on real photos only."
        )
    scores = [
        detector.predict(load_image_equalized(r["path"], equalize_jpeg_quality)).prob_ai
        for r in rows
    ]
    return calibrate(scores, [r["generator"] for r in rows], alpha), len(rows)
