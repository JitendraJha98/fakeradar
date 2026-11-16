"""Gate M: can the lens law be measured at all? Records in, M1..M5 verdicts out.

Thresholds are copied from the design spec (section 5, Phase 1) and must not
be edited after any score has been seen.

Record keys used: source ("pairs", "render", "real_wide"), variant
("wide_depth_narrow", "wide_depth_self", "narrow_depth_self", a render mode,
or "photo"), condition, split, scene, applicable, blur_spread, resid_lens,
r2_lens.
"""

from __future__ import annotations

import math

import numpy as np

THRESHOLDS = {
    "M1_r2": 0.5,
    "M1_share_depth_narrow": 0.80,
    "M1_share_depth_self": 0.70,
    "M2_share": 0.90,
    "M3_auroc": 0.85,
    "M4_fpr": 0.05,
    "M4_unlawful_flagged": 0.50,
    "M4_lawful_flagged": 0.10,
    "M5_condition": "s050_q70",
    "M5_auroc_drop": 0.10,
    "M5_unlawful_flagged": 0.35,
}
UNLAWFUL = ("step", "position", "missing_foreground")


def auroc(pos, neg) -> float:
    """Chance that a random `pos` value exceeds a random `neg` value; ties count half."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.sort(np.asarray(neg, dtype=np.float64))
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    below = np.searchsorted(neg, pos, side="left")
    tied = np.searchsorted(neg, pos, side="right") - below
    return float(np.mean((below + 0.5 * tied) / neg.size))


def _finite(x) -> bool:
    return isinstance(x, int | float) and math.isfinite(x)


def _pick(records, source, variant, condition, usable_only=True):
    return [
        r
        for r in records
        if r["source"] == source
        and r["variant"] == variant
        and r["condition"] == condition
        and (r.get("applicable") or not usable_only)
    ]


def _resid(records, source, variant, condition) -> list[float]:
    rows = _pick(records, source, variant, condition)
    return [r["resid_lens"] for r in rows if _finite(r["resid_lens"])]


def _share(flags) -> float:
    flags = list(flags)
    return float(np.mean(flags)) if flags else float("nan")


def _separation(records, dev, condition) -> dict:
    """AUROC per violation type and flag rates at the dev real-photo threshold."""
    lawful = _resid(records, "render", "lawful", condition)
    aurocs = {m: auroc(_resid(records, "render", m, condition), lawful) for m in UNLAWFUL}
    real = _resid(dev, "real_wide", "photo", condition)
    q = 100 * (1 - THRESHOLDS["M4_fpr"])
    threshold = float(np.percentile(real, q)) if real else float("nan")
    unlawful = [v for m in UNLAWFUL for v in _resid(records, "render", m, condition)]
    return {
        "auroc": aurocs,
        "threshold": threshold,
        "n_real": len(real),
        "unlawful_flagged": _share(v > threshold for v in unlawful),
        "lawful_flagged": _share(v > threshold for v in lawful),
    }


def gate_m(records: list[dict], split: str) -> dict:
    """Evaluate M1..M5 on `split`; the flagging threshold always comes from dev."""
    t = THRESHOLDS
    here = [r for r in records if r["split"] == split]
    dev = [r for r in records if r["split"] == "dev"]

    def fit_share(variant):
        rows = _pick(here, "pairs", variant, "clean")
        good = [_finite(r["r2_lens"]) and r["r2_lens"] >= t["M1_r2"] for r in rows]
        return _share(good), len(rows)

    narrow_share, n_narrow = fit_share("wide_depth_narrow")
    self_share, n_self = fit_share("wide_depth_self")
    m1 = {
        "value": {
            "depth_narrow": narrow_share,
            "n_depth_narrow": n_narrow,
            "depth_self": self_share,
            "n_depth_self": n_self,
        },
        "passed": bool(
            narrow_share >= t["M1_share_depth_narrow"] and self_share >= t["M1_share_depth_self"]
        ),
    }

    def spreads(variant):
        rows = _pick(here, "pairs", variant, "clean", usable_only=False)
        return {r["scene"]: r["blur_spread"] for r in rows if _finite(r["blur_spread"])}

    wide, narrow = spreads("wide_depth_self"), spreads("narrow_depth_self")
    both = sorted(set(wide) & set(narrow), key=str)
    m2_share = _share(wide[s] > narrow[s] for s in both)
    m2 = {"value": m2_share, "n_pairs": len(both), "passed": bool(m2_share >= t["M2_share"])}

    clean = _separation(here, dev, "clean")
    m3 = {
        "value": clean["auroc"],
        "passed": bool(all(a >= t["M3_auroc"] for a in clean["auroc"].values())),
    }
    m4 = {
        "value": {
            k: clean[k] for k in ("threshold", "n_real", "unlawful_flagged", "lawful_flagged")
        },
        "passed": bool(
            clean["unlawful_flagged"] >= t["M4_unlawful_flagged"]
            and clean["lawful_flagged"] <= t["M4_lawful_flagged"]
        ),
    }

    moved = _separation(here, dev, t["M5_condition"])
    drops = {m: clean["auroc"][m] - moved["auroc"][m] for m in UNLAWFUL}
    m5 = {
        "value": {
            "auroc": moved["auroc"],
            "auroc_drop": drops,
            "unlawful_flagged": moved["unlawful_flagged"],
        },
        "passed": bool(
            all(d <= t["M5_auroc_drop"] for d in drops.values())
            and moved["unlawful_flagged"] >= t["M5_unlawful_flagged"]
        ),
    }

    checks = {"M1": m1, "M2": m2, "M3": m3, "M4": m4, "M5": m5}
    return {"split": split, **checks, "passed": all(c["passed"] for c in checks.values())}
