"""Gate B: does the calibrated false-accusation rate hold on unseen phones?

The pass rule is fixed in the Phase 3 plan and must not be edited after any
score has been seen. A record has: phone, role ("cal" or "audit"), platform,
score (higher means more likely generated).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .conformal import calibrate, flag

GATE_B = {
    "levels": (0.05, 0.01),
    "ratio": 1.5,
    "max_failed": 1,
    "hard_ratio": 2.0,
    "bootstrap": 2000,
    "held_out": "telegram",
}


def _by_phone(records: list[dict], role: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    grouped: dict[str, list] = defaultdict(list)
    for r in records:
        if r["role"] == role:
            grouped[r["phone"]].append((float(r["score"]), str(r["platform"])))
    return {
        phone: (np.array([s for s, _ in rows]), np.array([p for _, p in rows]))
        for phone, rows in sorted(grouped.items())
    }


def _stack(phones: dict, names) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate([phones[n][0] for n in names]),
        np.concatenate([phones[n][1] for n in names]),
    )


def _rates(cal_scores, cal_strata, audit_scores, audit_strata, alpha, held_out) -> dict[str, float]:
    seen = cal_strata != held_out
    cal = calibrate(cal_scores[seen], cal_strata[seen], alpha)
    flagged = flag(cal, audit_scores, audit_strata)
    return {str(p): float(flagged[audit_strata == p].mean()) for p in np.unique(audit_strata)}


def gate_b(records: list[dict], seed: int = 0, n_boot: int | None = None) -> dict:
    """Evaluate the fixed pass rule. Intervals bootstrap calibration and audit phones."""
    both = {r["phone"] for r in records if r["role"] == "cal"} & {
        r["phone"] for r in records if r["role"] == "audit"
    }
    if both:
        raise ValueError(f"phones in both calibration and audit: {sorted(both)}")
    cal, audit = _by_phone(records, "cal"), _by_phone(records, "audit")
    held_out = GATE_B["held_out"]
    n_boot = GATE_B["bootstrap"] if n_boot is None else n_boot
    rng = np.random.default_rng(seed)
    cal_names, audit_names = list(cal), list(audit)
    draws = [
        (rng.choice(cal_names, len(cal_names)), rng.choice(audit_names, len(audit_names)))
        for _ in range(n_boot)
    ]

    checks, held = {}, {}
    for alpha in GATE_B["levels"]:
        point = _rates(*_stack(cal, cal_names), *_stack(audit, audit_names), alpha, held_out)
        boot: dict[str, list[float]] = defaultdict(list)
        for c_draw, a_draw in draws:
            for p, v in _rates(
                *_stack(cal, c_draw), *_stack(audit, a_draw), alpha, held_out
            ).items():
                boot[p].append(v)
        for platform, rate in point.items():
            lo, hi = (float(x) for x in np.percentile(boot[platform], [2.5, 97.5]))
            if platform == held_out:
                held[str(alpha)] = rate
                continue
            checks[f"{platform}@{alpha}"] = {
                "rate": rate,
                "low": lo,
                "high": hi,
                "passed": bool(rate <= GATE_B["ratio"] * alpha or lo <= alpha),
                "over_hard_limit": bool(rate > GATE_B["hard_ratio"] * alpha),
            }

    n_failed = sum(not c["passed"] for c in checks.values())
    hard = any(c["over_hard_limit"] for c in checks.values())
    return {
        "checks": checks,
        "held_out": {"platform": held_out, "rates": held},
        "n_failed": n_failed,
        "passed": bool(n_failed <= GATE_B["max_failed"] and not hard),
    }
