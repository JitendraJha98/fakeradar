"""Detection metrics (pure numpy, no sklearn dependency).

Conventions: label 1 = AI-generated ("positive"), scores are P(AI).
We report AUROC and AP for ranking quality, accuracy at 0.5, expected
calibration error (ECE), and TPR at a fixed low FPR — the metric that matters
operationally, because false-accusing real photos is the costly error.
"""

from __future__ import annotations

import numpy as np


def _validate(y_true, y_score) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=np.float64).ravel()
    s = np.asarray(y_score, dtype=np.float64).ravel()
    if y.shape != s.shape:
        raise ValueError("y_true and y_score must have the same shape")
    return y, s


def roc_auc(y_true, y_score) -> float:
    """AUROC via the Mann-Whitney U statistic (tie-aware)."""
    y, s = _validate(y_true, y_score)
    pos, neg = (y == 1).sum(), (y == 0).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < len(sorted_s):  # average ranks over ties
        j = i
        while j + 1 < len(sorted_s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def average_precision(y_true, y_score) -> float:
    y, s = _validate(y_true, y_score)
    if (y == 1).sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    precision = tp / np.arange(1, len(y_sorted) + 1)
    return float((precision * y_sorted).sum() / y_sorted.sum())


def accuracy_at(y_true, y_score, threshold: float = 0.5) -> float:
    y, s = _validate(y_true, y_score)
    return float(((s >= threshold).astype(np.float64) == y).mean())


def balanced_accuracy(y_true, y_score, threshold: float = 0.5) -> float:
    y, s = _validate(y_true, y_score)
    pred = s >= threshold
    tpr = pred[y == 1].mean() if (y == 1).any() else np.nan
    tnr = (~pred[y == 0]).mean() if (y == 0).any() else np.nan
    return float(np.nanmean([tpr, tnr]))


def tpr_at_fpr(y_true, y_score, max_fpr: float = 0.05) -> tuple[float, float]:
    """Return (TPR, threshold) at the largest threshold whose FPR <= max_fpr."""
    y, s = _validate(y_true, y_score)
    thresholds = np.unique(s)[::-1]
    best = (0.0, float(thresholds[0]) if len(thresholds) else 1.0)
    for t in thresholds:
        pred = s >= t
        neg = (y == 0).sum()
        fpr = pred[y == 0].sum() / neg if neg else 0.0
        if fpr <= max_fpr:
            tpr = pred[y == 1].mean() if (y == 1).any() else 0.0
            best = (float(tpr), float(t))
        else:
            break
    return best


def expected_calibration_error(y_true, y_score, n_bins: int = 10) -> float:
    y, s = _validate(y_true, y_score)
    bins = np.clip((s * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - s[mask].mean())
    return float(ece)


def summarize(y_true, y_score) -> dict[str, float]:
    tpr, thr = tpr_at_fpr(y_true, y_score, 0.05)
    return {
        "auroc": roc_auc(y_true, y_score),
        "ap": average_precision(y_true, y_score),
        "acc@0.5": accuracy_at(y_true, y_score),
        "balanced_acc": balanced_accuracy(y_true, y_score),
        "tpr@5%fpr": tpr,
        "thr@5%fpr": thr,
        "ece": expected_calibration_error(y_true, y_score),
    }
