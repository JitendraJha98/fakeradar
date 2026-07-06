import numpy as np

from fakeradar.evaluation.metrics import (
    average_precision,
    expected_calibration_error,
    roc_auc,
    summarize,
    tpr_at_fpr,
)


def test_auroc_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 5000)
    s = rng.random(5000)
    assert abs(roc_auc(y, s) - 0.5) < 0.03


def test_auroc_handles_ties():
    y = np.array([0, 1, 0, 1])
    s = np.array([0.5, 0.5, 0.5, 0.5])
    assert abs(roc_auc(y, s) - 0.5) < 1e-9


def test_ap_bounds():
    y = np.array([0, 1, 0, 1, 1])
    s = np.array([0.1, 0.9, 0.2, 0.8, 0.7])
    assert 0.99 <= average_precision(y, s) <= 1.0


def test_tpr_at_fpr_and_ece_and_summary():
    y = np.array([0] * 50 + [1] * 50)
    s = np.concatenate([np.linspace(0, 0.4, 50), np.linspace(0.6, 1.0, 50)])
    tpr, thr = tpr_at_fpr(y, s, 0.05)
    assert tpr == 1.0 and 0.35 < thr <= 0.6
    assert 0.0 <= expected_calibration_error(y, s) <= 1.0
    m = summarize(y, s)
    assert m["auroc"] == 1.0 and 0 <= m["ece"] <= 1
