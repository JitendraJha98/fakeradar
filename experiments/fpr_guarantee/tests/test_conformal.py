import numpy as np
import pytest
from fprcal.conformal import calibrate, conformal_threshold, flag


def test_threshold_is_the_right_order_statistic():
    assert conformal_threshold(np.arange(1, 100), 0.05) == 95
    assert conformal_threshold(np.arange(1, 100), 0.01) == 99


def test_threshold_is_infinite_when_there_are_too_few_photos_for_the_level():
    assert conformal_threshold(np.arange(10), 0.01) == float("inf")


@pytest.mark.parametrize("alpha", [0.05, 0.01])
def test_flagged_share_of_exchangeable_real_photos_matches_the_level(alpha):
    rng = np.random.default_rng(0)
    scores = rng.normal(size=20000)
    strata = np.array(["a"] * 20000)
    cal = calibrate(scores[:10000], strata[:10000], alpha)
    assert abs(flag(cal, scores[10000:], strata[10000:]).mean() - alpha) < 0.004


def test_small_stratum_falls_back_to_the_pooled_threshold():
    rng = np.random.default_rng(1)
    scores = np.concatenate([rng.normal(size=5000), rng.normal(5, 1, size=50)])
    strata = np.array(["big"] * 5000 + ["small"] * 50)
    cal = calibrate(scores, strata, 0.01)
    assert cal.fallback == {"small"}
    assert "small" not in cal.thresholds and np.isfinite(cal.pooled)
    assert flag(
        cal, np.array([cal.pooled + 1, cal.pooled - 1]), np.array(["small", "small"])
    ).tolist() == [True, False]


def test_unseen_stratum_uses_the_pooled_threshold():
    rng = np.random.default_rng(2)
    cal = calibrate(rng.normal(size=2000), np.array(["a"] * 2000), 0.05)
    assert flag(cal, np.array([cal.pooled + 1.0]), np.array(["never-seen"])).tolist() == [True]


def test_identical_scores_flag_nothing():
    cal = calibrate(np.ones(500), np.array(["a"] * 500), 0.05)
    assert not flag(cal, np.ones(100), np.array(["a"] * 100)).any()


def test_strata_are_calibrated_separately():
    rng = np.random.default_rng(3)
    scores = np.concatenate([rng.normal(0, 1, 4000), rng.normal(3, 1, 4000)])
    strata = np.array(["low"] * 4000 + ["high"] * 4000)
    cal = calibrate(scores, strata, 0.05)
    assert cal.thresholds["high"] > cal.thresholds["low"] + 2


def test_non_finite_scores_are_rejected_with_their_count():
    scores = np.array([0.1, np.nan, np.inf, 0.3])
    with pytest.raises(ValueError, match="2"):
        calibrate(scores, np.array(["a"] * 4), 0.05)
