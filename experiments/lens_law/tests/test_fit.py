import numpy as np
import pytest
from lenslaw.fit import fit_constant, fit_thin_lens, fit_valley, score_points

SCORE_KEYS = {
    "n",
    "blur_spread",
    "depth_spread",
    "resid_const",
    "resid_lens",
    "r2_lens",
    "resid_valley",
    "r2_valley",
    "c0",
    "k",
    "d_focus",
}


def lens_b(d, c0=0.8, k=6.0, d_focus=0.4):
    return 0.5 * np.log(c0**2 + k**2 * (d - d_focus) ** 2)


def lawful(n=400, noise=0.05, seed=0, lo=0.0, hi=1.0):
    rng = np.random.default_rng(seed)
    d = rng.uniform(lo, hi, n)
    return d, lens_b(d) + rng.normal(0, noise, n), np.ones(n)


def test_thin_lens_fit_recovers_the_focus_and_explains_lawful_data():
    d, b, w = lawful()
    r = fit_thin_lens(d, b, w)
    assert abs(r.params["d_focus"] - 0.4) < 0.03
    assert r.r2 >= 0.95


def test_lens_residual_is_unchanged_by_scale_shift_and_flip_of_depth():
    d, b, w = lawful()
    base = fit_thin_lens(d, b, w).resid
    for d2 in (3 * d + 2, 1 - d):
        assert abs(fit_thin_lens(d2, b, w).resid - base) / base < 0.05


def test_focus_is_reported_in_the_units_of_the_input_depth():
    d, b, w = lawful()
    assert fit_thin_lens(3 * d + 2, b, w).params["d_focus"] == pytest.approx(3.2, abs=0.1)


def test_step_data_fits_the_lens_law_much_worse_than_lawful_data():
    d, b_law, w = lawful()
    rng = np.random.default_rng(1)
    b_step = np.where(d >= 0.4, np.log(0.8), np.log(4.0)) + rng.normal(0, 0.05, d.size)
    lawful_resid = fit_thin_lens(d, b_law, w).resid
    step_resid = fit_thin_lens(d, b_step, w).resid
    assert step_resid >= 3 * lawful_resid
    assert fit_valley(d, b_step, w).resid < step_resid


def test_points_on_one_side_of_the_focus_give_a_finite_fit():
    d, b, w = lawful(lo=0.45, hi=1.0)
    r = fit_thin_lens(d, b, w)
    assert np.isfinite(r.resid) and np.isfinite(r.r2)
    assert all(np.isfinite(v) for v in r.params.values())


def test_all_points_at_one_depth_fall_back_to_a_constant():
    b = np.array([0.1, 0.2, 0.3, 0.4])
    r = fit_thin_lens(np.full(4, 0.5), b, np.ones(4))
    assert r.resid == pytest.approx(fit_constant(b, np.ones(4)).resid)
    assert r.params["k"] == 0.0


def test_valley_fit_is_exact_on_unimodal_data():
    d = np.linspace(0, 1, 50)
    assert fit_valley(d, np.abs(d - 0.3) ** 0.5, np.ones(50)).resid < 1e-9


def test_valley_fit_cannot_follow_a_hill():
    d = np.linspace(0, 1, 50)
    assert fit_valley(d, -np.abs(d - 0.5), np.ones(50)).resid > 0.05


def test_constant_fit_residual_is_the_weighted_standard_deviation():
    assert fit_constant(np.array([0.0, 2.0]), np.array([1.0, 1.0])).resid == pytest.approx(1.0)


def test_zero_weight_points_do_not_affect_the_fit():
    d, b, w = lawful()
    b_bad, w_bad = b.copy(), w.copy()
    b_bad[:50] += 5.0
    w_bad[:50] = 0.0
    kept = fit_thin_lens(d[50:], b[50:], w[50:]).resid
    assert fit_thin_lens(d, b_bad, w_bad).resid == pytest.approx(kept, rel=1e-6)


def test_score_points_reports_every_key_and_the_blur_spread():
    d, b, w = lawful()
    s = score_points(d, b, w)
    assert set(s) == SCORE_KEYS
    assert s["n"] == 400
    expected = float(np.exp(np.percentile(b, 90) - np.percentile(b, 10)))
    assert s["blur_spread"] == pytest.approx(expected, rel=1e-6)
    assert s["r2_lens"] >= 0.95


def test_score_points_on_empty_input_has_n_zero_and_nan_scores():
    s = score_points(np.array([]), np.array([]), np.array([]))
    assert set(s) == SCORE_KEYS
    assert s["n"] == 0
    assert np.isnan(s["resid_lens"])


def test_non_finite_points_are_ignored():
    d, b, w = lawful()
    b2 = b.copy()
    b2[0] = np.nan
    b2[1] = np.inf
    s = score_points(d, b2, w)
    assert s["n"] == 398
    assert np.isfinite(s["resid_lens"])
