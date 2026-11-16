import math

import numpy as np
import pytest
from lenslaw.blur_classic import BlurMap, estimate_blur


def step_edge(sigma: float, size: int = 256, lo: float = 0.2, hi: float = 0.8) -> np.ndarray:
    """A vertical edge at the centre column, blurred by a Gaussian of `sigma` pixels."""
    x = np.arange(size) - size / 2 + 0.5
    profile = np.array([0.5 * (1 + math.erf(v / (sigma * math.sqrt(2)))) for v in x])
    return np.tile(lo + (hi - lo) * profile, (size, 1)).astype(np.float32)


@pytest.mark.parametrize("sigma", [1.0, 2.0, 4.0, 8.0])
def test_blur_of_a_step_edge_is_recovered_within_thirty_percent(sigma):
    est = estimate_blur(step_edge(sigma))
    values = est.sigma[np.isfinite(est.sigma)]
    assert values.size > 0
    assert abs(np.median(values) - sigma) / sigma < 0.30


def test_sharper_edges_get_smaller_estimates_than_blurrier_ones():
    def med(sigma):
        s = estimate_blur(step_edge(sigma)).sigma
        return np.median(s[np.isfinite(s)])

    assert med(1.0) < med(2.0) < med(4.0) < med(8.0)


def test_flat_image_has_no_estimates():
    est = estimate_blur(np.full((128, 128), 0.5, dtype=np.float32))
    assert isinstance(est, BlurMap)
    assert est.sigma.shape == (128, 128) and est.conf.shape == (128, 128)
    assert not np.isfinite(est.sigma).any()
    assert (est.conf == 0).all()


def test_noise_never_produces_infinite_or_out_of_range_blur():
    rng = np.random.default_rng(0)
    est = estimate_blur(rng.uniform(0, 1, (128, 128)).astype(np.float32))
    assert not np.isinf(est.sigma).any()
    finite = est.sigma[np.isfinite(est.sigma)]
    assert ((finite > 0.3) & (finite < 60)).all()


def test_estimates_sit_on_the_edge_not_in_the_flat_regions():
    est = estimate_blur(step_edge(2.0))
    cols = np.where(np.isfinite(est.sigma).any(axis=0))[0]
    assert cols.min() >= 128 - 12 and cols.max() <= 128 + 12


def test_confidence_is_between_zero_and_one_and_positive_where_estimated():
    est = estimate_blur(step_edge(2.0))
    assert est.conf.min() >= 0 and est.conf.max() <= 1
    assert (est.conf[np.isfinite(est.sigma)] > 0).all()


def test_image_smaller_than_the_coarsest_scale_does_not_raise():
    est = estimate_blur(step_edge(1.0, size=24))
    assert est.sigma.shape == (24, 24)


def test_blur_estimate_is_the_same_when_the_device_is_named():
    img = step_edge(2.0)
    a, b = estimate_blur(img), estimate_blur(img, device="cpu")
    assert np.array_equal(a.sigma, b.sigma, equal_nan=True)
