import numpy as np
import pytest
from lenslaw.synth import (
    BASE_SIGMA,
    KINDS,
    blur_linear,
    defocus_kernel,
    degrade,
    label_for,
    make_training_patch,
)


def second_moment(kernel):
    r = kernel.shape[0] // 2
    x = np.arange(-r, r + 1)
    return float((kernel.sum(axis=0) * x * x).sum())


def noise(seed=0, size=200):
    return np.random.default_rng(seed).uniform(0, 1, (size, size, 3)).astype(np.float32)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("sigma", [1.0, 2.0, 6.0])
def test_kernel_sums_to_one_and_has_the_requested_spread(kind, sigma):
    k = defocus_kernel(sigma, kind)
    assert k.shape[0] == k.shape[1] and k.shape[0] % 2 == 1
    assert k.sum() == pytest.approx(1.0, abs=1e-5)
    assert second_moment(k) == pytest.approx(sigma**2, rel=0.12)


def test_unknown_kernel_kind_is_rejected():
    with pytest.raises(ValueError):
        defocus_kernel(2.0, "hexagon")


def test_blur_keeps_light_energy_and_removes_detail():
    img = noise()
    out = blur_linear(img, defocus_kernel(3.0, "disc"))
    assert out.shape == img.shape
    assert out.std() < 0.4 * img.std()
    from lenslaw.render import srgb_to_linear

    assert srgb_to_linear(out).mean() == pytest.approx(srgb_to_linear(img).mean(), rel=0.01)


def test_degrade_without_any_degradation_returns_the_centre_crop():
    img = noise()
    out = degrade(
        img, sigma=0.0, kind="gaussian", scale=1.0, jpeg_quality=None, noise_std=0.0, out=64
    )
    assert out.shape == (64, 64, 3) and out.dtype == np.float32
    assert np.abs(out - img[68:132, 68:132]).max() < 1 / 255 + 1e-6


def test_degrade_output_size_is_fixed_whatever_the_scale():
    img = noise()
    for scale in (0.5, 0.75, 1.0):
        out = degrade(img, 2.0, "gaussian", scale, 80, 0.01, out=64, rng=np.random.default_rng(0))
        assert out.shape == (64, 64, 3)
        assert 0.0 <= out.min() and out.max() <= 1.0


def test_label_combines_the_base_sharpness_with_the_scaled_blur():
    assert label_for(0.0, 1.0) == pytest.approx(np.log(BASE_SIGMA))
    assert label_for(4.0, 0.5) == pytest.approx(0.5 * np.log(BASE_SIGMA**2 + 2.0**2))
    assert label_for(4.0, 1.0) > label_for(4.0, 0.5) > label_for(1.0, 0.5)


def test_training_patch_is_reproducible_and_labelled_within_range():
    img = noise(size=260)
    a_patch, a_label = make_training_patch(img, np.random.default_rng(5), out=96)
    b_patch, b_label = make_training_patch(img, np.random.default_rng(5), out=96)
    assert np.array_equal(a_patch, b_patch) and a_label == b_label
    assert a_patch.shape == (96, 96, 3) and a_patch.dtype == np.float32
    labels = [make_training_patch(img, np.random.default_rng(s), out=96)[1] for s in range(40)]
    assert min(labels) >= np.log(BASE_SIGMA) - 1e-6 and max(labels) <= np.log(13.0)
    assert np.std(labels) > 0.3


def test_training_patch_needs_a_source_crop_large_enough_for_the_blur():
    with pytest.raises(ValueError):
        make_training_patch(noise(size=100), np.random.default_rng(0), out=96)
