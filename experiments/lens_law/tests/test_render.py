import numpy as np
import pytest
from lenslaw.render import MODES, blur_field, linear_to_srgb, render_defocus, srgb_to_linear


def ramp(h=64, w=96):
    return np.tile(np.linspace(0, 1, w, dtype=np.float32), (h, 1))


def test_the_four_modes_are_named():
    assert MODES == ("lawful", "step", "position", "missing_foreground")


def test_lawful_field_is_zero_at_focus_and_linear_in_distance_from_it():
    d = ramp()
    f = blur_field(d, "lawful", d_focus=0.4, sigma_max=4.0)
    row, depth = f[0], d[0]
    assert row[np.argmin(np.abs(depth - 0.4))] < 0.1
    far = depth < 0.4
    slope = np.polyfit(np.abs(depth[far] - 0.4), row[far], 1)
    assert np.allclose(row[far], np.polyval(slope, np.abs(depth[far] - 0.4)), atol=1e-4)
    assert np.percentile(f, 95) == pytest.approx(4.0, rel=0.05)


def test_step_field_takes_only_two_values_and_blurs_the_far_side():
    d = ramp()
    f = blur_field(d, "step", d_focus=0.5, sigma_max=4.0)
    assert set(np.unique(f).tolist()) == {0.0, 4.0}
    assert (f[d < 0.4] == 4.0).all() and (f[d > 0.5] == 0.0).all()


def test_position_field_ignores_depth():
    a = blur_field(ramp(), "position", d_focus=0.5, sigma_max=4.0)
    b = blur_field(ramp()[:, ::-1].copy(), "position", d_focus=0.2, sigma_max=4.0)
    assert np.array_equal(a, b)
    assert a[0, 0] == pytest.approx(4.0, rel=0.05) and a[32, 0] < 0.2


def test_missing_foreground_keeps_everything_nearer_than_focus_sharp():
    d = ramp()
    law = blur_field(d, "lawful", d_focus=0.5, sigma_max=4.0)
    f = blur_field(d, "missing_foreground", d_focus=0.5, sigma_max=4.0)
    assert (f[d > 0.5] == 0).all()
    assert np.array_equal(f[d <= 0.5], law[d <= 0.5])


def test_constant_depth_map_gives_a_finite_field():
    for mode in MODES:
        f = blur_field(np.full((32, 32), 0.3, np.float32), mode, d_focus=0.3, sigma_max=4.0)
        assert np.isfinite(f).all()


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        blur_field(ramp(), "sideways", d_focus=0.5, sigma_max=4.0)


def test_zero_blur_field_returns_the_image_unchanged():
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 1, (48, 64, 3)).astype(np.float32)
    out = render_defocus(img, np.zeros((48, 64), np.float32))
    assert out.shape == img.shape
    assert np.abs(out - img).max() < 1e-3


def test_rendering_keeps_the_mean_light_energy():
    # Optical blur conserves light, so the mean is compared in linear light, not sRGB.
    rng = np.random.default_rng(1)
    img = rng.uniform(0, 1, (64, 96, 3)).astype(np.float32)
    out = render_defocus(img, blur_field(ramp(), "lawful", d_focus=0.4, sigma_max=4.0))
    before, after = srgb_to_linear(img).mean(), srgb_to_linear(out).mean()
    assert abs(after - before) / before < 0.02


def test_srgb_and_linear_conversions_are_inverses():
    c = np.linspace(0, 1, 101, dtype=np.float32)
    assert np.abs(linear_to_srgb(srgb_to_linear(c)) - c).max() < 1e-5
    assert srgb_to_linear(np.array([0.5], np.float32))[0] == pytest.approx(0.214, abs=0.002)


def test_more_blur_in_the_field_removes_more_detail():
    rng = np.random.default_rng(2)
    img = rng.uniform(0, 1, (64, 96, 3)).astype(np.float32)
    field = np.zeros((64, 96), np.float32)
    field[:, 48:] = 4.0
    out = render_defocus(img, field)
    assert out[:, 56:].std() < 0.5 * out[:, :40].std()


def test_render_is_the_same_when_the_device_is_named():
    rng = np.random.default_rng(3)
    img = rng.uniform(0, 1, (48, 64, 3)).astype(np.float32)
    field = blur_field(ramp(48, 64), "lawful", d_focus=0.4, sigma_max=3.0)
    assert np.array_equal(render_defocus(img, field), render_defocus(img, field, device="cpu"))
