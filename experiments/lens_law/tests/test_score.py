import numpy as np
import pytest
from lenslaw.depth import MODEL_ID, normalize_inverse_depth
from lenslaw.fit import SCORE_KEYS
from lenslaw.render import MODES, blur_field, render_defocus
from lenslaw.score import WORK_LONG_SIDE, load_rgb, score_arrays, to_gray
from lenslaw.transport import CONDITIONS, apply_condition
from PIL import Image


def make_scene(seed: int, h: int = 384, w: int = 512):
    """Sharp random-rectangle texture, with depth that ramps left to right plus a near box."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 0.5, np.float32)
    for _ in range(160):
        rh, rw = rng.integers(24, 96, 2)
        y, x = rng.integers(0, h - 8), rng.integers(0, w - 8)
        img[y : y + rh, x : x + rw] = rng.uniform(0.05, 0.95, 3)
    depth = np.tile(np.linspace(0, 1, w, dtype=np.float32), (h, 1))
    depth[h // 3 : 2 * h // 3, w // 8 : w // 4] = 0.9
    return img, depth


def test_lawful_blur_fits_the_lens_law_better_than_each_unlawful_blur():
    wins = dict.fromkeys(MODES[1:], 0)
    for seed in range(10):
        img, depth = make_scene(seed)
        resid = {}
        for mode in MODES:
            field = blur_field(depth, mode, d_focus=0.5, sigma_max=4.0)
            resid[mode] = score_arrays(render_defocus(img, field), depth)["resid_lens"]
        for mode in wins:
            wins[mode] += resid["lawful"] < resid[mode]
    assert all(count >= 8 for count in wins.values()), wins


def test_lawful_render_is_applicable_and_has_every_score_key():
    img, depth = make_scene(0)
    out = score_arrays(render_defocus(img, blur_field(depth, "lawful", 0.5, 4.0)), depth)
    assert set(out) == set(SCORE_KEYS) | {"applicable"}
    assert out["applicable"] is True
    assert out["r2_lens"] > 0.5


def test_flat_image_is_not_applicable_and_does_not_raise():
    out = score_arrays(np.full((256, 256, 3), 0.5, np.float32), np.zeros((256, 256), np.float32))
    assert out["applicable"] is False
    assert out["n"] == 0


def test_depth_map_at_another_resolution_is_resized_to_the_image():
    img, depth = make_scene(1)
    rendered = render_defocus(img, blur_field(depth, "lawful", 0.5, 4.0))
    small = score_arrays(rendered, depth[::2, ::2])
    assert small["applicable"] is True
    assert small["resid_lens"] == pytest.approx(
        score_arrays(rendered, depth)["resid_lens"], rel=0.25
    )


def test_to_gray_uses_luma_weights():
    rgb = np.zeros((2, 2, 3), np.float32)
    rgb[..., 1] = 1.0
    assert to_gray(rgb)[0, 0] == pytest.approx(0.587, abs=1e-3)


@pytest.mark.parametrize("mode", ["L", "RGBA", "P", "CMYK"])
def test_load_rgb_converts_other_modes_to_float_rgb(mode):
    img = Image.new("RGB", (60, 40), (255, 128, 0)).convert(mode)
    arr = load_rgb(img)
    assert arr.shape == (40, 60, 3) and arr.dtype == np.float32
    assert 0.0 <= arr.min() and arr.max() <= 1.0


def test_load_rgb_scales_sixteen_bit_images_and_reads_from_a_path(tmp_path):
    path = tmp_path / "deep.png"
    Image.fromarray(np.full((40, 60), 40000, np.uint16)).save(path)
    arr = load_rgb(path)
    assert arr.shape == (40, 60, 3)
    assert arr.mean() == pytest.approx(40000 / 65535, abs=0.01)


def test_load_rgb_caps_the_long_side():
    assert WORK_LONG_SIDE == 1536
    assert load_rgb(Image.new("RGB", (3000, 2000))).shape == (1024, 1536, 3)


def test_conditions_are_the_three_from_the_spec():
    assert CONDITIONS == ("clean", "s075_q85", "s050_q70")


def test_transport_conditions_scale_both_sides():
    img = Image.new("RGB", (400, 300), (10, 200, 30))
    assert apply_condition(img, "clean").size == (400, 300)
    assert apply_condition(img, "s075_q85").size == (300, 225)
    assert apply_condition(img, "s050_q70").size == (200, 150)


def test_unknown_transport_condition_is_rejected():
    with pytest.raises(ValueError):
        apply_condition(Image.new("RGB", (8, 8)), "s010_q01")


def test_normalised_depth_maps_percentiles_to_zero_and_one():
    raw = np.linspace(5.0, 25.0, 1001, dtype=np.float32).reshape(1, -1)
    out = normalize_inverse_depth(raw)
    assert np.percentile(out, 2) == pytest.approx(0.0, abs=1e-4)
    assert np.percentile(out, 98) == pytest.approx(1.0, abs=1e-4)


def test_normalising_a_constant_depth_map_gives_zeros():
    out = normalize_inverse_depth(np.full((8, 8), 3.0, np.float32))
    assert out.shape == (8, 8) and (out == 0).all()


def test_depth_model_is_the_apache_licensed_small_one():
    assert MODEL_ID == "depth-anything/Depth-Anything-V2-Small-hf"


def test_scoring_is_the_same_when_the_device_is_named():
    img, depth = make_scene(2)
    rendered = render_defocus(img, blur_field(depth, "lawful", 0.5, 4.0))
    assert score_arrays(rendered, depth) == score_arrays(rendered, depth, device="cpu")
