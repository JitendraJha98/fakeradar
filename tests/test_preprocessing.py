import pytest

torch = pytest.importorskip("torch")

from PIL import Image  # noqa: E402
from torchvision.transforms import functional as TF  # noqa: E402

from fakeradar.preprocessing import crop_positions, extract_crops  # noqa: E402


def test_grid_positions_order_bounds_uniqueness():
    w, h, size = 1000, 800, 256
    pos = crop_positions(w, h, size, max_crops=9)
    assert len(pos) == 9 == len(set(pos))  # full 3x3 grid, all unique
    assert pos[0] == ((w - size) // 2, (h - size) // 2)  # center first
    for x, y in pos:
        assert 0 <= x <= w - size and 0 <= y <= h - size


def test_positions_capped_and_deduped():
    assert len(crop_positions(1000, 800, 256, max_crops=4)) == 4
    # image exactly crop-sized: every grid cell collapses to (0, 0)
    assert crop_positions(256, 256, 256, max_crops=9) == [(0, 0)]
    assert len(crop_positions(1000, 800, 256, max_crops=0)) == 1  # floor of 1


def test_max_crops_8_is_real():
    """Regression: the old grid maxed out at 5 positions, silently ignoring
    the accurate tier's max_crops=8."""
    assert len(crop_positions(1000, 800, 256, max_crops=8)) == 8


def test_extract_crops_never_resamples_large_images():
    img = Image.effect_noise((600, 400), 64).convert("RGB")
    crops = extract_crops(img, size=256, max_crops=9)
    assert crops.shape == (9, 3, 256, 256)
    x, y = crop_positions(600, 400, 256, 9)[0]
    direct = TF.to_tensor(img.crop((x, y, x + 256, y + 256)))
    assert torch.equal(
        crops[0], direct
    ), "large-image crops must be pixel-identical (no resampling)"


def test_extract_crops_upscales_small_images_only():
    img = Image.new("RGB", (100, 80), (10, 20, 30))
    crops = extract_crops(img, size=256, max_crops=4)
    assert crops.shape[1:] == (3, 256, 256)
    assert crops.shape[0] >= 1
