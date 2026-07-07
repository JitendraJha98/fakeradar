import random

import numpy as np
import pytest
from PIL import Image

from fakeradar.training.augment import RobustnessAugment


def test_identity_when_all_probabilities_zero():
    aug = RobustnessAugment(p_jpeg=0, p_webp=0, p_resize=0, p_blur=0, p_noise=0, p_hflip=0)
    img = Image.effect_noise((120, 90), 40).convert("RGB")
    out = aug(img)
    assert np.array_equal(np.asarray(out), np.asarray(img))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_output_is_rgb_and_size_sane(seed):
    random.seed(seed)
    np.random.seed(seed)
    aug = RobustnessAugment()
    img = Image.effect_noise((300, 200), 40).convert("RGB")
    for _ in range(10):
        out = aug(img)
        assert out.mode == "RGB"
        w, h = out.size
        assert w >= 64 and h >= 64
        # resize_scale is (0.5, 1.5): dimensions stay within those bounds
        assert w <= 300 * 1.5 + 1 and h <= 200 * 1.5 + 1


def test_jpeg_recompression_changes_pixels():
    random.seed(0)
    aug = RobustnessAugment(
        p_jpeg=1.0, jpeg_quality=(30, 30), p_webp=0, p_resize=0, p_blur=0, p_noise=0, p_hflip=0
    )
    img = Image.effect_noise((128, 128), 60).convert("RGB")
    out = aug(img)
    assert out.size == img.size
    assert not np.array_equal(np.asarray(out), np.asarray(img))
