import numpy as np
import pytest
from fprcal.prep import CROP, RESIZE, commfor_preprocess, parse_fodb_name
from PIL import Image


def test_preprocessing_matches_the_detectors_test_transform():
    assert (RESIZE, CROP) == (440, 384)
    out = commfor_preprocess(Image.new("RGB", (900, 600), (128, 128, 128)))
    assert tuple(out.shape) == (3, 384, 384)
    expected = (128 / 255 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    assert np.allclose(out.mean(dim=(1, 2)).numpy(), expected, atol=1e-4)


def test_preprocessing_takes_the_centre_of_the_shorter_side():
    img = Image.new("RGB", (880, 440), (0, 0, 0))
    img.paste((255, 255, 255), (0, 0, 440, 440))  # left half white
    out = commfor_preprocess(img)
    assert out[0, :, :150].mean() > 1.0 and out[0, :, -150:].mean() < -1.0


def test_non_rgb_images_are_converted():
    assert tuple(commfor_preprocess(Image.new("L", (500, 500), 10)).shape) == (3, 384, 384)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("D01_Motorola_E3_1/orig/D01_img_orig_0001.jpg", ("D01", "orig", "0001")),
        (
            "fodb/D17_Apple_iPhone7_1/whatsapp/D17_img_whatsapp_0143.jpg",
            ("D17", "whatsapp", "0143"),
        ),
        ("D05_Apple_iPhone6s_1/facebook/D05_img_facebook_0007.JPG", ("D05", "facebook", "0007")),
    ],
)
def test_phone_platform_and_scene_are_read_from_the_file_name(name, expected):
    assert parse_fodb_name(name) == expected


def test_names_that_are_not_fodb_photos_are_skipped():
    assert parse_fodb_name("inspection/compare_devices/scene_orig_0001.jpg") is None
    assert parse_fodb_name("README.md") is None
