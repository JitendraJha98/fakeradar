import io

from PIL import Image
from score_fodb_local import decode, role_of


def jpeg_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (640, 480), (90, 120, 30)).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def test_a_good_photo_decodes_to_the_detector_input():
    name, size, tensor = decode(("D01_x/orig/D01_img_orig_0001.jpg", jpeg_bytes()))
    assert size == (640, 480) and tuple(tensor.shape) == (3, 384, 384)


def test_a_truncated_photo_is_reported_as_unreadable_not_raised():
    name, size, tensor = decode(("D26_x/twitter/D26_img_twitter_0082.jpg", jpeg_bytes()[:-400]))
    assert name.endswith("0082.jpg") and size is None and tensor is None


def test_roles_are_fixed_by_the_phone_id():
    assert role_of("D02") == role_of("D02")
    assert {role_of(f"D{i:02d}") for i in range(1, 28)} == {"cal", "audit"}
