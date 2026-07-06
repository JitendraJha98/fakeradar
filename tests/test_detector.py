import pytest

torch = pytest.importorskip("torch")

from PIL import Image  # noqa: E402

from fakeradar.detector import Detector  # noqa: E402
from fakeradar.models import FakeRadarModel, load_checkpoint, save_checkpoint  # noqa: E402
from fakeradar.config import get_preset  # noqa: E402


def test_detector_smoke(tmp_path):
    det = Detector(tier="fast", allow_random_init=True, device="cpu")
    img = Image.new("RGB", (600, 400), (120, 90, 200))
    r = det.predict(img)
    assert 0.0 <= r.prob_ai <= 1.0
    assert r.n_crops >= 1
    assert set(r.per_branch) == {"gradient", "frequency"}
    d = r.to_dict()
    assert d["trained"] is False


def test_checkpoint_roundtrip(tmp_path):
    model = FakeRadarModel(get_preset("fast"))
    p = tmp_path / "ck.pt"
    save_checkpoint(model, p)
    model2 = load_checkpoint(p)
    assert model2.config.tier == "fast"
