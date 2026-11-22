import pytest

torch = pytest.importorskip("torch")

from PIL import Image  # noqa: E402

from fakeradar.config import get_preset  # noqa: E402
from fakeradar.detector import Detector  # noqa: E402
from fakeradar.models import FakeRadarModel, load_checkpoint, save_checkpoint  # noqa: E402


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


def test_max_crops_8_used_on_large_images():
    """Regression: crop grid used to cap at 5, silently ignoring max_crops=8."""
    cfg = get_preset("fast")
    cfg.max_crops = 8
    det = Detector(config=cfg, allow_random_init=True, device="cpu")
    img = Image.effect_noise((900, 700), 40).convert("RGB")
    assert det.predict(img).n_crops == 8


def test_checkpoint_metadata_and_format_check(tmp_path):
    from fakeradar import __version__
    from fakeradar.models import read_checkpoint

    p = tmp_path / "ck.pt"
    save_checkpoint(FakeRadarModel(get_preset("fast")), p, extra={"epoch": 1})
    ckpt = read_checkpoint(p)
    assert ckpt["fakeradar_version"] == __version__ and ckpt["epoch"] == 1

    bogus = tmp_path / "bogus.pt"
    torch.save({"state_dict": {}}, bogus)
    with pytest.raises(ValueError, match="not a fakeradar checkpoint"):
        load_checkpoint(bogus)


def test_checkpoint_with_nan_temperature_is_rejected(tmp_path):
    model = FakeRadarModel(get_preset("fast"))
    model.fusion.temperature.fill_(float("nan"))
    p = tmp_path / "bad_t.pt"
    save_checkpoint(model, p)
    with pytest.raises(ValueError, match="invalid calibration temperature"):
        load_checkpoint(p)


def test_fast_v0_zoo_entry_points_at_the_published_repo():
    from fakeradar.models import MODEL_ZOO

    assert MODEL_ZOO["fast-v0"] == {
        "repo_id": "JitendraJha98/fakeradar-fast",
        "filename": "fast_v0.pt",
    }
