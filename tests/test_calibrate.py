import pytest

torch = pytest.importorskip("torch")

from PIL import Image  # noqa: E402

from fakeradar.config import get_preset  # noqa: E402
from fakeradar.detector import Detector  # noqa: E402
from fakeradar.evaluation.calibrate import calibrate_checkpoint, collect_logits  # noqa: E402
from fakeradar.models import FakeRadarModel, read_checkpoint, save_checkpoint  # noqa: E402


def _tiny_model():
    cfg = get_preset("fast")
    cfg.crop_size, cfg.max_crops = 64, 4
    cfg.fusion.embed_dim, cfg.fusion.hidden_dim = 32, 64
    torch.manual_seed(0)
    return FakeRadarModel(cfg).eval()


def _manifest(tmp_path, n=6):
    lines = ["path,label,generator"]
    for i in range(n):
        p = tmp_path / f"im{i}.png"
        Image.effect_noise((150 + 10 * i, 120), 20 + 15 * i).convert("RGB").save(p)
        lines.append(f"{p},{i % 2},{'sd' if i % 2 else 'real'}")
    m = tmp_path / "val.csv"
    m.write_text("\n".join(lines) + "\n")
    return m


def test_collect_logits_matches_detector_protocol(tmp_path):
    """The temperature must be fit on the exact quantity Detector.predict
    rescales: the mean fused logit over the native-res crop grid."""
    model = _tiny_model()
    manifest = _manifest(tmp_path)
    logits, labels = collect_logits(model, manifest)
    assert logits.shape == labels.shape == (6,)

    ckpt = tmp_path / "m.pt"
    save_checkpoint(model, ckpt)
    det = Detector(checkpoint=ckpt, device="cpu")
    for i in range(6):
        p = det.predict(tmp_path / f"im{i}.png").prob_ai
        assert p == pytest.approx(float(torch.sigmoid(logits[i])), abs=1e-5)


def test_calibrate_checkpoint_preserves_metadata(tmp_path):
    model = _tiny_model()
    ckpt = tmp_path / "best.pt"
    save_checkpoint(model, ckpt, extra={"epoch": 3, "val": {"auroc": 0.9}})
    stats = calibrate_checkpoint(ckpt, _manifest(tmp_path))

    saved = read_checkpoint(ckpt)
    assert saved["epoch"] == 3 and saved["val"] == {"auroc": 0.9}
    assert saved["calibration"]["temperature"] == pytest.approx(stats["temperature"])
    assert stats["n"] == 6
