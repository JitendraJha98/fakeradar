"""Guaranteed false-accusation rate: thresholds from real photos only."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from PIL import Image  # noqa: E402

from fakeradar.config import get_preset  # noqa: E402
from fakeradar.detector import Detector  # noqa: E402
from fakeradar.evaluation.guarantee import (  # noqa: E402
    Calibration,
    calibrate,
    calibrate_real_manifest,
    conformal_threshold,
    flag,
    load_calibration,
    save_calibration,
)


def test_threshold_is_the_right_order_statistic():
    assert conformal_threshold(np.arange(1, 100), 0.05) == 95
    assert conformal_threshold(np.arange(1, 100), 0.01) == 99
    assert conformal_threshold(np.arange(10), 0.01) == float("inf")


@pytest.mark.parametrize("alpha", [0.05, 0.01])
def test_flagged_share_of_exchangeable_real_photos_matches_the_level(alpha):
    rng = np.random.default_rng(0)
    scores, strata = rng.normal(size=20000), np.array(["a"] * 20000)
    cal = calibrate(scores[:10000], strata[:10000], alpha)
    assert abs(flag(cal, scores[10000:], strata[10000:]).mean() - alpha) < 0.004


def test_small_and_unseen_strata_use_the_pooled_threshold():
    rng = np.random.default_rng(1)
    scores = np.concatenate([rng.normal(size=5000), rng.normal(5, 1, size=50)])
    cal = calibrate(scores, np.array(["big"] * 5000 + ["small"] * 50), 0.01)
    assert cal.fallback == {"small"} and "small" not in cal.thresholds
    above, below = cal.pooled + 1, cal.pooled - 1
    assert flag(cal, np.array([above, below]), np.array(["small", "new"])).tolist() == [True, False]


def test_non_finite_scores_are_rejected_with_their_count():
    with pytest.raises(ValueError, match="2"):
        calibrate(np.array([0.1, np.nan, np.inf, 0.3]), np.array(["a"] * 4), 0.05)


def test_calibration_round_trips_through_json(tmp_path):
    cal = Calibration({"orig": 0.7, "whatsapp": 0.9}, 0.8, 0.01, {"tiny"})
    path = tmp_path / "fpr.json"
    save_calibration(cal, path, n=1234)
    data = json.loads(path.read_text())
    assert data["alpha"] == 0.01 and data["n"] == 1234 and data["fallback"] == ["tiny"]
    back = load_calibration(path)
    assert back.thresholds == cal.thresholds and back.pooled == 0.8 and back.fallback == {"tiny"}


@pytest.fixture(scope="module")
def detector():
    cfg = get_preset("fast")
    cfg.crop_size, cfg.max_crops = 64, 1
    torch.manual_seed(0)
    return Detector(config=cfg, allow_random_init=True, device="cpu")


def _manifest(tmp_path, labels):
    lines = ["path,label,generator"]
    for i, (label, stratum) in enumerate(labels):
        p = tmp_path / f"{i}.png"
        Image.effect_noise((96, 96), 10 + 9 * i).convert("RGB").save(p)
        lines.append(f"{p},{label},{stratum}")
    m = tmp_path / "real.csv"
    m.write_text("\n".join(lines) + "\n")
    return m


def test_real_manifest_gives_one_threshold_per_stratum(detector, tmp_path):
    m = _manifest(tmp_path, [(0, "orig")] * 5 + [(0, "whatsapp")] * 5)
    cal, n = calibrate_real_manifest(detector, m, alpha=0.25)
    assert n == 10 and set(cal.thresholds) == {"orig", "whatsapp"}
    assert 0.0 <= cal.pooled <= 1.0


def test_generated_images_in_the_calibration_manifest_are_refused(detector, tmp_path):
    m = _manifest(tmp_path, [(0, "orig")] * 3 + [(1, "sd14")])
    with pytest.raises(ValueError, match="real"):
        calibrate_real_manifest(detector, m, alpha=0.25)
