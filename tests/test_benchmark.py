import csv

import pytest

torch = pytest.importorskip("torch")

from PIL import Image  # noqa: E402

from fakeradar.config import get_preset  # noqa: E402
from fakeradar.detector import Detector  # noqa: E402
from fakeradar.evaluation.benchmark import (  # noqa: E402
    PERTURBATIONS,
    run_benchmark,
    run_robustness,
)


@pytest.fixture(scope="module")
def detector():
    cfg = get_preset("fast")
    cfg.crop_size, cfg.max_crops = 64, 1
    torch.manual_seed(0)
    return Detector(config=cfg, allow_random_init=True, device="cpu")


@pytest.fixture
def manifest(tmp_path):
    rows = [("real", 0), ("real", 0), ("sd14", 1), ("sd14", 1), ("mj", 1)]
    lines = ["path,label,generator"]
    for i, (gen, label) in enumerate(rows):
        p = tmp_path / f"{i}.png"
        Image.effect_noise((96, 96), 10 + 20 * i).convert("RGB").save(p)
        lines.append(f"{p},{label},{gen}")
    m = tmp_path / "test.csv"
    m.write_text("\n".join(lines) + "\n")
    return m


def test_benchmark_table_and_raw_scores(detector, manifest, tmp_path):
    md_path, scores = tmp_path / "lb.md", tmp_path / "scores.csv"
    md = run_benchmark(detector, manifest, md_path, scores)
    assert md_path.read_text().strip() == md.strip()
    header = md.splitlines()[0]
    for col in ("AUROC", "AP", "acc@0.5", "TPR@5%FPR", "ECE", "abstain"):
        assert col in header
    assert "| mj | 1 |" in md and "| sd14 | 2 |" in md
    assert "| ALL | 3 |" in md and "MEAN (2 gen)" in md

    with open(scores, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 5 and {"path", "label", "generator", "prob_ai"} <= set(rows[0])


def test_robustness_table_and_raw_scores(detector, manifest, tmp_path):
    scores = tmp_path / "rob.csv"
    md = run_robustness(detector, manifest, scores_csv=scores)
    for name in PERTURBATIONS:
        assert f"| {name} |" in md
    with open(scores, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 5 * len(PERTURBATIONS)


def _scores(path):
    with open(path, newline="") as f:
        return [float(r["prob_ai"]) for r in csv.DictReader(f)]


def test_benchmark_equalized_changes_png_scores_and_says_so(detector, manifest, tmp_path):
    raw, eq = tmp_path / "raw.csv", tmp_path / "eq.csv"
    md_raw = run_benchmark(detector, manifest, scores_csv=raw)
    md_eq = run_benchmark(detector, manifest, scores_csv=eq, equalize_jpeg_quality=96)
    assert _scores(raw) != _scores(eq)  # every source in this manifest is a PNG
    assert "JPEG quality 96" in md_eq and "not equalized" in md_raw


def test_benchmark_equalized_leaves_jpeg_sources_alone(detector, tmp_path):
    lines = ["path,label,generator"]
    for i, (gen, label) in enumerate([("real", 0), ("sd14", 1)]):
        p = tmp_path / f"{i}.jpg"
        Image.effect_noise((96, 96), 30 + 40 * i).convert("RGB").save(p, quality=90)
        lines.append(f"{p},{label},{gen}")
    m = tmp_path / "jpeg.csv"
    m.write_text("\n".join(lines) + "\n")
    raw, eq = tmp_path / "raw.csv", tmp_path / "eq.csv"
    run_benchmark(detector, m, scores_csv=raw)
    run_benchmark(detector, m, scores_csv=eq, equalize_jpeg_quality=96)
    assert _scores(raw) == _scores(eq)


def test_robustness_accepts_the_equalize_setting(detector, manifest, tmp_path):
    md = run_robustness(detector, manifest, equalize_jpeg_quality=96)
    assert "JPEG quality 96" in md
