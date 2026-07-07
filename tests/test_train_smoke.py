"""End-to-end CPU training smoke test on synthetic data: exercises the full
train -> validate -> checkpoint -> calibrate pipeline in seconds."""

import csv

import pytest

torch = pytest.importorskip("torch")

import yaml  # noqa: E402
from PIL import Image  # noqa: E402

from fakeradar.models import load_checkpoint  # noqa: E402
from fakeradar.training.train import train  # noqa: E402

TINY_DETECTOR = {
    "tier": "fast",
    "crop_size": 64,
    "max_crops": 2,
    "threshold_real": 0.35,
    "threshold_fake": 0.65,
    "branches": [
        {"name": "gradient", "enabled": True, "params": {"use_lgrad": False}},
        {"name": "frequency", "enabled": True, "params": {"n_bins": 32}},
        {"name": "semantic", "enabled": False, "params": {}},
    ],
    "fusion": {"embed_dim": 32, "hidden_dim": 64, "dropout": 0.0},
}


def _make_dataset(root, n_per_class=2, size=(96, 96)):
    rows = []
    for i in range(n_per_class * 2):
        label = i % 2
        p = root / f"img_{i}.png"
        # different noise stats per class so the task is not degenerate
        Image.effect_noise(size, 20 if label == 0 else 80).convert("RGB").save(p)
        rows.append((str(p), label, "synthetic" if label else "real"))
    return rows


def _write_manifest(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "generator"])
        w.writerows(rows)


def test_train_end_to_end_with_calibration(tmp_path):
    rows = _make_dataset(tmp_path, n_per_class=3)
    train_csv, val_csv = tmp_path / "train.csv", tmp_path / "val.csv"
    _write_manifest(train_csv, rows)
    _write_manifest(val_csv, rows)

    det_yaml = tmp_path / "tiny.yaml"
    det_yaml.write_text(yaml.safe_dump(TINY_DETECTOR))
    train_yaml = tmp_path / "train_config.yaml"
    train_yaml.write_text(
        yaml.safe_dump(
            {
                "detector": str(det_yaml),
                "data": {
                    "train_manifest": str(train_csv),
                    "val_manifest": str(val_csv),
                    "num_workers": 0,
                },
                "optim": {
                    "epochs": 1,
                    "batch_size": 2,
                    "lr": 1.0e-3,
                    "weight_decay": 0.0,
                    "warmup_steps": 0,
                    "amp": False,
                    "aux_loss_weight": 0.3,
                },
                "run": {
                    "out_dir": str(tmp_path / "run"),
                    "seed": 0,
                    "log_every": 1000,
                    "calibrate": True,
                },
            }
        )
    )

    best = train(train_yaml)
    assert best.exists()

    # metrics.csv: header + one epoch row with the full paper-metric columns
    with open(tmp_path / "run" / "metrics.csv", newline="") as f:
        rows_csv = list(csv.reader(f))
    assert rows_csv[0][:3] == ["epoch", "train_loss", "val_auroc"]
    assert "val_ece" in rows_csv[0]
    assert len(rows_csv) == 2

    # checkpoint reloads and carries the fitted calibration
    model = load_checkpoint(best)
    assert model.config.crop_size == 64
    ckpt = torch.load(best, map_location="cpu", weights_only=True)
    cal = ckpt.get("calibration")
    assert cal is not None and cal["temperature"] > 0
    assert float(model.fusion.temperature) == pytest.approx(cal["temperature"], rel=1e-5)
