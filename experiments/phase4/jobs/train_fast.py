"""Kaggle entry (GPU): train fakeradar's fast tier on GenImage SD v1.4, then benchmark.

Protocol: train on Stable Diffusion v1.4 only; GenImage's `val` split of every
generator is the test set; model selection and calibration use a 5% hold-out
of the training images. Both classes get the same compression history
(equalize_jpeg_quality = 96), which closes the "JPEG means real" shortcut.
Sizes below are set per run; the report records what was used.
"""

import json
import os
import random
import shutil
import time
from pathlib import Path

import torch
import yaml

from fakeradar.config import get_preset
from fakeradar.detector import Detector
from fakeradar.evaluation.benchmark import run_benchmark, run_robustness
from fakeradar.training.genimage import discover_generators, rows_for, stratified_holdout
from fakeradar.training.train import train

N_TRAIN_PER_CLASS = 50000
N_TEST_AI_PER_GENERATOR = 1000
N_TEST_REAL = 2000
N_ROBUST_PER_CLASS = 500
EPOCHS = 5
EQUALIZE = 96
SEED = 42  # training seed; the data subset is the same for every seed
USE_COHERENCE = True  # False = the coherence ablation
TRAIN_GENERATOR = "stable_diffusion_v_1_4"
OUT = Path("/kaggle/working")
rng = random.Random(0)
report = {"sizes": {}, "seconds": {}, "torch": torch.__version__, "gpu": torch.cuda.is_available()}


def find_root(marker: str) -> Path:
    """The GenImage folder that has `<marker>` below it (e.g. 'stable_diffusion_v_1_4/train')."""
    for base, dirs, _files in os.walk("/kaggle/input"):
        if "GenImage" in dirs and (Path(base) / "GenImage" / marker).is_dir():
            return Path(base) / "GenImage"
        if base.count(os.sep) > 7:
            dirs[:] = []
    raise FileNotFoundError(marker)


def write(path: Path, rows: list[list]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("path,label,generator\n")
        f.writelines(f"{p},{label},{gen}\n" for p, label, gen in rows)


t0 = time.time()
train_root = find_root(f"{TRAIN_GENERATOR}/train")
rows = rows_for(train_root, TRAIN_GENERATOR, "train")
by_label = {label: [r for r in rows if r[1] == label] for label in (0, 1)}
report["sizes"]["train_available"] = {str(k): len(v) for k, v in by_label.items()}
picked = [
    r
    for label in (0, 1)
    for r in rng.sample(by_label[label], min(N_TRAIN_PER_CLASS, len(by_label[label])))
]
train_rows, val_rows = stratified_holdout(picked, 0.05, seed=0)

val_root = find_root("ADM/val")
test_rows, real_pool = [], []
for gen in discover_generators(val_root, "val"):
    gen_rows = rows_for(val_root, gen, "val")
    ai = [r for r in gen_rows if r[1] == 1]
    test_rows += rng.sample(ai, min(N_TEST_AI_PER_GENERATOR, len(ai)))
    real_pool += [r for r in gen_rows if r[1] == 0]
test_rows += rng.sample(real_pool, min(N_TEST_REAL, len(real_pool)))
robust_rows = [
    r
    for label in (0, 1)
    for r in rng.sample([x for x in test_rows if x[1] == label], N_ROBUST_PER_CLASS)
]
manifests = OUT / "manifests"
manifests.mkdir(exist_ok=True)
for name, part in (
    ("train", train_rows),
    ("val", val_rows),
    ("test", test_rows),
    ("robust", robust_rows),
):
    write(manifests / f"{name}.csv", part)
    report["sizes"][name] = len(part)
report["seconds"]["manifests"] = int(time.time() - t0)
report["test_generators"] = sorted({r[2] for r in test_rows})
print(json.dumps(report, indent=1), flush=True)

detector = get_preset("fast").to_dict()
if not USE_COHERENCE:
    detector["branches"][0]["params"]["use_coherence"] = False
report["seed"], report["use_coherence"] = SEED, USE_COHERENCE
config = {
    "detector": detector,
    "data": {
        "train_manifest": str(manifests / "train.csv"),
        "val_manifest": str(manifests / "val.csv"),
        "num_workers": 4,
        "equalize_jpeg_quality": EQUALIZE,
    },
    "optim": {
        "epochs": EPOCHS,
        "batch_size": 64,
        "lr": 2.0e-4,
        "weight_decay": 0.05,
        "warmup_steps": 500,
        "amp": True,
        "grad_clip": 1.0,
        "aux_loss_weight": 0.3,
    },
    "run": {
        "out_dir": str(OUT / "run"),
        "device": "auto",
        "seed": SEED,
        "log_every": 50,
        "calibrate": True,
    },
}
(OUT / "train_config.yaml").write_text(yaml.safe_dump(config))
t0 = time.time()
best = train(OUT / "train_config.yaml")
report["seconds"]["train"] = int(time.time() - t0)
report["train_images_per_second"] = round(
    len(train_rows) * EPOCHS / max(1, report["seconds"]["train"]), 1
)
print(json.dumps(report, indent=1), flush=True)

det = Detector(checkpoint=best)
t0 = time.time()
for tag, quality in (("raw", None), ("equalized", EQUALIZE)):
    run_benchmark(
        det,
        manifests / "test.csv",
        OUT / f"benchmark_{tag}.md",
        OUT / f"benchmark_{tag}_scores.csv",
        quality,
    )
report["seconds"]["benchmark"] = int(time.time() - t0)
t0 = time.time()
run_robustness(
    det,
    manifests / "robust.csv",
    OUT / "robustness_equalized.md",
    OUT / "robustness_scores.csv",
    EQUALIZE,
)
report["seconds"]["robustness"] = int(time.time() - t0)

shutil.copy(OUT / "run" / "metrics.csv", OUT / "metrics.csv")
with open(OUT / "phase4_report.json", "w") as f:
    json.dump(report, f, indent=1)
print(json.dumps(report, indent=1))
print((OUT / "benchmark_equalized.md").read_text())
