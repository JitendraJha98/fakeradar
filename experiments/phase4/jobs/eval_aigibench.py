"""Kaggle entry (GPU, internet): one bounded evaluation on 2024-25 generators.

AIGIBench (Li et al. 2025, HorizonTEL/AIGIBench, CC-BY-NC-SA-4.0) test subsets.
For each subset: download the zip to /tmp, take a fixed sample of up to N real
and N generated images (sorted by a hash of the path), score them straight
from the zip with fakeradar fast-v0, accurate-v0 (published weights) and the
public Community Forensics model, then delete the zip. Writes per-image scores
and a summary table. Nothing is trained or tuned here.
"""

import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import time
import zipfile

import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision.transforms import functional as TF

from fakeradar.detector import Detector
from fakeradar.evaluation.metrics import summarize
from fakeradar.preprocessing import load_image_equalized

SUBSETS = [
    "FLUX1-dev",
    "SD3",
    "Midjourney",
    "DALLE-3",
    "Imagen3",
    "SDXL",
    "CommunityAI",
    "SocialRF",
]
N = 500
EQUALIZE = 96
OUT = "/kaggle/working"
TMP = "/tmp/aigibench"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(TMP, exist_ok=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm"], check=False)

subprocess.run(
    [
        "git",
        "clone",
        "-q",
        "--depth",
        "1",
        "https://github.com/JeongsooP/Community-Forensics",
        "/tmp/commfor",
    ],
    check=True,
)
sys.path.insert(0, "/tmp/commfor")
from models import ViTClassifier  # noqa: E402

commfor = (
    ViTClassifier.from_pretrained("OwensLab/commfor-model-384", device=DEVICE).to(DEVICE).eval()
)
detectors = {
    "fast": Detector(
        checkpoint=hf_hub_download("JitendraJha98/fakeradar-fast", "fast_v0.pt"), device=DEVICE
    ),
    "accurate": Detector(
        checkpoint=hf_hub_download("JitendraJha98/fakeradar-accurate", "accurate_v0.pt"),
        device=DEVICE,
    ),
}


def commfor_prob(img: Image.Image) -> float:
    x = TF.center_crop(TF.resize(img.convert("RGB"), 440), [384, 384])
    x = TF.normalize(TF.to_tensor(x), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    with torch.no_grad():
        return float(torch.sigmoid(commfor(x[None].to(DEVICE)).flatten()[0]))


def label_of(name: str):
    low = name.lower()
    if "0_real" in low or "/real/" in low:
        return 0
    if "1_fake" in low or "/fake/" in low:
        return 1
    return None


def sample(names: list[str], n: int) -> list[str]:
    return sorted(names, key=lambda s: hashlib.sha256(s.encode()).hexdigest())[:n]


report = {"subsets": {}, "n_per_class": N, "equalize": EQUALIZE}
rows = []
for subset in SUBSETS:
    t0 = time.time()
    entry = report["subsets"].setdefault(subset, {})
    try:
        path = hf_hub_download(
            "HorizonTEL/AIGIBench", f"test/{subset}.zip", repo_type="dataset", local_dir=TMP
        )
        entry["download_s"] = int(time.time() - t0)
        zf = zipfile.ZipFile(path)
        names = [
            n
            for n in zf.namelist()
            if n.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp"))
        ]
        by_label = {
            0: [n for n in names if label_of(n) == 0],
            1: [n for n in names if label_of(n) == 1],
        }
        entry["available"] = {str(k): len(v) for k, v in by_label.items()}
        entry["example_names"] = names[:3]
        picked = [(n, lab) for lab in (0, 1) for n in sample(by_label[lab], N)]
        for name, lab in picked:
            data = zf.read(name)
            try:
                with Image.open(io.BytesIO(data)) as im:
                    img = im.copy()
                    img.format = im.format
            except Exception as e:  # noqa: BLE001
                entry.setdefault("unreadable", []).append(f"{name}: {type(e).__name__}")
                continue
            rec = {
                "subset": subset,
                "path": name,
                "label": lab,
                "format": img.format,
                "w": img.width,
                "h": img.height,
            }
            rec["commfor"] = commfor_prob(img)
            for tier, det in detectors.items():
                rec[f"{tier}_raw"] = det.predict(img.convert("RGB")).prob_ai
                rec[f"{tier}_eq"] = det.predict(load_image_equalized(img, EQUALIZE)).prob_ai
            rows.append(rec)
        zf.close()
        os.remove(path)
        sub = [r for r in rows if r["subset"] == subset]
        y = [r["label"] for r in sub]
        for key in ("fast_raw", "fast_eq", "accurate_raw", "accurate_eq", "commfor"):
            m = summarize(y, [r[key] for r in sub])
            entry[key] = {"auroc": m["auroc"], "tpr@5%fpr": m["tpr@5%fpr"], "acc@0.5": m["acc@0.5"]}
        entry["n"] = {"real": y.count(0), "fake": y.count(1)}
        entry["formats"] = {
            str(lab): sorted({r["format"] for r in sub if r["label"] == lab}) for lab in (0, 1)
        }
    except Exception as e:  # noqa: BLE001
        entry["error"] = f"{type(e).__name__}: {e}"[:400]
    entry["seconds"] = int(time.time() - t0)
    print(subset, json.dumps(entry)[:600], flush=True)
    with open(os.path.join(OUT, "aigibench_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    with open(os.path.join(OUT, "aigibench_scores.csv"), "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
print("done")
