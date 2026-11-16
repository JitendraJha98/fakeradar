"""Kaggle entry (CPU, internet): confirm the two things Phase 3 depends on.

1. The reference detector (Community Forensics, MIT) loads and scores images.
Writes smoke3_report.json. Keeps no images.
"""

import json
import os
import subprocess
import sys
import time

import torch
from PIL import Image

report = {}
OUT = "/kaggle/working"

# ---- reference detector ----
try:
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/JeongsooP/Community-Forensics",
            "/tmp/commfor",
        ],
        check=True,
        capture_output=True,
    )
    sys.path.insert(0, "/tmp/commfor")
    import dataprocessor_hf as dphf
    import models

    processor = dphf.CommForImageProcessor.from_pretrained(
        "OwensLab/commfor-data-preprocessor", size=384
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.ViTClassifier.from_pretrained("OwensLab/commfor-model-384", device=device)
    model = model.to(device).eval()
    names = sorted(os.listdir("/tmp/commfor/test_images"))[:5]
    imgs = [Image.open(os.path.join("/tmp/commfor/test_images", n)).convert("RGB") for n in names]
    t0 = time.time()
    with torch.no_grad():
        batch = processor(imgs, mode="test")["pixel_values"]
        logits = model(batch.to(device)).flatten().cpu()
    report["detector"] = {
        "ok": True,
        "input_shape": list(batch.shape),
        "generated_test_images_prob": [round(float(p), 4) for p in torch.sigmoid(logits)],
        "device": device,
        "seconds_for_5": round(time.time() - t0, 2),
        "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
    }
    real_dir = None
    for base, _dirs, files in os.walk("/kaggle/input"):
        if "unsplash_manifest.csv" in files:
            real_dir = os.path.join(base, "unsplash")
    if real_dir:
        reals = [
            Image.open(os.path.join(real_dir, n)).convert("RGB")
            for n in sorted(os.listdir(real_dir))[:20]
        ]
        with torch.no_grad():
            pixels = processor(reals, mode="test")["pixel_values"].to(device)
            probs = torch.sigmoid(model(pixels).flatten()).cpu()
        report["detector"]["real_unsplash_prob"] = [round(float(p), 4) for p in probs]
except Exception as e:  # noqa: BLE001
    report["detector"] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:600]}
print(json.dumps(report, indent=1), flush=True)

print(json.dumps(report, indent=1)[:6000])
with open(os.path.join(OUT, "smoke3_report.json"), "w") as f:
    json.dump(report, f, indent=1)
