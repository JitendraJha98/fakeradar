"""Kaggle entry: one-minute check that a job starts and sees its inputs."""

import json
import os

import torch

report = {
    "cuda": torch.cuda.is_available(),
    "gpus": torch.cuda.device_count(),
    "inputs": sorted(
        os.path.relpath(os.path.join(base, d), "/kaggle/input")
        for base, dirs, _files in os.walk("/kaggle/input")
        for d in dirs
        if base.count(os.sep) <= 4
    )[:40],
}
print(json.dumps(report, indent=1))
with open("/kaggle/working/smoke.json", "w") as f:
    json.dump(report, f, indent=1)
