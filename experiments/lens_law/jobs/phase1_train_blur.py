"""Kaggle entry (GPU): train blur estimator v1.

Sharp development-split photos (f/8 or narrower, downscaled so they are sharp
at the pixel level) get synthetic defocus through lenslaw.synth; the network
learns the log blur size and its own uncertainty. No generated image and no
test-split photo is used. Writes blur_net.pt and train_report.json.
"""

import csv
import json
import os
import time

import numpy as np
import torch
from lenslaw.blur_net import BlurNet, nll_loss
from lenslaw.synth import make_training_patch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

OUT = "/kaggle/working"
PATCH = 96
SOURCE = 2 * PATCH + 48
LONG_SIDE = 768  # photos are shrunk to this first, so in-focus detail is pixel-sharp
STEPS = 8000  # about 0.5M patches; patch synthesis on 4 CPU cores sets the pace
BATCH = 64
LR = 1e-3
N_VAL = 2000
EVAL_EVERY = 1000
BINS = ((-0.7, 0.0), (0.0, 0.7), (0.7, 1.4), (1.4, 2.6))  # log blur: ~0.5-1, 1-2, 2-4, 4-13 px
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def find_file(name: str) -> str:
    for base, _dirs, files in os.walk("/kaggle/input"):
        if name in files:
            return os.path.join(base, name)
    raise FileNotFoundError(name)


def load_small(path: str) -> np.ndarray | None:
    with Image.open(path) as im:
        im = im.convert("RGB")
        scale = LONG_SIDE / max(im.size)
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
        arr = np.asarray(im, dtype=np.uint8)
    return arr if min(arr.shape[:2]) >= SOURCE else None


class Patches(Dataset):
    """Deterministic stream of (patch, log blur) pairs: item i always uses the same seed."""

    def __init__(self, arrays: list[np.ndarray], length: int, seed: int):
        self.arrays, self.length, self.seed = arrays, length, seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int):
        rng = np.random.default_rng([self.seed, i])
        img = self.arrays[int(rng.integers(len(self.arrays)))]
        y = int(rng.integers(0, img.shape[0] - SOURCE + 1))
        x = int(rng.integers(0, img.shape[1] - SOURCE + 1))
        crop = img[y : y + SOURCE, x : x + SOURCE].astype(np.float32) / 255.0
        patch, label = make_training_patch(crop, rng, out=PATCH)
        return torch.from_numpy(patch).permute(2, 0, 1), torch.tensor(label, dtype=torch.float32)


manifest_path = find_file("train_manifest.csv")
img_dir = os.path.join(os.path.dirname(manifest_path), "unsplash_train")
with open(manifest_path, encoding="utf-8") as f:
    manifest = list(csv.DictReader(f))
arrays = {"train": [], "val": []}
for row in manifest:
    arr = load_small(os.path.join(img_dir, row["photo_id"] + ".jpg"))
    if arr is not None:
        arrays[row["role"]].append(arr)
print({k: len(v) for k, v in arrays.items()}, "photos loaded", flush=True)

val_x, val_y = zip(*[Patches(arrays["val"], N_VAL, seed=999)[i] for i in range(N_VAL)], strict=True)
val_x, val_y = torch.stack(val_x), torch.stack(val_y)

loader = DataLoader(
    Patches(arrays["train"], STEPS * BATCH, seed=1),
    batch_size=BATCH,
    num_workers=4,
    persistent_workers=True,
    pin_memory=DEVICE == "cuda",
)
model = BlurNet().to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS, pct_start=0.05)
scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")


@torch.no_grad()
def evaluate() -> dict:
    model.eval()
    preds = []
    for i in range(0, N_VAL, 250):
        with torch.autocast(DEVICE, enabled=DEVICE == "cuda"):
            preds.append(model(val_x[i : i + 250].to(DEVICE)).float().cpu())
    pred = torch.cat(preds)
    mid = pred.shape[-1] // 2
    mean, std = pred[:, 0, mid, mid], torch.exp(0.5 * pred[:, 1, mid, mid])
    err = (mean - val_y).abs()
    sure = std <= 0.5
    out = {
        "nll": float(nll_loss(pred, val_y)),
        "mae": float(err.mean()),
        "sure_share": float(sure.float().mean()),
        "mae_sure": float(err[sure].mean()) if sure.any() else float("nan"),
        "mean_std": float(std.mean()),
        "rmse_over_std_sure": float(((mean - val_y)[sure] / std[sure]).pow(2).mean().sqrt())
        if sure.any()
        else float("nan"),
    }
    for lo, hi in BINS:
        pick = (val_y >= lo) & (val_y < hi)
        if pick.any():
            out[f"mae_{lo}_{hi}"] = float(err[pick].mean())
            out[f"sure_{lo}_{hi}"] = float(sure[pick].float().mean())
    model.train()
    return out


history = []
t0 = time.time()
model.train()
running = 0.0
for step, (x, y) in enumerate(loader, start=1):
    x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
    with torch.autocast(DEVICE, enabled=DEVICE == "cuda"):
        loss = nll_loss(model(x).float(), y)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    scaler.step(opt)
    scaler.update()
    sched.step()
    running += loss.item()
    if step % EVAL_EVERY == 0 or step == STEPS:
        metrics = {
            "step": step,
            "train_nll": running / EVAL_EVERY,
            "seconds": int(time.time() - t0),
        }
        metrics.update(evaluate())
        history.append(metrics)
        running = 0.0
        print(json.dumps(metrics), flush=True)
        torch.save(model.state_dict(), os.path.join(OUT, "blur_net.pt"))
        with open(os.path.join(OUT, "train_report.json"), "w") as f:
            json.dump(
                {
                    "device": DEVICE,
                    "photos": {k: len(v) for k, v in arrays.items()},
                    "history": history,
                },
                f,
                indent=1,
            )
print("done", flush=True)
