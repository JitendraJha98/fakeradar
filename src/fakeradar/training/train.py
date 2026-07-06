"""Training entrypoint.

    fakeradar train configs/train_fast.yaml

Single-GPU (or CPU for smoke tests). AMP, cosine LR with warmup, main BCE
loss + auxiliary per-branch BCE, best checkpoint selected by val AUROC.
"""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from ..config import DetectorConfig
from ..evaluation.metrics import average_precision, roc_auc
from ..models import FakeRadarModel, save_checkpoint
from .datasets import ManifestDataset


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _cosine_warmup(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


@torch.no_grad()
def evaluate(model: FakeRadarModel, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    scores, labels = [], []
    for batch in loader:
        crop = batch["crop"].to(device)
        semantic = batch.get("semantic")
        semantic = semantic.to(device) if semantic is not None else None
        out = model(crop, semantic)
        scores.append(torch.sigmoid(out["logit"]).cpu())
        labels.append(batch["label"])
    s = torch.cat(scores).numpy()
    y = torch.cat(labels).numpy()
    return {
        "auroc": roc_auc(y, s),
        "ap": average_precision(y, s),
        "acc@0.5": float(((s >= 0.5) == (y >= 0.5)).mean()),
    }


def train(config_path: str | Path) -> Path:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    det_cfg = DetectorConfig.from_yaml(cfg["detector"])
    data, optim, run = cfg["data"], cfg["optim"], cfg["run"]
    _seed(run.get("seed", 42))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FakeRadarModel(det_cfg).to(device)

    semantic_pre = None
    for name in model.resized_branch_names:
        semantic_pre = model.branches[name].preprocess
        break

    ds_train = ManifestDataset(
        data["train_manifest"], det_cfg.crop_size, augment=True, semantic_preprocess=semantic_pre
    )
    ds_val = ManifestDataset(
        data["val_manifest"], det_cfg.crop_size, augment=False, semantic_preprocess=semantic_pre
    )
    dl_train = DataLoader(
        ds_train,
        batch_size=optim["batch_size"],
        shuffle=True,
        num_workers=data.get("num_workers", 4),
        pin_memory=(device == "cuda"),
        drop_last=True,
    )
    dl_val = DataLoader(ds_val, batch_size=optim["batch_size"], num_workers=data.get("num_workers", 4))

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=optim["lr"], weight_decay=optim["weight_decay"])
    total_steps = optim["epochs"] * len(dl_train)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: _cosine_warmup(s, optim.get("warmup_steps", 0), total_steps)
    )
    scaler = torch.cuda.amp.GradScaler(enabled=optim.get("amp", True) and device == "cuda")
    bce = torch.nn.BCEWithLogitsLoss()
    aux_w = optim.get("aux_loss_weight", 0.3)

    out_dir = Path(run["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "metrics.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_auroc", "val_ap", "val_acc"])

    best_auroc, best_path = -1.0, out_dir / "best.pt"
    step = 0
    for epoch in range(optim["epochs"]):
        model.train()
        running = 0.0
        for batch in dl_train:
            crop = batch["crop"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            semantic = batch.get("semantic")
            semantic = semantic.to(device, non_blocking=True) if semantic is not None else None

            with torch.autocast(device_type=device, enabled=scaler.is_enabled()):
                out = model(crop, semantic)
                loss = bce(out["logit"], label)
                for bl in out["branch_logits"].values():
                    loss = loss + aux_w * bce(bl, label) / len(out["branch_logits"])

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if optim.get("grad_clip"):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, optim["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            step += 1
            if step % run.get("log_every", 50) == 0:
                print(f"epoch {epoch} step {step}/{total_steps} loss {running / run.get('log_every', 50):.4f}")
                running = 0.0

        metrics = evaluate(model, dl_val, device)
        print(f"[val] epoch {epoch}: {metrics}")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, f"{loss.item():.4f}", metrics["auroc"], metrics["ap"], metrics["acc@0.5"]]
            )
        save_checkpoint(model, out_dir / "last.pt", extra={"epoch": epoch, "val": metrics})
        if metrics["auroc"] > best_auroc:
            best_auroc = metrics["auroc"]
            save_checkpoint(model, best_path, extra={"epoch": epoch, "val": metrics})
            print(f"  ↳ new best AUROC {best_auroc:.4f} -> {best_path}")

    print(f"Done. Best val AUROC {best_auroc:.4f}. Checkpoint: {best_path}")
    return best_path
