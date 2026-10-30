"""Training entrypoint.

    fakeradar train configs/train_fast.yaml
    fakeradar train configs/train_fast.yaml --resume runs/fast_v0/last.pt

Single-GPU (or CPU for smoke tests). AMP, cosine LR with warmup, main BCE
loss + auxiliary per-branch BCE, best checkpoint selected by val AUROC.
`last.pt` carries optimizer/scheduler/scaler state so an interrupted run
(preempted Colab/cloud GPU) resumes where it stopped.
"""

from __future__ import annotations

import csv
import math
import random
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from ..config import TIER_PRESETS, DetectorConfig, get_preset
from ..evaluation.metrics import summarize
from ..models import FakeRadarModel, read_checkpoint, save_checkpoint
from .datasets import ManifestDataset, require_both_classes

_MAX_CONSECUTIVE_BAD_STEPS = 10


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _cosine_warmup(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


def resolve_detector_config(spec: Any, config_dir: Path) -> DetectorConfig:
    """The training config's `detector:` key may be a detector YAML path
    (relative to the CWD, else to the training config's own directory), an
    inline mapping, or a tier preset name ("fast"/"accurate" — works from a
    pip install with no configs/ directory)."""
    if isinstance(spec, dict):
        return DetectorConfig.from_dict(spec)
    path = Path(str(spec))
    if path.is_file():
        return DetectorConfig.from_yaml(path)
    if not path.is_absolute() and (config_dir / path).is_file():
        return DetectorConfig.from_yaml(config_dir / path)
    if str(spec) in TIER_PRESETS:
        return get_preset(str(spec))
    raise FileNotFoundError(
        f"detector: {spec!r} is neither a detector YAML nor a preset {sorted(TIER_PRESETS)}"
    )


def _resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


@torch.no_grad()
def evaluate(model: FakeRadarModel, loader: DataLoader, device: str) -> dict[str, float]:
    """Full paper-protocol metric set (AUROC, AP, acc, TPR@5%FPR, ECE, ...)."""
    model.eval()
    scores, labels = [], []
    for batch in loader:
        crop = batch["crop"].to(device)
        semantic = batch.get("semantic")
        semantic = semantic.to(device) if semantic is not None else None
        out = model(crop, semantic)
        scores.append(torch.sigmoid(out["logit"]).float().cpu())
        labels.append(batch["label"])
    s = torch.cat(scores).numpy()
    y = torch.cat(labels).numpy()
    return summarize(y, s)


def train(config_path: str | Path, resume: str | Path | None = None) -> Path:
    config_path = Path(config_path)
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    det_cfg = resolve_detector_config(cfg["detector"], config_path.parent)
    data, optim, run = cfg["data"], cfg["optim"], cfg["run"]
    _seed(run.get("seed", 42))

    device = _resolve_device(run.get("device", "auto"))
    device_type = torch.device(device).type  # "cuda:1" -> "cuda" for autocast/GradScaler
    model = FakeRadarModel(det_cfg).to(device)
    semantic_pre = model.semantic_preprocessor()
    num_workers = data.get("num_workers", 4)

    ds_train = ManifestDataset(
        data["train_manifest"], det_cfg.crop_size, augment=True, semantic_preprocess=semantic_pre
    )
    ds_val = ManifestDataset(
        data["val_manifest"], det_cfg.crop_size, augment=False, semantic_preprocess=semantic_pre
    )
    require_both_classes(ds_train.rows, f"train manifest {data['train_manifest']}")
    require_both_classes(ds_val.rows, f"val manifest {data['val_manifest']}")
    dl_train = DataLoader(
        ds_train,
        batch_size=optim["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device_type == "cuda"),
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    dl_val = DataLoader(ds_val, batch_size=optim["batch_size"], num_workers=num_workers)
    if len(dl_train) == 0:
        raise ValueError(
            f"train manifest has {len(ds_train)} rows < batch_size {optim['batch_size']} "
            "(drop_last=True) — nothing to train on."
        )

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=optim["lr"], weight_decay=optim["weight_decay"])
    total_steps = optim["epochs"] * len(dl_train)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: _cosine_warmup(s, optim.get("warmup_steps", 0), total_steps)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=optim.get("amp", True) and device_type == "cuda")
    bce = torch.nn.BCEWithLogitsLoss()
    aux_w = optim.get("aux_loss_weight", 0.3)

    out_dir = Path(run["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "metrics.csv"
    best_auroc, best_path = -1.0, out_dir / "best.pt"
    start_epoch, step = 0, 0

    if resume is not None:
        ckpt = read_checkpoint(resume, map_location=device)
        if ckpt["config"] != det_cfg.to_dict():
            raise ValueError(f"{resume} was trained with a different detector config.")
        if "optimizer" not in ckpt:
            raise ValueError(f"{resume} has no optimizer state — resume from a run's last.pt.")
        model.load_state_dict(ckpt["state_dict"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, step = ckpt["epoch"] + 1, ckpt["step"]
        best_auroc = ckpt.get("best_auroc", -1.0)
        print(f"resumed from {resume}: epoch {start_epoch}, step {step}, best AUROC {best_auroc}")
    if resume is None or not log_path.exists():
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(
                [
                    "epoch",
                    "train_loss",
                    "val_auroc",
                    "val_ap",
                    "val_acc",
                    "val_tpr@5%fpr",
                    "val_ece",
                ]
            )

    log_every = run.get("log_every", 50)
    bad_steps = 0
    for epoch in range(start_epoch, optim["epochs"]):
        model.train()
        running, epoch_loss, epoch_batches = 0.0, 0.0, 0
        for batch in dl_train:
            crop = batch["crop"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            semantic = batch.get("semantic")
            semantic = semantic.to(device, non_blocking=True) if semantic is not None else None

            with torch.autocast(device_type=device_type, enabled=scaler.is_enabled()):
                out = model(crop, semantic)
                loss = bce(out["logit"], label)
                for bl in out["branch_logits"].values():
                    loss = loss + aux_w * bce(bl, label) / len(out["branch_logits"])

            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):  # one fp16 overflow must not poison the weights
                bad_steps += 1
                warnings.warn(f"non-finite loss at step {step}; batch skipped", stacklevel=1)
                if bad_steps >= _MAX_CONSECUTIVE_BAD_STEPS:
                    raise RuntimeError(
                        f"{bad_steps} consecutive non-finite losses — training diverged "
                        "(try a lower lr or amp: false)."
                    )
                continue
            bad_steps = 0
            scaler.scale(loss).backward()
            if optim.get("grad_clip"):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, optim["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            epoch_loss += loss.item()
            epoch_batches += 1
            step += 1
            if step % log_every == 0:
                print(f"epoch {epoch} step {step}/{total_steps} loss {running / log_every:.4f}")
                running = 0.0

        metrics = evaluate(model, dl_val, device)
        print(f"[val] epoch {epoch}: {metrics}")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    epoch,
                    f"{epoch_loss / max(1, epoch_batches):.4f}",
                    metrics["auroc"],
                    metrics["ap"],
                    metrics["acc@0.5"],
                    metrics["tpr@5%fpr"],
                    metrics["ece"],
                ]
            )
        if not math.isnan(metrics["auroc"]) and metrics["auroc"] > best_auroc:
            best_auroc = metrics["auroc"]
            save_checkpoint(model, best_path, extra={"epoch": epoch, "val": metrics})
            print(f"  ↳ new best AUROC {best_auroc:.4f} -> {best_path}")
        save_checkpoint(
            model,
            out_dir / "last.pt",
            extra={
                "epoch": epoch,
                "step": step,
                "val": metrics,
                "best_auroc": best_auroc,
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "scaler": scaler.state_dict(),
            },
        )

    if not best_path.exists():
        raise RuntimeError("No epoch produced a valid val AUROC; no best.pt was written.")

    if run.get("calibrate", True):
        from ..evaluation.calibrate import calibrate_checkpoint

        stats = calibrate_checkpoint(
            best_path, data["val_manifest"], num_workers=num_workers, device=device
        )
        print(
            f"[calibration] T={stats['temperature']:.3f} "
            f"ECE {stats['ece_before']:.4f} -> {stats['ece_after']:.4f}"
        )

    print(f"Done. Best val AUROC {best_auroc:.4f}. Checkpoint: {best_path}")
    return best_path
