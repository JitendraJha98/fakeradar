"""High-level detection API.

    from fakeradar import Detector

    det = Detector(tier="fast", checkpoint="runs/fast_v0/best.pt")
    result = det.predict("photo.jpg")
    print(result.prob_ai, result.verdict, result.per_branch)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .config import DetectorConfig, get_preset
from .constants import VERDICT_FAKE, VERDICT_REAL, VERDICT_UNCERTAIN
from .models import FakeRadarModel, download_pretrained, load_checkpoint
from .preprocessing import extract_crops, load_image


@dataclass
class DetectionResult:
    source: str
    prob_ai: float
    verdict: str
    per_branch: dict[str, float]
    gate_weights: dict[str, float]
    n_crops: int
    latency_ms: float
    tier: str
    trained: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "prob_ai": round(self.prob_ai, 4),
            "verdict": self.verdict,
            "per_branch": {k: round(v, 4) for k, v in self.per_branch.items()},
            "gate_weights": {k: round(v, 4) for k, v in self.gate_weights.items()},
            "n_crops": self.n_crops,
            "latency_ms": round(self.latency_ms, 1),
            "tier": self.tier,
            "trained": self.trained,
            **self.extra,
        }


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Detector:
    """Loads a model tier + weights and scores images."""

    def __init__(
        self,
        tier: str = "fast",
        checkpoint: str | Path | None = None,
        pretrained: str | None = None,
        config: DetectorConfig | None = None,
        device: str = "auto",
        allow_random_init: bool = False,
    ):
        self.device = _auto_device() if device == "auto" else device
        self.trained = True

        if checkpoint is not None:
            self.model = load_checkpoint(checkpoint, map_location=self.device)
        elif pretrained is not None:
            self.model = load_checkpoint(download_pretrained(pretrained), map_location=self.device)
        else:
            cfg = config or get_preset(tier)
            if not allow_random_init:
                raise ValueError(
                    "No weights given. Pass checkpoint=..., pretrained=..., or set "
                    "allow_random_init=True for smoke tests (outputs will be meaningless)."
                )
            self.model = FakeRadarModel(cfg)
            self.trained = False

        self.config = self.model.config
        self.model.to(self.device).eval()

    # ------------------------------------------------------------- predict
    @torch.no_grad()
    def predict(self, source) -> DetectionResult:
        t0 = time.perf_counter()
        img = load_image(source)
        crops = extract_crops(img, self.config.crop_size, self.config.max_crops).to(self.device)

        semantic = None
        for name in self.model.resized_branch_names:
            semantic = self.model.branches[name].preprocess(img).to(self.device)
            break  # one global semantic view shared by resized branches

        out = self.model(crops, semantic)
        logit = out["logit"].mean()  # aggregate crops in logit space
        prob = float(torch.sigmoid(logit / self.model.fusion.temperature))
        per_branch = {n: float(torch.sigmoid(v.mean())) for n, v in out["branch_logits"].items()}
        gates = {n: float(w) for n, w in out["gate_weights"].items()}

        if prob >= self.config.threshold_fake:
            verdict = VERDICT_FAKE
        elif prob <= self.config.threshold_real:
            verdict = VERDICT_REAL
        else:
            verdict = VERDICT_UNCERTAIN

        return DetectionResult(
            source=str(getattr(source, "filename", None) or source),
            prob_ai=prob,
            verdict=verdict,
            per_branch=per_branch,
            gate_weights=gates,
            n_crops=crops.shape[0],
            latency_ms=(time.perf_counter() - t0) * 1000,
            tier=self.config.tier,
            trained=self.trained,
        )

    def predict_batch(self, sources: list) -> list[DetectionResult]:
        return [self.predict(s) for s in sources]
