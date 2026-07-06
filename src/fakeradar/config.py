"""Detector configuration: dataclasses, YAML loading, built-in tier presets."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class BranchConfig:
    name: str
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class FusionConfig:
    embed_dim: int = 256
    hidden_dim: int = 512
    dropout: float = 0.1


@dataclass
class DetectorConfig:
    tier: str = "fast"
    crop_size: int = 256
    max_crops: int = 4
    threshold_real: float = 0.35
    threshold_fake: float = 0.65
    branches: list[BranchConfig] = field(default_factory=list)
    fusion: FusionConfig = field(default_factory=FusionConfig)

    def enabled_branches(self) -> list[BranchConfig]:
        return [b for b in self.branches if b.enabled]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DetectorConfig:
        branches = [BranchConfig(**b) for b in d.get("branches", [])]
        fusion = FusionConfig(**d.get("fusion", {}))
        keys = {"tier", "crop_size", "max_crops", "threshold_real", "threshold_fake"}
        rest = {k: v for k, v in d.items() if k in keys}
        return cls(branches=branches, fusion=fusion, **rest)

    @classmethod
    def from_yaml(cls, path: str | Path) -> DetectorConfig:
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))


# ---------------------------------------------------------------------------
# Built-in presets (mirrored by configs/*.yaml; code is the source of truth so
# the installed package works without any external files).
# ---------------------------------------------------------------------------

_FAST = {
    "tier": "fast",
    "crop_size": 256,
    "max_crops": 4,
    "threshold_real": 0.35,
    "threshold_fake": 0.65,
    "branches": [
        {"name": "gradient", "enabled": True, "params": {"use_lgrad": False}},
        {"name": "frequency", "enabled": True, "params": {"n_bins": 128}},
        {"name": "semantic", "enabled": False, "params": {}},
    ],
    "fusion": {"embed_dim": 256, "hidden_dim": 512, "dropout": 0.1},
}

_ACCURATE = {
    **_FAST,
    "tier": "accurate",
    "max_crops": 8,
    "branches": [
        {"name": "gradient", "enabled": True, "params": {"use_lgrad": False}},
        {"name": "frequency", "enabled": True, "params": {"n_bins": 128}},
        {
            "name": "semantic",
            "enabled": True,
            "params": {"model_name": "vit_base_patch16_clip_224.openai"},
        },
    ],
}

TIER_PRESETS: dict[str, dict[str, Any]] = {"fast": _FAST, "accurate": _ACCURATE}


def get_preset(tier: str) -> DetectorConfig:
    if tier not in TIER_PRESETS:
        raise ValueError(f"Unknown tier {tier!r}. Available: {sorted(TIER_PRESETS)}")
    return DetectorConfig.from_dict(TIER_PRESETS[tier])
