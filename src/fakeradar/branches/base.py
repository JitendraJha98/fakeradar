"""Branch base class + registry.

A *branch* is one forensic view of the image. Every branch maps an image
tensor to a fixed-size feature vector; the fusion head combines them.

Adding a new branch = subclass DetectionBranch, decorate with
@register_branch("yourname"), enable it in a config. Nothing else changes.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

BRANCH_REGISTRY: dict[str, type[DetectionBranch]] = {}


def register_branch(name: str):
    def deco(cls: type[DetectionBranch]) -> type[DetectionBranch]:
        cls.name = name
        BRANCH_REGISTRY[name] = cls
        return cls

    return deco


class DetectionBranch(nn.Module):
    """Base class for forensic branches.

    Attributes:
        name: registry key.
        feature_dim: dimension of the output feature vector.
        input_kind: "crop" -> receives native-resolution crops (forensic
            branches must NOT get resized inputs, resizing destroys the
            high-frequency traces they rely on). "resized" -> receives the
            branch's own preprocessed global view (semantic branches).
    """

    name: str = "base"
    feature_dim: int = 0
    input_kind: str = "crop"

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,3,H,W) in [0,1]
        raise NotImplementedError


def build_branch(name: str, **params: Any) -> DetectionBranch:
    if name not in BRANCH_REGISTRY:
        raise KeyError(f"Unknown branch {name!r}. Registered: {sorted(BRANCH_REGISTRY)}")
    return BRANCH_REGISTRY[name](**params)
