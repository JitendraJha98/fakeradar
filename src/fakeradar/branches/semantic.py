"""Semantic branch — frozen foundation-model features.

CLIP/DINO-style ViT features generalize surprisingly well to *unseen*
generators (UniversalFakeDetect, Ojha et al. 2023, and successors): diffusion
images cluster measurably in foundation-model feature space even for models
never seen in training. We keep the backbone completely frozen — only the
fusion head learns — which is what preserves that generalization.

This branch is optional ("accurate" tier). It requires `pip install
fakeradar[semantic]` (timm) and a one-time weight download; the "fast" tier
runs without it, fully offline.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..constants import CLIP_MEAN, CLIP_STD, DEFAULT_SEMANTIC_SIZE
from .base import DetectionBranch, register_branch


@register_branch("semantic")
class SemanticBranch(DetectionBranch):
    input_kind = "resized"

    def __init__(self, model_name: str = "vit_base_patch16_clip_224.openai"):
        super().__init__()
        try:
            import timm
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "The semantic branch needs timm. Install with: pip install 'fakeradar[semantic]'"
            ) from e

        self.model_name = model_name
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        cfg = getattr(self.backbone, "pretrained_cfg", {}) or {}
        self.image_size = int((cfg.get("input_size") or (3, DEFAULT_SEMANTIC_SIZE, 0))[1])
        mean = torch.tensor(cfg.get("mean", CLIP_MEAN)).view(1, 3, 1, 1)
        std = torch.tensor(cfg.get("std", CLIP_STD)).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        self.feature_dim = int(self.backbone.num_features)
        # small trainable projection so the frozen features can adapt scale
        self.proj = nn.Sequential(nn.LayerNorm(self.feature_dim), nn.Linear(self.feature_dim, self.feature_dim))

    def preprocess(self, pil_image) -> torch.Tensor:
        """PIL RGB image -> (1, 3, S, S) tensor in [0,1] at the backbone's size."""
        from torchvision.transforms import functional as TF

        s = self.image_size
        img = TF.resize(pil_image, s, antialias=True)
        img = TF.center_crop(img, [s, s])
        return TF.to_tensor(img).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x in [0,1], already resized
        with torch.no_grad():
            feats = self.backbone((x - self.mean) / self.std)
        return self.proj(feats)

    def train(self, mode: bool = True):  # keep backbone frozen in train mode
        super().train(mode)
        self.backbone.eval()
        return self
