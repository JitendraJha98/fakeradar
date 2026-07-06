"""Frequency branch — spectral fingerprints.

Up-sampling in generative decoders creates periodic patterns and abnormal
high-frequency energy in the Fourier domain (the classic "GAN grid" and its
diffusion-era relatives). This branch computes an azimuthally-averaged
log-power spectrum (radial profile) plus a few band-energy statistics, and
learns on that compact representation with a small MLP.

Cheap (<1M params), fully deterministic front-end, strong complement to the
spatial gradient branch.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import DetectionBranch, register_branch


@register_branch("frequency")
class FrequencyBranch(DetectionBranch):
    feature_dim = 256
    input_kind = "crop"

    def __init__(self, n_bins: int = 128):
        super().__init__()
        self.n_bins = n_bins
        self._cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        self.mlp = nn.Sequential(
            nn.Linear(n_bins + 4, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
        )

    # ------------------------------------------------------------- spectrum
    def _radial_index(self, h: int, w: int, device: torch.device):
        key = (h, w, str(device))
        if key not in self._cache:
            cy, cx = h / 2.0, w / 2.0
            yy = torch.arange(h, device=device).view(-1, 1).float() - cy
            xx = torch.arange(w, device=device).view(1, -1).float() - cx
            r = torch.sqrt(yy * yy + xx * xx)
            r = r / (r.max() + 1e-8)
            idx = (r * (self.n_bins - 1)).round().long().flatten()  # (H*W,)
            counts = torch.bincount(idx, minlength=self.n_bins).clamp(min=1).float()
            self._cache[key] = (idx, counts)
        return self._cache[key]

    def radial_profile(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (profile (B, n_bins), stats (B, 4)) from images in [0,1]."""
        gray = x.mean(dim=1)  # (B,H,W)
        spec = torch.fft.fftshift(torch.fft.fft2(gray), dim=(-2, -1))
        mag = torch.log1p(spec.abs())  # (B,H,W)
        B, H, W = mag.shape

        idx, counts = self._radial_index(H, W, mag.device)
        flat = mag.reshape(B, -1)
        prof = torch.zeros(B, self.n_bins, device=mag.device, dtype=flat.dtype)
        prof.scatter_add_(1, idx.unsqueeze(0).expand(B, -1), flat)
        prof = prof / counts

        nb = self.n_bins
        low = prof[:, : nb // 4].mean(dim=1)
        mid = prof[:, nb // 4 : (3 * nb) // 4].mean(dim=1)
        hi = prof[:, (3 * nb) // 4 :].mean(dim=1)
        total = prof.mean(dim=1) + 1e-8
        stats = torch.stack([hi / total, mid / total, hi - low, prof.std(dim=1)], dim=1)

        prof = (prof - prof.mean(dim=1, keepdim=True)) / (prof.std(dim=1, keepdim=True) + 1e-6)
        return prof, stats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prof, stats = self.radial_profile(x)
        return self.mlp(torch.cat([prof, stats], dim=1))
