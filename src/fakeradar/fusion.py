"""Fusion head.

Combines per-branch feature vectors into one calibrated real/AI logit.

Design choices:
  * Per-branch projection to a shared embedding, then a *learned softmax gate*
    over branches — the model learns how much to trust each forensic view,
    and the gate weights are exposed for interpretability.
  * Auxiliary per-branch logits (deep supervision). They stabilize training
    and let the CLI report "which branch fired" on every image.
  * A temperature buffer for post-hoc calibration (fit on a val set), so the
    output probability is meaningful, not just a ranking score.
"""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

T_MIN, T_MAX = 0.05, 20.0  # sane range for the calibration temperature


class FusionHead(nn.Module):
    def __init__(
        self,
        branch_dims: dict[str, int],
        embed_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.branch_names = list(branch_dims.keys())
        self.proj = nn.ModuleDict(
            {
                n: nn.Sequential(nn.Linear(d, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
                for n, d in branch_dims.items()
            }
        )
        self.aux_heads = nn.ModuleDict({n: nn.Linear(embed_dim, 1) for n in branch_dims})
        self.gate = nn.Parameter(torch.zeros(len(branch_dims)))
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.register_buffer("temperature", torch.ones(1))

    def forward(self, feats: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        z = {n: self.proj[n](f) for n, f in feats.items()}
        weights = torch.softmax(self.gate, dim=0)
        pooled = torch.stack([weights[i] * z[n] for i, n in enumerate(self.branch_names)], 0).sum(0)
        logit = self.mlp(pooled).squeeze(-1)
        branch_logits = {n: self.aux_heads[n](z[n]).squeeze(-1) for n in self.branch_names}
        return {
            "logit": logit,
            "branch_logits": branch_logits,
            "gate_weights": {n: weights[i] for i, n in enumerate(self.branch_names)},
        }

    # ---------------------------------------------------------- calibration
    @torch.no_grad()
    def probability(self, logit: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logit / self.temperature)

    @torch.no_grad()
    def fit_temperature(
        self, logits: torch.Tensor, labels: torch.Tensor, n_grid: int = 2001
    ) -> float:
        """Temperature scaling on held-out (logits, labels).

        The NLL is convex in 1/T, so a dense log-spaced grid over the allowed
        range [T_MIN, T_MAX] finds the global optimum deterministically
        (resolution ~0.3%). This replaces an unbounded LBFGS fit that diverged
        to T=NaN on perfectly separable held-out sets (where the optimum is
        T -> 0), silently turning every later prediction into NaN.
        """
        logits = logits.detach().float().flatten()
        labels = labels.detach().float().flatten().to(logits.device)
        if not torch.isfinite(logits).all():
            raise ValueError("Cannot calibrate: logits contain NaN/inf.")
        temps = torch.logspace(math.log10(T_MIN), math.log10(T_MAX), n_grid, device=logits.device)
        temps = temps.clamp(T_MIN, T_MAX)  # float32 logspace endpoints round just outside
        nll = torch.empty_like(temps)
        for i in range(0, n_grid, 256):  # chunked: bounded memory on large val sets
            t = temps[i : i + 256, None]
            nll[i : i + 256] = F.binary_cross_entropy_with_logits(
                logits[None, :] / t, labels.expand(t.shape[0], -1), reduction="none"
            ).mean(dim=1)
        best = int(torch.argmin(nll))
        if best in (0, n_grid - 1):
            warnings.warn(
                f"fitted temperature hit the bound T={float(temps[best]):g} "
                "(held-out set perfectly separable or tiny?); calibration is unreliable.",
                stacklevel=3,  # past the no_grad wrapper, to the caller
            )
        self.temperature.fill_(float(temps[best]))
        return float(self.temperature.item())
