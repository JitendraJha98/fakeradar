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

import torch
import torch.nn as nn


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

    def fit_temperature(self, logits: torch.Tensor, labels: torch.Tensor, steps: int = 200) -> float:
        """Simple temperature scaling on held-out (logits, labels)."""
        log_t = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=steps)
        bce = nn.BCEWithLogitsLoss()

        def closure():
            opt.zero_grad()
            loss = bce(logits / log_t.exp(), labels.float())
            loss.backward()
            return loss

        opt.step(closure)
        self.temperature.fill_(float(log_t.exp().item()))
        return float(self.temperature.item())
