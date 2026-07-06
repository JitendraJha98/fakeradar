"""Gradient-field branch — fakeradar's signature forensic view.

Generative decoders (GAN or diffusion) synthesize images through stacks of
up-sampling + convolution. That process leaves statistical traces in the
*local gradient structure* of the image that natural camera pipelines do not
produce. This branch makes those traces explicit before learning on them:

  1. SRM high-pass residuals   — classic steganalysis filters that suppress
                                 content and keep sensor/synthesis noise.
  2. NPR residual              — image minus its own down-up resampled copy
                                 (Neighboring Pixel Relationships style),
                                 which highlights up-sampling artifacts.
  3. Gradient magnitude        — Scharr gradient field of the luma channel.
  4. Gradient coherence        — structure-tensor coherence of that field;
                                 synthetic textures show anomalous local
                                 gradient alignment.
  5. (optional) LGrad channel  — gradients of a pretrained CNN w.r.t. the
                                 input (Tan et al., CVPR 2023 style). Needs a
                                 torchvision weight download, so off by
                                 default to keep the fast tier fully offline.

The stacked residual maps feed a small ResNet-18 encoder (~11M params).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..constants import IMAGENET_MEAN, IMAGENET_STD
from .base import DetectionBranch, register_branch


def _pad_to_5x5(k: torch.Tensor) -> torch.Tensor:
    pad = (5 - k.shape[-1]) // 2
    return F.pad(k, (pad, pad, pad, pad))


def _srm_kernels() -> torch.Tensor:
    """Four fixed high-pass kernels, shape (4, 1, 5, 5)."""
    kv = torch.tensor(
        [
            [-1.0, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1],
        ]
    ) / 12.0
    lap = _pad_to_5x5(torch.tensor([[0.0, 1, 0], [1, -4, 1], [0, 1, 0]]))
    sx = _pad_to_5x5(torch.tensor([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 4.0)
    sy = _pad_to_5x5(torch.tensor([[-1.0, -2, -1], [0, 0, 0], [1, 2, 1]]) / 4.0)
    return torch.stack([kv, lap, sx, sy]).unsqueeze(1)


def _scharr_kernels() -> tuple[torch.Tensor, torch.Tensor]:
    gx = torch.tensor([[-3.0, 0, 3], [-10, 0, 10], [-3, 0, 3]]) / 16.0
    gy = gx.t().contiguous()
    return gx.view(1, 1, 3, 3), gy.view(1, 1, 3, 3)


@register_branch("gradient")
class GradientFieldBranch(DetectionBranch):
    feature_dim = 512
    input_kind = "crop"

    def __init__(self, use_lgrad: bool = False):
        super().__init__()
        self.use_lgrad = use_lgrad

        srm = _srm_kernels()  # (4,1,5,5)
        self.register_buffer("srm", srm.repeat(3, 1, 1, 1))  # (12,1,5,5), depthwise x3ch
        gx, gy = _scharr_kernels()
        self.register_buffer("scharr_x", gx)
        self.register_buffer("scharr_y", gy)

        in_ch = 12 + 3 + 1 + 1 + (3 if use_lgrad else 0)  # srm + npr + gmag + coherence (+lgrad)
        from torchvision.models import resnet18

        enc = resnet18(weights=None)
        enc.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        enc.fc = nn.Identity()
        self.encoder = enc

        self._lgrad_net: nn.Module | None = None
        if use_lgrad:
            self._init_lgrad()

    # ------------------------------------------------------------------ lgrad
    def _init_lgrad(self) -> None:
        from torchvision.models import ResNet18_Weights, resnet18

        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
        for p in net.parameters():
            p.requires_grad_(False)
        self._lgrad_net = net
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("_lg_mean", mean)
        self.register_buffer("_lg_std", std)

    def _lgrad_map(self, x: torch.Tensor) -> torch.Tensor:
        """Input-gradient of a frozen pretrained CNN (LGrad-style transform)."""
        assert self._lgrad_net is not None
        with torch.enable_grad():
            xg = x.detach().clone().requires_grad_(True)
            logits = self._lgrad_net((xg - self._lg_mean) / self._lg_std)
            score = torch.logsumexp(logits, dim=1).sum()
            (grad,) = torch.autograd.grad(score, xg)
        g = grad.detach()
        return g / (g.flatten(1).abs().amax(dim=1).view(-1, 1, 1, 1) + 1e-8)

    # ---------------------------------------------------------------- stacks
    def forensic_stack(self, x: torch.Tensor) -> torch.Tensor:
        """Build the (B, C, H, W) stack of gradient-domain residual maps."""
        B, C, H, W = x.shape
        srm = F.conv2d(x, self.srm, padding=2, groups=3)  # (B,12,H,W)

        down = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
        up = F.interpolate(down, size=(H, W), mode="bilinear", align_corners=False)
        npr = x - up  # (B,3,H,W)

        gray = x.mean(dim=1, keepdim=True)
        gx = F.conv2d(gray, self.scharr_x, padding=1)
        gy = F.conv2d(gray, self.scharr_y, padding=1)
        gmag = torch.sqrt(gx * gx + gy * gy + 1e-12)

        jxx = F.avg_pool2d(gx * gx, 7, stride=1, padding=3)
        jyy = F.avg_pool2d(gy * gy, 7, stride=1, padding=3)
        jxy = F.avg_pool2d(gx * gy, 7, stride=1, padding=3)
        coherence = torch.sqrt((jxx - jyy) ** 2 + 4 * jxy * jxy + 1e-12) / (jxx + jyy + 1e-6)

        maps = [srm, npr, gmag, coherence]
        if self.use_lgrad:
            maps.append(self._lgrad_map(x))
        return torch.cat(maps, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.forensic_stack(x))
