"""Model assembly, checkpoints, model zoo, ONNX export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .branches import build_branch
from .config import DetectorConfig
from .fusion import FusionHead


class FakeRadarModel(nn.Module):
    """All enabled branches + fusion head as one trainable module."""

    def __init__(self, config: DetectorConfig):
        super().__init__()
        self.config = config
        self.branches = nn.ModuleDict(
            {b.name: build_branch(b.name, **b.params) for b in config.enabled_branches()}
        )
        if not self.branches:
            raise ValueError("At least one branch must be enabled.")
        dims = {n: br.feature_dim for n, br in self.branches.items()}
        f = config.fusion
        self.fusion = FusionHead(dims, f.embed_dim, f.hidden_dim, f.dropout)

    @property
    def crop_branch_names(self) -> list[str]:
        return [n for n, b in self.branches.items() if b.input_kind == "crop"]

    @property
    def resized_branch_names(self) -> list[str]:
        return [n for n, b in self.branches.items() if b.input_kind == "resized"]

    def forward(self, crop: torch.Tensor, semantic: torch.Tensor | None = None) -> dict[str, Any]:
        feats: dict[str, torch.Tensor] = {}
        for name in self.crop_branch_names:
            feats[name] = self.branches[name](crop)
        for name in self.resized_branch_names:
            if semantic is None:
                raise ValueError(f"Branch {name!r} needs the `semantic` input tensor.")
            sem_feat = self.branches[name](semantic)
            if sem_feat.shape[0] == 1 and crop.shape[0] > 1:  # one global view, many crops
                sem_feat = sem_feat.expand(crop.shape[0], -1)
            feats[name] = sem_feat
        return self.fusion(feats)


# ------------------------------------------------------------------ ckpt io
def save_checkpoint(model: FakeRadarModel, path: str | Path, extra: dict | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "fakeradar.v1",
            "config": model.config.to_dict(),
            "state_dict": model.state_dict(),
            **(extra or {}),
        },
        path,
    )


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> FakeRadarModel:
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    config = DetectorConfig.from_dict(ckpt["config"])
    model = FakeRadarModel(config)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if missing or unexpected:
        print(
            f"[fakeradar] checkpoint loaded with missing={len(missing)} unexpected={len(unexpected)} keys"
        )
    return model


# ---------------------------------------------------------------- model zoo
MODEL_ZOO: dict[str, dict[str, str]] = {
    # Published after v0.1 training runs — see ROADMAP.md.
    "fast-v0": {"repo_id": "jitendra-jha/fakeradar-fast", "filename": "fast_v0.pt"},
    "accurate-v0": {"repo_id": "jitendra-jha/fakeradar-accurate", "filename": "accurate_v0.pt"},
}


def download_pretrained(name: str) -> Path:
    if name not in MODEL_ZOO:
        raise KeyError(f"Unknown zoo model {name!r}. Available: {sorted(MODEL_ZOO)}")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Downloading zoo weights needs huggingface_hub. Install: pip install 'fakeradar[hub]'"
        ) from e
    entry = MODEL_ZOO[name]
    return Path(hf_hub_download(repo_id=entry["repo_id"], filename=entry["filename"]))


# --------------------------------------------------------------- onnx export
class _ExportWrapper(nn.Module):
    """Crop-only forward that outputs the fused probability (semantic branch
    excluded from ONNX v0 — export the fast tier)."""

    def __init__(self, model: FakeRadarModel):
        super().__init__()
        if model.resized_branch_names:
            raise ValueError("ONNX export currently supports crop-only (fast tier) models.")
        self.model = model

    def forward(self, crop: torch.Tensor) -> torch.Tensor:
        out = self.model(crop)
        return torch.sigmoid(out["logit"] / self.model.fusion.temperature)


def export_onnx(
    model: FakeRadarModel, path: str | Path, crop_size: int = 256, opset: int = 18
) -> Path:
    """Export the fast-tier model to ONNX for CPU/edge deployment.

    Needs torch>=2.9 (dynamo exporter — the frequency branch's FFT has no
    TorchScript-exporter mapping). A warmup forward runs first so the radial-
    index cache is populated and its `bincount` is baked in as a constant
    instead of hitting an unsupported ONNX op.
    """
    major, minor = (int(v) for v in torch.__version__.split(".")[:2])
    if (major, minor) < (2, 9):
        raise RuntimeError(
            f"ONNX export needs torch>=2.9 (dynamo exporter); found {torch.__version__}."
        )
    wrapper = _ExportWrapper(model).eval()
    dummy = torch.rand(1, 3, crop_size, crop_size)
    with torch.no_grad():
        wrapper(dummy)  # warm the frequency branch's radial-index cache
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(path),
        input_names=["crop"],
        output_names=["prob_ai"],
        dynamic_shapes={"crop": {0: "batch"}},
        opset_version=opset,
        dynamo=True,
    )
    return path


def verify_onnx_parity(
    model: FakeRadarModel,
    path: str | Path,
    crop_size: int = 256,
    batch: int = 2,
    atol: float = 1e-4,
) -> float:
    """Run the exported graph and eager torch on the same random input and
    return the max abs probability difference (raises if it exceeds `atol`)."""
    try:
        import onnxruntime as ort
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "The parity check needs onnxruntime. Install: pip install 'fakeradar[onnx]'"
        ) from e
    import numpy as np

    x = torch.rand(batch, 3, crop_size, crop_size)
    model.eval()
    with torch.no_grad():
        out = model(x)
        ref = torch.sigmoid(out["logit"] / model.fusion.temperature).cpu().numpy()
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"crop": x.numpy()})[0]
    diff = float(np.abs(got.ravel() - ref.ravel()).max())
    if diff > atol:
        raise RuntimeError(f"ONNX parity check failed: max abs prob diff {diff:.3g} > {atol}")
    return diff
