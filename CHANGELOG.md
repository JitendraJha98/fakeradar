# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `fakeradar calibrate` command: fit the output temperature on held-out data
  (ECE reported before/after); training now runs it automatically on the best
  checkpoint (`run.calibrate`, default true) and checkpoints store the stats.
- 3×3 crop grid (center, corners, edge midpoints): `max_crops` up to 9 now
  takes effect — the old grid silently capped at 5 positions.
- ONNX parity verification: `export-onnx` compares the exported graph against
  eager PyTorch when onnxruntime is installed; `verify_onnx_parity` API.
- Validation metrics now include TPR@5%FPR and ECE (paper protocol);
  `metrics.csv` gains those columns and logs epoch-mean train loss.
- `py.typed` marker; CI matrix extended to Python 3.13/3.14 and installs the
  onnx extra so the export parity test runs.

### Fixed
- ONNX export of the fast tier actually works: warmup forward bakes the
  frequency branch's `bincount` as a constant, export uses the dynamo exporter
  (torch>=2.9, clear error otherwise) with dynamic batch.
- REST API: invalid uploads return HTTP 400 instead of 500; version is read
  from `fakeradar.__version__`; Python 3.14 compatibility (annotation
  resolution of `UploadFile`).
- `FusionHead.fit_temperature`: device-safe (GPU logits no longer crash) and
  the fitted temperature is clamped to [0.05, 20].
- LGrad channel (`use_lgrad=true`): the frozen backbone no longer bloats
  checkpoints (~45MB) and `model.train()` no longer flips its BatchNorm into
  training mode.
- Training uses the non-deprecated `torch.amp` GradScaler API.
- CLI: missing weights produce a clean error message instead of a traceback;
  `export-onnx` no longer dies on cp1252 Windows consoles.

### Removed
- Duplicate root-level `paper.tex` and `GRAFT_paper_draft.pdf` (canonical
  copies live in `paper/`).

## [0.1.0] - 2026-07-07

### Added
- Initial public release.
- Multi-branch forensic detector: gradient-field + frequency branches, with an
  optional frozen CLIP semantic branch, fused into one temperature-calibrated
  P(AI) score.
- `fakeradar` CLI: `scan`, `train`, `benchmark`, `robustness`, `export-onnx`,
  and `serve` (local REST API).
- `fast` (fully offline, CPU-friendly, ~12M params) and `accurate` (adds frozen
  CLIP ViT-B) tiers.
- Training pipeline with a transport-chain augmenter (JPEG/WebP/resize/blur/
  noise) so robustness is a training-time property; best checkpoint by val AUROC.
- Native-resolution crop scoring, ONNX export for the fast tier, and a Python
  `Detector` API.

[Unreleased]: https://github.com/JitendraJha98/fakeradar/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/JitendraJha98/fakeradar/releases/tag/v0.1.0
