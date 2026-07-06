# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
