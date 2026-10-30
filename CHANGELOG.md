# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `fakeradar prepare-genimage` (logic now in `fakeradar.training.genimage`, so
  pip installs have it; `scripts/prepare_genimage.py` is a thin wrapper):
  handles the released `<generator>/<subset>/<split>/` layout, writes
  train / val / test manifests whose default names match
  `configs/train_fast.yaml`, `--test-generators all`, absolute paths.
- `fakeradar train --resume runs/<run>/last.pt` — `last.pt` now carries
  optimizer, scheduler and GradScaler state; metrics.csv is appended.
- Training config: `run.device`; `detector:` accepts a preset name
  (`fast`/`accurate`), an inline mapping, or a path relative to the CWD or
  to the training config; new `configs/train_accurate.yaml`.
- Benchmark: TPR@5%FPR, ECE and abstention-rate columns (the paper metric
  set; `metrics.abstention_rate`), `MEAN` over-generator AUROC row,
  `--scores-csv` raw per-image scores for benchmark and robustness;
  `--pretrained`/`--device` on benchmark, robustness, scan and serve.
- REST API: `--pretrained`, `--device`, `--max-upload-mb` (413 above it and
  on decompression bombs); inference runs in the threadpool so it no longer
  blocks the event loop.
- `scan` reports unreadable files as error rows and keeps going (exit code 1
  if any failed). Demo: `--pretrained` and `FAKERADAR_CHECKPOINT` /
  `FAKERADAR_PRETRAINED` env vars (HF Space), shows both class probabilities.
- Checkpoints record `fakeradar_version`; loading checks the format and the
  calibration temperature and rejects non-fakeradar files with a clear error.
- Detector config validation: unknown keys, `max_crops` outside [1, 9],
  `crop_size` < 32, and inverted/out-of-range thresholds raise.
- `fakeradar calibrate` command: fit the output temperature on held-out data
  (ECE reported before/after; `--num-workers`, `--device`); training now runs
  it automatically on the best checkpoint (`run.calibrate`, default true) and
  checkpoints store the stats.
- 3×3 crop grid (center, corners, edge midpoints): `max_crops` up to 9 now
  takes effect — the old grid silently capped at 5 positions.
- ONNX parity verification: `export-onnx` compares the exported graph against
  eager PyTorch when onnxruntime is installed; `verify_onnx_parity` API.
- Validation metrics now include TPR@5%FPR and ECE (paper protocol);
  `metrics.csv` gains those columns and logs epoch-mean train loss.
- `py.typed` marker; CI matrix extended to Python 3.13/3.14 and installs the
  onnx extra so the export parity test runs.

### Fixed
- **Temperature fit could store T=NaN** on a perfectly separable held-out set
  (unbounded LBFGS diverging toward T -> 0), making every later prediction NaN
  while the verdict silently read `uncertain`. Now a deterministic bounded
  grid search (NLL is convex in 1/T), with a warning when the optimum sits on
  a bound. Metrics refuse NaN scores instead of producing garbage tables, and
  the ONNX parity check fails on NaN instead of reporting "OK".
- **AMP train/inference skew:** the gradient branch's forensic stack ran in
  fp16 under CUDA autocast, where the coherence channel drifts by O(1) vs the
  fp32 used at inference. It now always runs in fp32.
- **Calibration protocol mismatch:** the temperature was fit on single
  center-crop logits but applied to the mean logit over up to 9 crops; it is
  now fit on exactly the quantity `Detector.predict` rescales.
- **Benchmark leakage:** the GenImage `val` split was used both for model
  selection/calibration and as the reported test set; selection now uses a
  hold-out carved from the training split.
- `prepare_genimage.py` wrote manifest names `train_fast.yaml` did not read
  and missed GenImage's nested subset directories.
- `calibrate` dropped the epoch/val metadata stored in the checkpoint.
- Training: fails fast on single-class train/val manifests or a train set
  smaller than one batch (previously: silent no-op / no best.pt); skips
  (and after 10 in a row, aborts on) non-finite losses; one unreadable
  training image is skipped with a warning instead of killing the run;
  `cuda:N` devices work with autocast; DataLoader workers no longer receive
  a pickled copy of the CLIP backbone on Windows/macOS (picklable
  `preprocessor()`), and are persistent across epochs.
- Manifests: labels other than 0/1 and missing columns raise with the
  offending line; an empty `generator` cell becomes `unknown`.
- Robustness decodes each image once instead of once per perturbation.
- Minimum torch is now 2.3 (`torch.amp.GradScaler`).
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
