# Engineering-complete v0.1 — design

Date: 2026-07-07 · Status: approved (option B) · Scope: make every mechanism the
paper/README promise actually work and be verified; no invented benchmark numbers.

## Context

fakeradar v0.1 code is architecturally complete but several promised mechanisms
are unreachable, broken, or untested. Training real weights (GPU + GenImage) is
explicitly out of scope for this change; paper tables stay `[TO FILL]` per
`paper/INTEGRITY_CHECKLIST.md`.

## Changes

### 1. Calibration becomes reachable (paper §3.4 promise)
- New `src/fakeradar/evaluation/calibrate.py`: `collect_logits(model, manifest, ...)`
  and `calibrate_checkpoint(checkpoint, manifest, out=None, ...) -> dict` — fits
  `FusionHead.fit_temperature` on held-out logits, reports ECE before/after,
  saves the checkpoint with the fitted temperature.
- New CLI command `fakeradar calibrate manifest.csv --checkpoint best.pt [--out p]`.
- `training/train.py` calibrates `best.pt` on the val manifest after training
  (config `run.calibrate`, default true).
- `FusionHead.fit_temperature`: fix device bug (log_t created on logits.device),
  clamp fitted T to [0.05, 20] for numerical sanity.

### 2. `max_crops: 8` becomes real (paper §3.4 K≤8)
- `preprocessing.crop_positions`: 3×3 grid — center first, then 4 corners, then
  4 edge midpoints; deduplicated; capped at `max_crops`. Small images (< crop
  size) remain the only upscale case.
- Paper §3.4 wording updated: "averaged over K ≤ 9 native crops (center,
  corners, edge midpoints)"; docs/architecture.md same.

### 3. ONNX export actually works (README §deployment promise)
- Empirical finding: export fails on `aten.bincount` (cold radial-index cache);
  with warm cache, torch ≥2.9 dynamo exporter handles fft2, parity 0.0.
- `models.export_onnx`: torch>=2.9 guard with clear error; warmup forward at
  export size; `dynamo=True` + `dynamic_shapes`; opset default 18.
- New `models.verify_onnx_parity(model, path, ...)` (lazy onnxruntime import);
  `export-onnx` CLI runs it when ORT is installed and prints max |Δprob|.
- CLI: `sys.stdout/stderr.reconfigure(errors="replace")` in export command —
  torch's emoji progress logging otherwise crashes cp1252 Windows consoles.
- `onnxscript` added to the `onnx` extra (required by the dynamo exporter);
  CI installs `.[dev,onnx]` so the parity test runs.

### 4. LGrad no longer corrupts state (latent, `use_lgrad=True` only)
- Frozen ResNet stored via `object.__setattr__` (not a submodule): stays out of
  `state_dict()`/`parameters()`, immune to `model.train()`; moved lazily to the
  input device inside `_lgrad_map`.

### 5. Training modernization
- `torch.amp.GradScaler("cuda", ...)` / autocast (deprecation-proof).
- Val metrics via `evaluation.metrics.summarize` → adds TPR@5%FPR and ECE to
  the log and metrics.csv (paper §4 metric set).
- metrics.csv logs epoch-mean train loss, not last-batch loss.

### 6. Server hardening
- Invalid uploads → HTTP 400 (not 500); version from `fakeradar.__version__`.

### 7. Tests (new files; all CPU, no downloads; optional deps skip cleanly)
- test_config.py — **configs/*.yaml ≡ code presets** (CLAUDE.md invariant),
  round-trips, unknown tier.
- test_fusion.py — gate softmax sums to 1, temperature effect, fit_temperature
  recovers a known miscalibration, device/clamp behavior.
- test_preprocessing.py — grid positions unique/ordered/capped; big images
  never resampled (pixel-identity vs PIL crop); small images upscaled.
- test_augment.py, test_datasets.py — shapes, labels, manifest edge cases.
- test_cli.py — Typer runner: version, scan --untrained-ok (table + --json).
- test_train_smoke.py — end-to-end 1-epoch CPU train on 8 synthetic images with
  a tiny detector config; asserts best.pt/metrics.csv; calibration runs.
- test_export_onnx.py — export + ORT parity < 1e-4 (skips without onnx extras).
- test_server.py — healthz, detect, invalid-image 400 (skips without fastapi).

### 8. Hygiene
- Delete duplicate root `paper.tex` + `GRAFT_paper_draft.pdf` (identical copies
  live in `paper/`).
- `py.typed` marker; CHANGELOG under [Unreleased]; README + docs/architecture.md
  + CLAUDE.md synced with the above (CI matrix wording, calibrate command).

## Error handling
- `Detector` init without weights → existing clear ValueError; CLI catches and
  prints a clean message instead of a traceback.
- `export_onnx` on old torch → informative RuntimeError naming the version.
- `calibrate` with a manifest containing a single class → explicit error.

## Testing/verification
Full pytest + ruff check/format (pinned 0.5.5, matching CI) + end-to-end CLI
runs (scan/export-onnx/calibrate on synthetic data) before push to origin/main.

## Out of scope
Training real weights, benchmark tables, HF Hub upload, PyPI release
(ROADMAP v0.1 items requiring GPU + GenImage).
