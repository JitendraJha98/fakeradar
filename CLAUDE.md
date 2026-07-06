# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`fakeradar` is a local-first detector for AI-generated images. It fuses several *forensic branches* (gradient-field, frequency-spectrum, optional frozen CLIP) into one temperature-calibrated P(AI) score. The whole package is `src/fakeradar` and ships a `fakeradar` CLI (Typer).

## Commands

```bash
pip install -e ".[dev]"                    # dev install (CI installs CPU torch first, see below)
pip install -e ".[all]"                    # + CLIP branch, REST API, ONNX, demo

pytest -q                                  # run tests
pytest tests/test_detector.py -q           # one file
pytest tests/test_detector.py::test_name   # one test
ruff check src tests                       # lint (matches CI)
ruff format src tests                      # format (pre-commit runs ruff --fix + ruff-format)

fakeradar scan photo.jpg folder/ --untrained-ok   # smoke-test the pipeline with random weights
fakeradar train configs/train_fast.yaml
fakeradar benchmark val.csv --checkpoint best.pt
fakeradar robustness val.csv --checkpoint best.pt   # re-eval under JPEG/resize/blur
fakeradar export-onnx best.pt fast.onnx
fakeradar serve --checkpoint best.pt                # local REST API, POST /v1/detect
```

CI (`.github/workflows/ci.yml`) installs CPU-only torch from the PyTorch index *before* `pip install -e ".[dev]"`, then runs `ruff check` + `pytest`, on Python 3.10 and 3.11. Do this locally if a fresh env pulls a CUDA torch build you don't want.

There is no published weight file yet (see `ROADMAP.md`); any command that scores images needs either `--checkpoint`, `--pretrained`, or `--untrained-ok` (random weights, meaningless scores — pipeline testing only).

## Architecture

The core abstraction is a **branch** = one forensic view. Data flows: image → branches → fusion head → calibrated logit.

- **`branches/`** — each branch subclasses `DetectionBranch` (`branches/base.py`) and self-registers via `@register_branch("name")` into `BRANCH_REGISTRY`. A branch declares `feature_dim` and `input_kind`:
  - `input_kind="crop"` (gradient, frequency): receives **native-resolution crops, never resized** — resizing destroys the high-frequency traces these branches depend on. This is a load-bearing design invariant, not a detail.
  - `input_kind="resized"` (semantic/CLIP): receives its own resized global view via the branch's `preprocess()`.
- **`fusion.py` `FusionHead`** — projects each branch to a shared embedding, combines them with a **learned softmax gate** (exposed as `gate_weights` for interpretability), emits a main logit plus **auxiliary per-branch logits** (deep supervision + "which branch fired" reporting). Holds a `temperature` buffer for post-hoc calibration.
- **`models.py` `FakeRadarModel`** — assembles enabled branches + fusion into one `nn.Module`; splits inputs by `crop_branch_names` / `resized_branch_names`. Also owns checkpoint I/O (format `"fakeradar.v1"`, loaded `weights_only=True`, `strict=False`), the `MODEL_ZOO` (HF Hub) registry, and ONNX export. **ONNX export only supports crop-only (fast tier) models.**
- **`detector.py` `Detector`** — the high-level API. Loads a tier + weights, runs `predict()` (extracts up to `max_crops` crops, aggregates logits by mean, applies temperature, thresholds into `real`/`uncertain`/`ai`). Returns a `DetectionResult` dataclass. Requires real weights unless `allow_random_init=True`.
- **`cli.py`** — Typer app; **torch imports happen inside each command** so `fakeradar --help` stays instant. Keep that pattern when adding commands.

### Adding a new forensic branch

Subclass `DetectionBranch`, decorate with `@register_branch("yourname")`, set `feature_dim`/`input_kind`, then enable it in a config. Nothing else changes — fusion adapts to whatever branches are enabled. (`docs/architecture.md` lists the other extension points: `PERTURBATIONS`, `MODEL_ZOO`.)

## Configuration — two distinct formats

Do not confuse them:

1. **Detector config** (`configs/fast.yaml`, `configs/accurate.yaml`) → `DetectorConfig` in `config.py`. Describes tier, crop size, thresholds, which branches are enabled, fusion dims. **The built-in presets in `config.py` (`_FAST`/`_ACCURATE`, `get_preset`) are the source of truth** so the installed package works with no external files; the YAMLs mirror them. Keep both in sync when changing a preset.
2. **Training config** (`configs/train_fast.yaml`) → consumed by `training/train.py`. It's a *wrapper*: a top-level `detector:` key that is a **path to a detector YAML**, plus `data`, `optim`, `run` blocks. `train()` loads the detector config from that path.

Tiers: `fast` = gradient + frequency, ~12M params, fully offline, CPU-friendly. `accurate` = adds the frozen CLIP ViT-B semantic branch (needs the `semantic` extra / `timm`, downloads weights once).

## Training

`training/train.py` is single-GPU (or CPU for smoke tests): AMP, cosine LR with warmup, main BCE + weighted auxiliary per-branch BCE, best checkpoint chosen by val AUROC. Datasets come from CSV manifests (`path,label[,generator]`) via `training/datasets.py`; `training/augment.py` is the transport-chain augmenter (JPEG/WebP/resize/blur/noise) applied to every training image — robustness is treated as a training-time property. `scripts/prepare_genimage.py` builds manifests from a GenImage download.

The benchmark protocol is deliberately cross-generator: train on one generator, evaluate on all + perturbed sets. Robustness tables are published alongside every release.

## Conventions

- Ruff line length 100, `ignore = ["E501"]`, target py310. Lint rules: `E, F, I, W, UP, B`.
- Tests add `src/` to `sys.path` via `tests/conftest.py`, so they run without an editable install, but CI does install the package.
- This working copy is **not a git repository** — run `git init` before any git workflow.
