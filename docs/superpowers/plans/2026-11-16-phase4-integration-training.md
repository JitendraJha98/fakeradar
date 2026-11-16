# Phase 4: Train and Benchmark the Existing Model, Integrate the Guarantee

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give fakeradar real, honestly measured numbers: train the existing fast tier with the GenImage JPEG shortcut closed, benchmark it, and ship the audited calibration as part of the package.

**Architecture:** Small, test-first changes inside `src/fakeradar` (one new preprocessing step, options threaded through the dataset, the benchmark harness and the CLI), then private Kaggle GPU jobs that embed the package, attach the Kaggle-hosted GenImage copies, train, and write raw score CSVs.

**Tech Stack:** existing fakeradar stack (PyTorch, Typer, Pillow), pytest, ruff 0.5.5, Kaggle CLI through `experiments/lens_law/kaggle_jobs.py`.

**Spec:** `docs/superpowers/specs/2026-11-14-lens-law-guaranteed-fpr-design.md` (Phase 4).

## Global Constraints

- CLAUDE.md invariants hold: forensic branches get native-resolution crops and are never resized; torch imports stay inside CLI commands; presets in `config.py` stay the source of truth.
- Default behaviour of existing commands does not change; every new behaviour is opt-in.
- Every published number comes from the harness with its raw score CSV (`paper/INTEGRITY_CHECKLIST.md`). Nothing is invented or copied from other papers.
- GenImage's `val` split is the test set; model selection and calibration use the hold-out carved from training.
- Kaggle items private, prefixed `fakeradar-`; no commits unless the owner asks.
- Kaggle T4s are several times slower than the A100 the configs assume: sessions of 8 hours or less, resumable from `last.pt`.

## Review Focus

1. A manifest mixing JPEG reals and PNG fakes: after equalization both classes must carry one JPEG pass at the reference quality, and JPEG sources must not be compressed a second time.
2. A file whose extension lies about its format (a JPEG named `.png`): the format is read from the file header, not the name.
3. Equalization switched off: outputs are byte-identical to today's behaviour.
4. Images smaller than the crop size: still upscaled exactly as before.
5. A resumed run whose config changed the equalization setting: rejected, like any other config change.

---

### Task 1: Equalize compression history

**Files:** Modify `src/fakeradar/preprocessing.py`; test `tests/test_preprocessing.py`.

**Interfaces:** Produces `load_image_equalized(source, jpeg_quality: int | None) -> PIL.Image.Image`: like `load_image`, but when `jpeg_quality` is set and the file is not already a JPEG (format read from the header), the image is JPEG-encoded once at that quality and decoded. `None` returns exactly what `load_image` returns.

- [ ] Tests: a PNG source comes back changed and equal to a manual JPEG round trip at the same quality; a JPEG source comes back pixel-identical to `load_image`; a JPEG saved with a `.png` name is treated as JPEG; `jpeg_quality=None` equals `load_image` for both.
- [ ] Implement. Run tests.

### Task 2: Thread it through training

**Files:** Modify `src/fakeradar/training/datasets.py`, `src/fakeradar/training/train.py`, `configs/train_fast.yaml`, `configs/train_accurate.yaml`; tests `tests/test_datasets.py`, `tests/test_train_smoke.py`.

**Interfaces:** `ManifestDataset(..., equalize_jpeg_quality: int | None = None)` uses `load_image_equalized` for every image, train and eval. Training config key `data.equalize_jpeg_quality` (absent = off) is passed to the train and val datasets and to the post-training calibration, and is stored in the checkpoint extras so `--resume` rejects a changed value.

- [ ] Tests: with the option, a PNG row's crop differs from the option-off crop and a JPEG row's does not; the smoke training run accepts the key; resuming with a different value raises.
- [ ] Implement. Run the whole suite.

### Task 3: Benchmark under the same rule

**Files:** Modify `src/fakeradar/evaluation/benchmark.py`, `src/fakeradar/evaluation/calibrate.py`, `src/fakeradar/cli.py`; tests `tests/test_benchmark.py`, `tests/test_cli.py`.

**Interfaces:** `run_benchmark(..., equalize_jpeg_quality=None)`, `run_robustness(..., equalize_jpeg_quality=None)`, `calibrate_checkpoint(..., equalize_jpeg_quality=None)`; CLI option `--equalize-jpeg INTEGER` on `benchmark`, `robustness` and `calibrate`. The markdown output states which setting was used.

- [ ] Tests: scores for PNG rows change with the option and JPEG rows do not; the output names the setting; the CLI passes it through.
- [ ] Implement. Run the whole suite and `ruff check src tests`.

### Task 4: Train and benchmark on Kaggle

**Files:** Modify `experiments/lens_law/kaggle_jobs.py` (bundle a chosen package directory); create `experiments/phase4/jobs/train_fast.py`.

- [ ] Check the licence fields of `vtphatt2/genimage-stable-diffusion-v1-4` and `vtphatt2/genimage-validation`; record them.
- [ ] Smoke job: build manifests from the attached datasets, train a few hundred steps, confirm speed and that `last.pt` resumes.
- [ ] Size the run from the measured speed (subset of the training images if needed) and train with `equalize_jpeg_quality: 96`; pull `best.pt`, `metrics.csv`.
- [ ] Benchmark and robustness on the GenImage `val` split, raw and equalized, with `--scores-csv`; pull the tables and CSVs.

### Task 5: Calibration in the package (outline)

Move `fprcal.conformal` into `src/fakeradar/evaluation/` with its tests and add `fakeradar calibrate-fpr REAL_MANIFEST --checkpoint ... --alpha 0.01`, which stores per-stratum thresholds in the checkpoint extras; `scan` then reports the verdict at that level. Detailed after Task 4's numbers exist.

### Task 6: Paper and docs (outline)

Fill tables only from Task 4's CSVs; fix the gate claim (one global parameter) and run the coherence ablation before any claim about it; add the Gate M and Gate B reports as the paper's two new results.
