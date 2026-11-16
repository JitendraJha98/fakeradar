# Lens-Law Phase 0-1 (Setup and Gate M) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the lens-law measurement code and run the measurement check (Gate M) that decides whether the thin-lens law can be measured at all.

**Architecture:** A small pure-numpy/torch package, `experiments/lens_law/lenslaw/`, turns an image plus an inverse-depth map into `(inverse depth, log blur)` points and fits them. A job runner embeds that package into private Kaggle script notebooks, which fetch data, run depth on GPU, and write JSONL records. A gate script turns records into the M1-M5 verdicts.

**Tech Stack:** Python 3.10+, numpy, torch (convolutions), Pillow, transformers (Depth Anything V2 Small, Kaggle only), Kaggle CLI 2.2.4, pytest, ruff 0.5.5.

**Spec:** `docs/superpowers/specs/2026-11-14-lens-law-guaranteed-fpr-design.md`

## Global Constraints

- Everything created on Kaggle is private and its slug starts with `fakeradar-`. Nothing else on the account is edited or deleted.
- No token or key in the repo, a notebook, a log, or the chat. Only `~/.kaggle/kaggle.json` is read, by the CLI.
- Experiment code lives in `experiments/lens_law/` and is not imported by `src/fakeradar`.
- Local code uses numpy, torch and Pillow only (scipy, OpenCV and scikit-learn are not installed locally).
- Gate thresholds are copied verbatim from spec section 5 and are not edited after any score is seen.
- The split file is written before any real image is scored: 60% dev, 40% test, by hash of the scene or photographer key.
- Ruff rules from `pyproject.toml` (line length 100, `E,F,I,W,UP,B`). Run tests with `PYTHONUTF8=1 <python> -m pytest experiments/lens_law/tests -q`.
- Images that may not be redistributed are fetched inside private notebooks; the repo stores IDs and URLs only.
- No commits: the owner has not asked for any, and the branch is `main`. Each task ends at a passing test run.
- Kaggle facts confirmed by the smoke test on 2026-11-14: two T4 GPUs (15 GB each), 4 CPUs, 33 GB RAM, 21 GB working disk, internet on, Python 3.13, torch 2.11, transformers 5.20, 30 GPU-hours per week.

## Review Focus

1. An image with no edges (flat sky, blank wall): scoring must return `applicable: False`, not raise.
2. A constant depth map: normalization must not divide by zero; the image is not applicable.
3. Points on one side of the focus plane only: the thin-lens fit must stay finite and may put the focus outside the observed range.
4. Non-RGB or 16-bit inputs (grayscale, alpha, 16-bit PNG, as in the aperture-pair set): loading must convert to 8-bit-range RGB floats.
5. Gradient ratios at or below 1 (noise, ringing after JPEG): they give no blur estimate and must be dropped, never turned into infinite or NaN blur that reaches a fit.

Each line has a test in the task that owns the code: 1 in Tasks 3 and 6, 2 in Tasks 5 and 6, 3 in Task 2, 4 in Task 6, 5 in Task 3.

---

## File structure

```
experiments/lens_law/
  lenslaw/__init__.py
  lenslaw/splits.py        hash-based dev/test assignment
  lenslaw/fit.py           constant, thin-lens and valley fits; score_points
  lenslaw/blur_classic.py  multi-scale gradient-ratio blur estimator (v0)
  lenslaw/patches.py       grid cells -> points; applicability rule
  lenslaw/render.py        lawful and unlawful defocus renders
  lenslaw/transport.py     the three transport conditions
  lenslaw/depth.py         Depth Anything V2 Small wrapper + normalization
  lenslaw/score.py         image + depth -> one record
  lenslaw/gate_m.py        records -> M1..M5
  kaggle_jobs.py           build, push, wait for, and pull private jobs
  jobs/phase1_data.py      Kaggle entry: fetch aperture pairs and real photos
  jobs/phase1_score.py     Kaggle entry: depth + scoring -> JSONL
  tests/conftest.py        adds experiments/lens_law to sys.path
  tests/test_*.py
  splits/                  split manifests (IDs only)
runs/lens_law/             pulled outputs (git-ignored)
```

### Task 1: Splits

**Files:** Create `lenslaw/splits.py`, `tests/test_splits.py`, `tests/conftest.py`, `lenslaw/__init__.py`.

**Interfaces:** Produces `assign_split(group_key: str, salt: str = "lenslaw-v1", dev_fraction: float = 0.6) -> str` returning `"dev"` or `"test"`.

- [ ] Test: same key gives the same split on repeated calls; over 5,000 keys the dev share is within 0.57-0.63; changing `salt` changes some assignments.
- [ ] Implement with SHA-256 of `salt + "/" + group_key`, first 8 bytes as a fraction.
- [ ] Run tests: pass.

### Task 2: Fits and the lens-law score

**Files:** Create `lenslaw/fit.py`, `tests/test_fit.py`.

**Interfaces:** Produces
- `FitResult(resid: float, r2: float, params: dict[str, float])`
- `fit_constant(b, w) -> FitResult`
- `fit_thin_lens(d, b, w, n_grid: int = 61) -> FitResult` with params `c0`, `k`, `d_focus`; model `b = 0.5 * log(c0**2 + k**2 * (d - d_focus)**2)`; `d_focus` searched over `[-0.5, 1.5]`; robust fit (Huber on residuals), reported `resid` is the plain weighted RMS residual
- `fit_valley(d, b, w) -> FitResult` (weighted isotonic regression, non-increasing then non-decreasing, best breakpoint)
- `score_points(d, b, w) -> dict` with keys `n, blur_spread, depth_spread, resid_const, resid_lens, r2_lens, resid_valley, r2_valley, c0, k, d_focus`

`d` is inverse depth normalized to `[0, 1]`, `b` is natural-log blur in pixels, `w` are non-negative weights. `blur_spread` is `exp(P90(b) - P10(b))`.

- [ ] Tests (synthetic points, 400 per case, noise 0.05):
  - lawful V with `d_focus=0.4, k=6, c0=0.8`: recovered `d_focus` within 0.03, `r2_lens >= 0.95`
  - the same data with `d -> 3*d + 2` and with `d -> 1 - d`: `resid_lens` changes by less than 5%
  - step data (blur 0.8 for `d >= 0.4`, 4.0 below): `resid_lens` at least 3x the lawful case, and `resid_valley` below `resid_lens`
  - points only on `d > d_focus`: result is finite (Review Focus 3)
  - valley fit on any exact unimodal sequence: `resid < 1e-9`
- [ ] Implement.
- [ ] Run tests: pass.

### Task 3: Blur estimator v0

**Files:** Create `lenslaw/blur_classic.py`, `tests/test_blur_classic.py`.

**Interfaces:** Produces `BlurMap(sigma: np.ndarray, conf: np.ndarray)` (both `(H, W)` float32; `sigma` is NaN where there is no estimate) and `estimate_blur(gray: np.ndarray, scales: tuple[int, ...] = (1, 2, 4, 8, 16), sigma0: float = 1.0) -> BlurMap`.

Method: at each scale, downsample by the factor with averaging, take gradient magnitude before and after a Gaussian re-blur of `sigma0`, keep edge pixels (local gradient maxima above a noise-relative threshold), and convert the ratio `R` to `sigma = sigma0 / sqrt(R**2 - 1)` times the scale factor. Ratios outside `(1.02, 6)` are dropped. Per pixel, the finest scale whose estimate lies in its reliable band (0.4 to 3 px at that scale) wins.

- [ ] Tests: a vertical step edge blurred with sigma in `{1, 2, 4, 8}` gives a median estimate within 30% of the truth; a flat image gives no finite estimates (Review Focus 1); an image of pure noise produces no infinite values (Review Focus 5).
- [ ] Implement.
- [ ] Run tests: pass.

### Task 4: Points and applicability

**Files:** Create `lenslaw/patches.py`, `tests/test_patches.py`.

**Interfaces:** Consumes `BlurMap`. Produces `Points(d, b, w, yx)` and
- `collect_points(blur: BlurMap, inv_depth: np.ndarray, cell: int = 32, min_edge_px: int = 12, max_depth_spread: float = 0.10) -> Points`
- `MIN_CELLS = 60`, `MIN_BLUR_SPREAD = 2.0`, `applicable(points: Points) -> bool`

A cell contributes one point: median log blur of its edge pixels, median inverse depth, weight `min(1, n_edge / 50)`. Cells whose inverse-depth range (P90 - P10) exceeds `max_depth_spread` are dropped as depth edges.

- [ ] Tests: a cell straddling a depth step is excluded; a cell with fewer than `min_edge_px` estimates is excluded; `applicable` is False with 59 cells, False with blur spread 1.9, True with 60 cells and spread 2.1; empty input returns empty `Points` and `applicable` False.
- [ ] Implement.
- [ ] Run tests: pass.

### Task 5: Renders

**Files:** Create `lenslaw/render.py`, `tests/test_render.py`.

**Interfaces:** Produces `MODES = ("lawful", "step", "position", "missing_foreground")`,
`blur_field(inv_depth: np.ndarray, mode: str, d_focus: float, sigma_max: float) -> np.ndarray` and
`render_defocus(rgb: np.ndarray, sigma_field: np.ndarray, n_levels: int = 12) -> np.ndarray` (sRGB floats in `[0, 1]`, blur applied in linear light, per-pixel interpolation between Gaussian-blurred layers).

Field definitions, with `u = |inv_depth - d_focus|` scaled so its 95th percentile is 1: lawful `sigma_max * u`; step `sigma_max` where `inv_depth < d_focus - 0.05`, else 0; position `sigma_max * |row/H - 0.5| * 2`; missing_foreground equals lawful where `inv_depth <= d_focus`, else 0.

- [ ] Tests: lawful field is zero at `d_focus` and linear in `|d - d_focus|`; step field takes only two values; position field is identical for two different depth maps; a constant depth map does not divide by zero (Review Focus 2); rendering with an all-zero field returns the input within 1e-3; mean intensity changes by less than 2% after rendering.
- [ ] Implement.
- [ ] Run tests: pass.

### Task 6: Scoring pipeline and the local end-to-end check

**Files:** Create `lenslaw/score.py`, `lenslaw/transport.py`, `lenslaw/depth.py`, `tests/test_score.py`, `tests/test_transport.py`.

**Interfaces:** Consumes Tasks 2-5. Produces
- `WORK_LONG_SIDE = 1536`, `load_rgb(path_or_pil) -> np.ndarray` (any mode or bit depth to float RGB in `[0, 1]`, long side capped)
- `score_arrays(rgb: np.ndarray, inv_depth: np.ndarray) -> dict` (the `score_points` keys plus `applicable`)
- `CONDITIONS = ("clean", "s075_q85", "s050_q70")`, `apply_condition(img: PIL.Image.Image, name: str) -> PIL.Image.Image`
- `MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"`, `normalize_inverse_depth(raw) -> np.ndarray` (P2/P98 to `[0, 1]`, zeros for a constant map), `load_depth(device)`, `inverse_depth(model, img) -> np.ndarray`; torch and transformers are imported inside the functions

- [ ] Tests:
  - end to end on a synthetic scene (random rectangles texture, depth ramp plus a box, oracle depth): over 10 seeds at `sigma_max=4`, the lawful render has lower `resid_lens` than each unlawful mode in at least 8 seeds
  - a flat image returns `applicable: False` without raising (Review Focus 1)
  - `load_rgb` accepts L, RGBA and 16-bit PNG (Review Focus 4)
  - `apply_condition("s050_q70")` halves both sides (to the nearest pixel)
  - `normalize_inverse_depth` of a constant array is all zeros (Review Focus 2)
- [ ] Implement.
- [ ] Run the whole suite and `ruff check experiments`: pass.

### Task 7: Kaggle job runner

**Files:** Create `kaggle_jobs.py`, `tests/test_kaggle_jobs.py`.

**Interfaces:** Produces
- `build_job(name: str, entry: Path, out_dir: Path, *, gpu: bool, internet: bool = True, dataset_sources: tuple[str, ...] = (), kernel_sources: tuple[str, ...] = ()) -> Path` writing `main.py` (bootstrap that unpacks an embedded base64 zip of `lenslaw/` and then runs the entry script) and `kernel-metadata.json`
- `push(job_dir: Path, timeout_s: int) -> str`, `wait(kernel: str, max_s: int) -> str`, `pull_output(kernel: str, dest: Path) -> None`

Metadata always has `is_private: "true"`, title `fakeradar <name>`, id `<username>/fakeradar-<name>`, and `machine_shape: "NvidiaTeslaT4"` when `gpu`. The username is read from `kaggle config view`, never from the key file. This replaces the spec's "code as a private dataset" with code embedded in the private notebook: one push is atomic and privacy is the same.

- [ ] Tests (no network): metadata is private and prefixed; a name with characters outside `[a-z0-9-]` is rejected; the embedded bundle round-trips to identical file bytes; the bootstrap never contains the string `kaggle.json`.
- [ ] Implement.
- [ ] Run tests: pass.

### Task 8: Phase 1 data on Kaggle

**Files:** Create `jobs/phase1_data.py`, `splits/phase1_pairs.csv`, `splits/phase1_real_wide.csv`.

- [ ] Fetch aperture pairs inside a private notebook, trying in order: the DPDD authors' link; the DPDD copy used by public deblurring repositories; the iDFD set already on Kaggle; EBB!. Record which source was used, its stated terms, the image count, and a pixel-alignment check (correlation of downsampled wide and narrow shots) in `data_report.json`.
- [ ] Fetch the Unsplash Lite metadata, select photos with `exif_aperture_value <= 2.8`, and download up to 1,500 at width 1536 from `photo_image_url`.
- [ ] Write the two split files with `assign_split` (scene ID for pairs, `photographer_username` for photos) and pull them into `splits/` before any scoring job runs.
- [ ] Verify: `data_report.json` lists at least 60 usable pairs and at least 600 real photos; otherwise move to the next source.

### Task 9: Phase 1 scoring on Kaggle

**Files:** Create `jobs/phase1_score.py`.

- [ ] For each pair: score the wide shot with depth from the narrow shot and with depth from itself; score the narrow shot.
- [ ] For each narrow (all-in-focus) shot: render the four modes at `sigma_max=4` with a focus quantile drawn from 0.3-0.7, re-estimate depth from each render, and score in each of the three conditions.
- [ ] For each real wide-aperture photo: score in each of the three conditions.
- [ ] Write one JSONL record per (image, variant, condition) with `id, source, split, variant, condition` plus the score keys; skip records that already exist so a re-run resumes.
- [ ] Verify: record counts match the manifests, and no record has NaN in `resid_lens` when `applicable` is true.

### Task 10: Gate M

**Files:** Create `lenslaw/gate_m.py`, `tests/test_gate_m.py`.

**Interfaces:** Produces `THRESHOLDS` (verbatim from the spec: M1 0.80 and 0.70 at `r2_lens >= 0.5` among pairs with blur spread of at least 2; M2 0.90; M3 0.85; M4 0.50 and 0.10 at the 5% threshold; M5 0.10 drop and 0.35), `auroc(pos, neg) -> float`, and `gate_m(records: list[dict], split: str) -> dict` with one entry per check holding `value` and `passed`, plus overall `passed`.

- [ ] Tests: hand-built record sets where each check is just above and just below its threshold flip only that check; `auroc` matches a brute-force pair count on small arrays, with ties counted as half.
- [ ] Implement.
- [ ] Run on dev records; freeze any tuned setting; run once on test records; write `runs/lens_law/gate_m/report.json` and a one-page memo with the V-plots of five real pairs.
- [ ] Apply the stop rule from the spec: pass leads to the Phase 2 plan; fail leads to one of the two allowed fix rounds (blur estimator v1, then a second depth model).
