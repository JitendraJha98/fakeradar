# Guaranteed False-Accusation Rate (Phase 3, up to Gate B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn any detector score into a decision whose rate of flagging real photos is bounded, and audit that bound on phones and sharing platforms never used for calibration.

**Architecture:** A small pure-numpy package, `experiments/fpr_guarantee/fprcal/`, holds the calibration maths (split-conformal thresholds per transport stratum, pooled fallback) and the Gate B audit. Private Kaggle jobs score real photos with a public reference detector and write one JSONL record per image; no image leaves the notebook.

**Tech Stack:** Python 3.10+, numpy, Pillow, torch (Kaggle only), Kaggle CLI 2.2.4 through `experiments/lens_law/kaggle_jobs.py`, pytest, ruff 0.5.5.

**Spec:** `docs/superpowers/specs/2026-11-14-lens-law-guaranteed-fpr-design.md` (Phase 3). The lens-law score is not wrapped: its gate failed (`experiments/lens_law/GATE_M_REPORT.md`).

## Global Constraints

- Everything on Kaggle is private and prefixed `fakeradar-`; nothing else on the account is touched.
- No token or key in the repo, a notebook, a log, or the chat.
- FODB terms: research use only, no redistribution, cite Hadwiger and Riess (2021). Jobs output scores only, never FODB images.
- Reference detector: Community Forensics `OwensLab/commfor-model-384` (weights and code MIT).
- Levels audited: 5% and 1%. The pass rule below is fixed before any score exists and is not edited afterwards.
- Phones are split 60% calibration / 40% audit by `assign_split(device_id, salt="fodb-phones")`. One platform, Telegram, is never used for calibration.
- No commits unless the owner asks. Ruff rules from `pyproject.toml`.

## Gate B pass rule (fixed now)

Strata: the camera original plus each shared platform seen in calibration (Facebook, Instagram, Twitter, WhatsApp), five in all. Checks: 5 strata x 2 levels = 10.

- A check **passes** if the measured rate on audit phones is at most 1.5 x the level, or if the lower end of its 95% interval is at or below the level. The interval is a bootstrap over audit phones and calibration phones together (2,000 resamples), so it carries both the audit sample's and the calibration set's randomness and respects that photos from one phone are not independent.
- **Gate B passes** if at most one of the ten checks fails and no measured rate exceeds twice its level.
- The held-out platform (Telegram, thresholded with the pooled calibration) is reported beside the gate and is not part of the pass rule. So are the same audits for a fixed 0.5 threshold and for temperature scaling.
- Expected sample per stratum: about 16 calibration phones x 143 photos and 11 audit phones x 143 photos. At the 1% level that is about 16 expected flags per stratum on audit phones.

## Review Focus

1. A stratum with too few calibration photos for the level (fewer than `ceil(1/alpha) - 1`): the threshold must fall back to the pooled calibration set, and say so, never return a threshold that cannot give the level.
2. Ties in scores (a detector that outputs many identical values): the threshold must not flag more than the level allows on the calibration set itself.
3. A stratum label never seen in calibration: it must use the pooled threshold, not raise.
4. NaN or infinite scores: rejected with an error naming the count, never silently dropped.
5. A phone that appears in both calibration and audit records: rejected, since it would void the audit.

Each line has a test in Task 1 or Task 2.

---

### Task 1: Conformal thresholds

**Files:** Create `experiments/fpr_guarantee/fprcal/__init__.py`, `fprcal/conformal.py`, `tests/conftest.py`, `tests/test_conformal.py`.

**Interfaces:** Produces
- `conformal_threshold(scores, alpha: float) -> float`: the `ceil((n + 1) * (1 - alpha))`-th smallest calibration score; `inf` if that rank exceeds `n`. A test image is flagged when its score is strictly greater.
- `Calibration(thresholds: dict[str, float], pooled: float, alpha: float, fallback: set[str])`
- `calibrate(scores, strata, alpha: float, min_count: int | None = None) -> Calibration`: one threshold per stratum with at least `min_count` photos (default `ceil(1/alpha) - 1`), pooled otherwise.
- `flag(cal: Calibration, scores, strata) -> np.ndarray` of bool.

- [ ] Tests: on 20,000 exchangeable draws split in two, the flagged share at 5% and 1% is within 0.3 points of the level; with `n = 50` and `alpha = 0.01` the per-stratum threshold is not used and the stratum is listed in `fallback`; an unseen stratum uses `pooled`; all-identical scores flag nothing; NaN scores raise `ValueError` naming the count.
- [ ] Implement.
- [ ] Run tests: pass.

### Task 2: Gate B audit

**Files:** Create `fprcal/gate_b.py`, `tests/test_gate_b.py`.

**Interfaces:** Consumes Task 1. Produces `GATE_B = {"levels": (0.05, 0.01), "ratio": 1.5, "max_failed": 1, "hard_ratio": 2.0, "bootstrap": 2000, "held_out": "telegram"}` and `gate_b(records: list[dict], seed: int = 0) -> dict` where a record has `phone`, `role` ("cal" or "audit"), `platform`, `score`. The result holds, per stratum and level, the measured rate, its interval and `passed`; the held-out platform's rates; and overall `passed`.

- [ ] Tests: synthetic exchangeable records pass; shifting one audit stratum's scores upward so its rate is 3x the level fails that check and the gate; a phone listed under both roles raises `ValueError`; the held-out platform never contributes a threshold; results are identical for the same seed.
- [ ] Implement.
- [ ] Run tests: pass.

### Task 3: Confirm the dependencies on Kaggle

**Files:** `experiments/lens_law/jobs/phase3_smoke.py` (written).

- [ ] Run it as a private CPU job. It must show the reference detector loads and scores, and report the FODB archive's folder layout (phones, platform versions, counts, image sizes).
- [ ] Record the layout in the ledger. If the detector does not load, try the 224-input model; if FODB does not download, stop and tell the owner.

### Task 4: Score FODB with the reference detector

**Files:** Create `experiments/fpr_guarantee/jobs/score_fodb.py`.

- [ ] Private GPU job: download the three FODB archives to temporary disk, score every image with the reference detector at its own preprocessing, and write `records.jsonl` with `phone`, `scene`, `platform`, `role`, `score` (the logit), `width`, `height`, `jpeg_quality` when readable. Resume by skipping records already written. Delete the archives at the end.
- [ ] Verify: about 23,106 records, 27 phones, six platform versions, no NaN scores.

### Task 5: Gate B

- [ ] Run `gate_b` on the records once. Write `experiments/fpr_guarantee/GATE_B_REPORT.md` with the ten checks, the held-out platform, and the fixed-threshold and temperature-scaling audits for comparison.
- [ ] If the gate fails, report the size of each miss and add the conservative variant (calibrate at a tighter level) as a separate, clearly labelled result. Do not edit the pass rule.

## After Gate B (outline only)

Detection rate at the guaranteed level needs generated images, which Phase 2 never produced. Decision for that step: generate a few hundred images with open-weight models on Kaggle from Unsplash photo descriptions, pass them through the simulated transport conditions, and report how many are flagged at the calibrated thresholds. The limits are stated with the result: simulated rather than real platform transport for generated images, and open-weight generators only.
