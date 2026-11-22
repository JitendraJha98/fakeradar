---
name: fakeradar-handoff
description: "THE single handoff for fakeradar (as of 2026-11-23): goal, every result with numbers, where all files live, how to run Kaggle/HF/tests on this laptop, what is pending, user decisions and standing rules"
metadata:
  node_type: memory
  type: project
  originSessionId: 9289c89e-05a0-447f-a184-53b554b5b023
  modified: 2026-11-22T22:00:13.715Z
---

One file to resume this project with any model or session. The user asked
for it on 2026-11-23 ("give me one final memory file"). Detailed decisions and
every raw number are in the ACTIVE ledger
`.superpowers/sdd/2026-11-16-phase4-integration-training/progress.md`
(git-ignored): read its latest `NEXT:` block first. Closed ledgers:
`.superpowers/sdd/2026-11-14-lens-law-phase0-1/progress.md`,
`.superpowers/sdd/2026-11-15-guaranteed-fpr-phase3/progress.md`.

## 1. Goal and what it became

`fakeradar` (`src/fakeradar`, CLI `fakeradar`) is a local-first AI-image
detector: gradient-field + frequency branches, optional frozen CLIP branch,
gated fusion, temperature calibration. The user wanted a genuinely new result
to present to researchers. A literature check (2026-11-14) showed the
original "GRAFT" novelty claims had prior art: hybrid low-level + CLIP fusion
(AIDE ICLR 2025, Co-SPY CVPR 2025), structure-tensor coherence (public GitHub
code), GenImage's JPEG-vs-PNG shortcut (Grommelt et al. 2024). Agreed plan
(spec `docs/superpowers/specs/2026-11-14-lens-law-guaranteed-fpr-design.md`):
A = lens-law test, B = guaranteed false-accusation rate, then train and
benchmark the existing model honestly.

## 2. Results (all measured; never invent or retype numbers)

**A. Lens-law test — STOPPED, negative.** Gate M (5 checks fixed in advance)
failed on the dev split with both blur estimators (edge-ratio v0, learned
0.92M-param v1) and both allowed fix rounds. Real wide-aperture photos fit
the thin-lens V worse (median residual 0.40) than deliberately broken renders
(0.27-0.39). On DPDD (310 dev scenes) 59-63% of real pairs fit the V (needed
80%/70%), M2 passes 96-99%: the law is measurable in most strongly defocused
photos but not reliably enough per image to be evidence. Test split never
scored. Report: `experiments/lens_law/GATE_M_REPORT.md`; code
`experiments/lens_law/` (lenslaw package + tests).

**B. Guarantee — Gate B PASSED (10/10).** Split-conformal thresholds from
real photos only, per sharing platform. Detector: Community Forensics
`OwensLab/commfor-model-384` (MIT). Data: FODB, 23,105 photos, 27 phones,
camera original + Facebook/Instagram/Twitter/WhatsApp/Telegram; 16
calibration / 11 audit phones; Telegram held out. Audit-phone rates 4.7-5.9%
at a 5% target, 0.8-1.3% at 1%; Telegram 3.6% / 0.9%. Pooled (unstratified)
calibration over-flags camera originals (10.0% at 5%). Detection rate on 200
SDXL images after 0.5x+JPEG70: 82% at the 0.5 cut vs 90-94% at calibrated 1%.
Report: `experiments/fpr_guarantee/GATE_B_REPORT.md` (+ .json); code
`experiments/fpr_guarantee/`; now also in the package as
`fakeradar calibrate-fpr REAL.csv --alpha 0.01` and
`fakeradar scan --fpr-thresholds FILE [--stratum NAME]`
(`src/fakeradar/evaluation/guarantee.py`).

**C. Benchmark (GenImage, train SD v1.4 only, 95k images, 5 epochs, T4,
compression equalized at JPEG 96).** Fast tier, 3 seeds: mean AUROC over 8
generators 0.9367 ± 0.0014 (as stored 0.9024); SD family ~0.999, ADM 0.859;
JPEG 30 0.776. Coherence ablation (2 seeds): no clean effect; removing it
costs 0.016-0.030 under strong JPEG/downscale. Accurate tier (+frozen CLIP), 3 seeds:
mean 0.9416 ± 0.0032, JPEG 30 0.869 ± 0.008 — much more robust than the fast
tier, stronger on Midjourney/GLIDE/BigGAN, weaker on ADM/VQDM. Baseline Community Forensics
re-evaluated (not retrained) on the same images: mean 0.9954, JPEG 30 0.918 —
fakeradar is NOT competitive on accuracy; the paper says so. All tables come
from `experiments/phase4/summarize.py` (writes `experiments/phase4/RESULTS.md`,
and with `--update-docs` the README table and `docs/benchmarks.md`
leaderboard) and `paper/make_tables.py` (writes
`paper/generated_tables_*.tex`). Run data: `runs/lens_law/phase4-*/`.

## 3. Published

https://huggingface.co/JitendraJha98/fakeradar-fast (public): `fast_v0.pt` =
fast tier seed 42, model card from `experiments/phase4/model_card.py`
(limits first; CC-BY-NC-SA-4.0). `MODEL_ZOO["fast-v0"]` points there. Re-upload
only the README when tables change. Also public since 2026-11-23:
https://huggingface.co/JitendraJha98/fakeradar-accurate (`accurate_v0.pt` =
accurate tier seed 42 incl. frozen CLIP, 393 MB; card from
`experiments/phase4/model_card_accurate.py`); `MODEL_ZOO["accurate-v0"]`
points there.

## 4. Paper

`paper/paper.tex` restructured (2026-11-22) around A (negative), B
(guarantee, the main contribution), C (honest benchmark). No LaTeX on this
laptop: compile on Overleaf. Remaining [TO FILL]: accurate tier 3rd seed,
same-data retrained baselines. 11 references added from web searches are
marked VERIFY in `paper/references.bib`. Status list:
`paper/INTEGRITY_CHECKLIST.md`. The user will rewrite it in their own voice.

## 5. How to run things on this laptop

- Python: `/c/Users/mjwea/AppData/Local/Microsoft/WindowsApps/python.exe`
  with `PYTHONUTF8=1` (Python 3.14; not on the bash PATH). torch 2.13.0+cpu
  with torchvision 0.28.0+cpu (a matched pair; a `timm` install once pulled a
  mismatched torch). Windows Smart App Control may block freshly installed
  DLLs once, and blocks `hf.exe`: use `python -m huggingface_hub.cli.hf`.
- Tests: `python -m pytest -q` (main, 104 pass), `python -m pytest
  experiments/lens_law/tests -q`, `python -m pytest experiments/fpr_guarantee/tests -q`;
  `ruff check src tests` (ruff 0.5.5).
- Editing gotchas: bash heredocs with `\n` inside Python strings get mangled
  (write scripts with the Write tool or use `chr(10)`); ruff reformats files,
  so re-read before Edit; repo is LF, write with `write_bytes(s.encode())`.
- Kaggle: CLI 2.2.4, `~/.kaggle/kaggle.json`, user `jitendrajha98`, 30 GPU-h
  per week (resets Fridays 05:30 IST), max 2 GPU sessions at once. Jobs:
  from `experiments/lens_law`, `python kaggle_jobs.py run <name> <entry.py>
  [--gpu] --timeout <s> [--pkg-dir ../../src/fakeradar] [--dataset-source ..]
  [--kernel-source ..] [--pull '<regex>'] [--no-wait]` → private notebook
  `fakeradar-<name>`. GenImage on Kaggle: `vtphatt2/genimage-stable-diffusion-v1-4`
  (train) and `vtphatt2/genimage-validation` (test), CC-BY-NC-SA-4.0; mount
  paths vary between notebooks (remap by the `GenImage/...` suffix). Training
  entry `experiments/phase4/jobs/train_fast.py` (constants at top: sizes,
  SEED, USE_COHERENCE); variants in `runs/lens_law/_jobs/`.
- Hugging Face: logged in on this machine as JitendraJha98 (token saved by
  the HF CLI; never write it anywhere).
- Long background watchers get killed when the laptop is low on memory: prefer
  short status/pull commands on a schedule.

## 6. Pending (decided with the user on 2026-11-23)

Done that day: CI fixed (colour codes broke two help-text tests on GitHub);
CLAUDE.md, docs/architecture.md, CHANGELOG.md, ROADMAP.md and the README
install section updated; the paper's baseline blank replaced by a stated
limitation (only `	ofill{re-run the literature sweep}` remains, on purpose);
a learning guide published as a private artifact
(https://claude.ai/artifact/8caHqCqappGZViHNcRNq7u).
Decisions: skip same-data baselines (limitation in the paper); skip more
generators for the detection-rate step; ONNX for the accurate tier and int8
are roadmap to-dos; skip training on many generators (scaling, not novelty);
skip the newer-phone audit (no dataset). Item 16 (evaluate on 2025-26
generators) was to be tried once, bounded, only if a usable public dataset
exists. Hugging Face Space demo (item 12) was being built.
User later: PyPI release, compile the paper on Overleaf, rewrite it in their
own voice, verify references, arXiv (needs an endorser for cs.CV), revoke the
Kaggle and HF tokens.

## 7. User decisions and standing rules

- Weights licence: code Apache-2.0, weights CC-BY-NC-SA-4.0 (training data
  licence; the user delegated the call, asked for Apache-2.0, it cannot be).
- The user commits themselves; never commit unless asked. Kaggle items stay
  private and prefixed `fakeradar-`; never touch other items (`nyaya-...`).
- Never invent, retype or "improve" numbers; never score a held-out split or
  change a gate threshold after seeing scores; record every decision in the
  active ledger.
- Never disable TLS verification (Kaggle cannot verify the FODB server; FODB
  is handled on the laptop). FODB: research only, no redistribution.
- The user pasted Kaggle and HF tokens in chat; never repeat or store them.
- The user works in short usage windows and wants work to continue without
  asking; keep this file and the active ledger current.

**Why:** usage limits and model switches interrupt sessions; this file plus
the active ledger are the only record that survives.
**How to apply:** read this file, then the active ledger's latest `NEXT:`
block; check Kaggle status before assuming a job is still running.
