# Lens-law evidence with a guaranteed false-accusation rate

Date: 2026-11-14. Status: proposed, waiting for the owner's go-ahead. Not committed.

## 1. Why this exists

fakeradar today is a careful build of ideas that are already published (hybrid
low-level + CLIP fusion: AIDE, Co-SPY), and it has no trained weights or
results. The accuracy race is led by ensembles of multi-billion-parameter
models. A 12M-parameter model will not win that race, so this plan does not
enter it.

It tests two things instead, built as one pipeline:

- **A. The lens-law test.** A photograph is taken through one lens, and optics
  fixes how blur relates to distance: blur size is proportional to
  `|1/distance - 1/focus distance|`. Plotted against inverse depth, the blur
  of one photo forms a V with a single focus point and the same slope on both
  sides. A depth network gives inverse depth up to an unknown scale and
  offset, which keeps a V a V, so no camera metadata is needed. A generator
  has no lens. The hypothesis is that generated images break this law more
  than photographs do, and that the gap survives resizing and JPEG because
  blur is coarse structure.
- **B. A guaranteed false-accusation rate.** Any score becomes a decision with
  a bounded rate of flagging real photos, by setting the threshold from real
  photos only (conformal prediction). The bound does not depend on which
  generator made the fakes. B is how A's score gets its threshold, and later
  how any learned score gets its threshold.

Nothing here can promise "best in market" or "perfect". What the plan does
guarantee is that every outcome, including a null result, is measured against
thresholds fixed in advance and can be reported honestly.

## 2. Claims under test

| ID | Claim | Tested in |
|---|---|---|
| H1 | The pipeline can measure the law: real wide-aperture photos form a V, and real photos re-rendered with deliberately wrong blur fail. | Phase 1 |
| H2 | On images with visible depth of field, generated images score worse than photographs. | Phase 2 |
| H3 | H2 still holds after a 0.5x downscale plus JPEG 70. | Phase 2 |
| H4 | The score reads blur geometry, not generator texture: photographs passed through a generator's autoencoder score like photographs. | Phase 2 |
| H5 | With real-only calibration, the rate of flagged real photos on cameras and platforms not used for calibration stays at the chosen level. | Phase 3 |

H1 comes first on purpose. A null on H2 only means "generators obey the law"
if H1 has already shown the measurement works.

## 3. Goals, non-goals, anti-goals

**Goals**
- A go/no-go answer on the lens-law test, with figures, from thresholds fixed
  before any data is scored.
- A calibration layer that gives any score a stated false-accusation rate, and
  an audit showing whether that rate holds on unseen cameras and platforms.
- If the gates pass: both pieces inside fakeradar, with the existing learned
  model trained and benchmarked honestly.

**Non-goals**
- Beating large foundation-model detectors on accuracy.
- Detecting partial edits or inpainting.
- A verdict on every image. The lens-law test only applies when depth of
  field is visible, and it abstains otherwise.
- Proving an image is real. Passing the test is weak evidence; failing it is
  the informative direction.

**Anti-goals (these make the work a failure even if the code runs)**
- Any number in a table that did not come from a reproducible run with raw
  scores saved (`paper/INTEGRITY_CHECKLIST.md`).
- Changing a gate threshold, the statistic, or the split after seeing test
  results.
- Claiming "first" without re-running the literature search, including the
  older splicing-forensics work on blur inconsistency, which has not been
  searched yet.
- Anything made public on Kaggle, or any image set republished, without the
  owner's say-so.
- A token or key in the repo, a notebook, a log, or the chat.

## 4. The pipeline

```
image
  |-- depth network ----------> inverse-depth map
  |-- blur estimator ---------> blur map + confidence
  |
  +-- confident patches away from depth edges
        -> points (inverse depth, log blur)
        -> fit thin-lens curve  b(D) = 0.5 * log(c0^2 + k^2 (D - Df)^2)
        -> lens-law score = robust residual of that fit
        -> applicable?  (enough patches, blur spread of at least 2x)
              no  -> abstain
              yes -> real-only calibration -> p-value -> verdict at level alpha
```

Secondary score: residual of a looser "valley" fit (blur falls then rises
with depth, any shape). It does not need the depth scale to be accurate, so
it separates "blur is not a function of depth at all" from "blur follows
depth but with the wrong shape".

Each verdict comes with a picture a person can check: the V-plot for that
image and an overlay of the patches that break the law.

## 5. Phases

Only Phases 0 to 2 are specified in detail. Later phases depend on the Phase 2
result and are outlined.

### Phase 0: setup

1. Install the Kaggle command-line tool into the laptop's Python and confirm
   the token works, without printing it.
2. Push a two-minute private smoke-test notebook through the API. It reports
   GPU type and memory, whether internet is on, disk space, and library
   versions. This settles the Kaggle facts marked "confirm" in section 7.
3. Embed the experiment code in each private Kaggle notebook and confirm the
   notebook can import it. Nothing is pushed to the public GitHub repo.
4. Resolve every "confirm at setup" entry in section 6 that Phases 1 and 2
   need, swapping in the listed fallback where a licence or download blocks
   use.
5. Write the split file (section 5.4) before any image is scored.

Exit: smoke test passes, and no unresolved "confirm" entries remain for
Phases 1 and 2.

### Phase 1: can we measure the law? (Gate M)

**Data.** Same-scene pairs shot at wide and narrow aperture; all-in-focus
photos for controlled renders (the narrow-aperture half of the pairs); real
wide-aperture photos from many cameras, chosen by the f-number in their EXIF
data.

**Steps.**
1. Depth: run the depth network on every image.
2. Blur estimator v0: the classical edge-based method (gradient ratio under
   re-blur, Zhuo and Sim 2011), the same family the closest prior work uses.
3. Blur estimator v1, built only if v0 fails the gate under transport: a
   small network trained on sharp real patches with synthetic defocus (disc
   and Gaussian kernels applied in linear light, highlights boosted), pushed
   through the repo's existing transport chain. It outputs blur size and a
   confidence. It is trained on real photos only; no generated image is used
   to train anything in A.
4. Controlled renders: from each all-in-focus photo and its depth map, render
   a lawful version (thin-lens blur, random focus plane and strength) and
   three unlawful versions: step (constant background blur behind the
   subject), position-based (blur set by image row, ignoring depth), and
   missing-foreground (near objects left sharp). Same renderer, same photo,
   same depth map; only the law differs. The test sees only the rendered
   image and must re-estimate depth from it, which also measures how much
   blur bends the depth estimate.

**Gate M** (tuned on the dev split, confirmed once on the test split):

| Check | Pass condition |
|---|---|
| M1, real photos form a V | On aligned aperture pairs, take depth from the narrow-aperture shot and blur from its wide-aperture twin. Among pairs whose wide shot has a blur spread of at least 2x, the thin-lens fit removes at least half of the blur variance left by a constant-blur fit in at least 80% of pairs. With depth taken from the wide shot itself, as the test must do in practice, the same holds in at least 70%. |
| M2, blur estimator tracks aperture | In at least 90% of aperture pairs, the wide-aperture image shows the larger blur spread. |
| M3, broken law is detected | Lawful vs unlawful renders: AUROC of at least 0.85 for each violation type at moderate strength (background blur about 4x the in-focus blur). |
| M4, at a usable threshold | At the threshold that flags 5% of real wide-aperture photos, at least 50% of unlawful renders are flagged and at most 10% of lawful renders. |
| M5, under transport | After a 0.5x downscale plus JPEG 70: M3 drops by at most 0.10 AUROC, and M4's unlawful rate stays at or above 35%. |

M1 is the only check on real optics. M3 and M4 use renders that are lawful by
construction, so they test detection of violations, not whether real photos
obey the law.

**Stop rule.** Two rounds of fixes are allowed (estimator v1; a second depth
model). If Gate M still fails, A stops, the measurement report is written,
and work moves to Phase 3 using learned scores only.

### Phase 2: do generators break the law? (Gate A)

**Data.**
- Real: photographs with visible depth of field, selected by the
  applicability rule and not by eye.
- Generated: for each real photo, its text description plus its real focal
  length and f-number become the prompt for at least four open-weight
  generators run on Kaggle. Content and camera settings are therefore matched
  between the two classes. One generator is held out: it is generated and
  scored for the test split only.
- Texture control: the real photos encoded and decoded by a generator's
  autoencoder, with no generation step.
- Optional: images from closed commercial generators, from a public set with
  usable terms or a small paid batch, if the owner approves.

Target size: about 800 real photos that pass the applicability rule (480 dev,
320 test), with the same prompts for each generator.

**Format equalization.** Every image, real or generated, goes through the
same final resize-and-encode step before scoring, so file format cannot act
as a shortcut for the reference detector.

**Conditions.** Clean; 0.75x plus JPEG 85; 0.5x plus JPEG 70.

**Reference detector.** One public learned detector with released weights is
scored on the same images in the same conditions, to show whether the
lens-law score catches images it misses after transport.

**Gate A** (test split, applicable images only; the 5% threshold is set on
dev real photos and applied unchanged to the test split):

| Check | Pass condition |
|---|---|
| A1, effect | Mean over generators: AUROC of at least 0.70 and detection rate of at least 25% at a 5% false-accusation rate. The held-out generator is reported separately. |
| A2, transport | At 0.5x plus JPEG 70, mean AUROC is at least 0.65 and has fallen by at most 0.07. |
| A3, not texture | Autoencoder-reconstructed real photos are flagged at no more than 10% at the 5% threshold. |
| Coverage | The share of images where the test applies is reported for both classes. No threshold, but it is stated plainly. |

**What each outcome means.**
- All pass: lens-law evidence works. It becomes the headline, and Phases 3 to
  5 proceed with it.
- A1 fails with Gate M passed: a null result. Generators obey the lens law to
  within what we can measure. The per-generator compliance chart is still
  reported, and B becomes the headline using learned scores.
- A1 passes and A3 fails: the score is partly a texture detector. It is
  reported as that, with no physics claim.
- A1 passes and A2 fails: the evidence exists but does not survive transport.
  The claim is narrowed to clean images.

Deliverable: a one-page go/no-go memo with the thresholds above, the measured
values, and the figures.

### 5.4 Split rule (fixed in Phase 0)

- Every source is split into dev (60%) and test (40%) by a hash of the image
  ID. Real photos are split by photographer or camera where that is known, so
  one photographer never appears on both sides.
- All choices (patch size, confidence cut, applicability rule, estimator
  version, depth model) are made on dev. The test split is scored once.
- One generator is never looked at during development.

### Phase 3: guaranteed false-accusation rate (outline)

- A calibration module: real photos in, threshold or p-value out, at a chosen
  level with a stated confidence. Calibration is stratified by transport
  state read from the file itself (file type, JPEG quality from the
  quantization tables, size bucket), with a pooled fallback when a stratum is
  thin.
- Scores wrapped: the lens-law score if Gate A passed, the reference learned
  detector, and later fakeradar's own model.
- Audit: real photos from many phones, each in a camera-original version and
  after real platforms. Calibrate on some phones and test on others.
- Phone photos rarely show depth of field, so the phone set exercises the
  learned score. The lens-law score is calibrated and audited on real photos
  with visible depth of field: the Phase 2 real set, calibrated on some
  photographers and audited on others, clean and after simulated transport.
- Gate B: on held-out phones and platforms, the measured false-accusation
  rate is inside the binomial 95% tolerance of the chosen level, at 5% and at
  1%, per stratum. The same audit is run for a fixed 0.5 threshold and for
  temperature scaling, and whatever it shows is reported. Detection rate at
  the guaranteed level comes from the Phase 2 images.
- If held-out phones break the guarantee, the size of the break is reported,
  a more conservative variant is added, and the claim is cut back to what the
  audit supports.

### Phase 4: into fakeradar, and training the learned model (outline)

- Move the pieces that passed their gates from `experiments/` into the
  package: an evidence module, the calibration module, and CLI output showing
  the verdict at a guaranteed level, the p-value, and the V-plot when the
  test applies.
- Train the existing learned model on Kaggle with the JPEG shortcut closed
  (both classes given the same compression history), then run the benchmark
  and robustness harness. This needs a training set reachable from Kaggle,
  which is not confirmed yet.
- Fix the two claims in the paper draft: the fusion gate is one global
  parameter, not a per-image explanation; and the coherence channel needs its
  ablation before any claim about it.

### Phase 5: write-up (outline)

Rewrite the paper around what was measured, positioned against the prior work
in section 10, with real tables, V-plots, the per-generator compliance chart,
and the false-accusation audit.

## 6. Data and models

"Verified" means a web search on 2026-11-14 confirmed the stated fact.
"Confirm" means it must be checked in Phase 0 before use.

| Item | Role | Status |
|---|---|---|
| Depth Anything V2 Small | Depth network that ships in the tool | Verified: Apache-2.0, 24.8M parameters, outputs relative inverse depth. Whether the scale is accurate enough for straight V arms is what Gate M measures. |
| A second depth model | Check that results do not depend on one depth network | Confirm. Research-only licence is acceptable because it is not shipped. |
| Unsplash Lite | Real photos from many cameras, with EXIF aperture and focal length | Verified: about 25,000 photos, free for commercial and non-commercial use, images may not be redistributed. Confirm: that it also carries text descriptions usable as prompts. Fallback: caption with a small open captioning model. |
| DPDD | Same-scene wide and narrow aperture pairs | Verified: 500 scenes, downloadable from the authors' repository. Confirm: licence terms, and that the two shots of each pair are pixel-aligned (check M1 depends on it). Fallback: EBB!. |
| EBB! | Alternative aperture pairs | Verified: about 5,000 pairs, one camera and lens. Confirm: download needs registration and the data licence is unclear. |
| Z-Image-Turbo | Generator | Verified: Apache-2.0, fits in 16 GB. |
| FLUX.2 klein 4B | Generator | Verified: Apache-2.0, 8 to 13 GB depending on source. |
| SDXL, Sana, FLUX.1-schnell, SD 3.5 Medium | Further generators, one of them held out | Confirm: licence, gating, and fit in 16 GB. Gated ones need the owner to accept the licence. |
| FODB | Real photos from 27 phones, camera-original plus five platforms | Verified: more than 23,000 images of 143 scenes, publicly available. Confirm: download link and terms. Fallback: VISION. |
| A public learned detector | Reference score | Confirm: choose one with released weights and a usable licence. |
| Closed commercial generators | Stronger test of H2 | Optional. Needs a public set with usable terms or a paid batch approved by the owner. |

Images that may not be redistributed are fetched inside private notebooks and
never republished. The repo stores only IDs and URLs.

## 7. Compute plan (Kaggle)

Confirmed on 2026-11-14 by the Kaggle CLI and a private smoke-test notebook:
30 GPU-hours per week, two T4 GPUs with 15 GB each, 4 CPUs, 33 GB RAM, 21 GB
working disk, internet on, and the API can push a private notebook with GPU
and internet switched on. The session length cap (reported as about 12 hours
by third-party pages) is not confirmed; jobs are planned for 8 hours or less.

| Phase | Work on GPU | Estimated GPU-hours |
|---|---|---|
| 0 | Smoke test | under 1 |
| 1 | Depth on control sets; blur estimator v1 if needed | 3 to 6 |
| 2 | Image generation (the main cost); autoencoder reconstructions; scoring in three conditions | 20 to 25 |
| 3 | Scoring the audit photos with the reference detector | about 5 |
| 4 | Training the learned model; benchmark runs | 15 to 25 |
| Total | | about 45 to 60 |

At a cap of 25 GPU-hours per week that is two to three weeks of quota. With
analysis between runs, the elapsed time is longer. These are estimates.

**Job rules.**
- Each job is planned for 8 hours or less, writes results as it goes, and
  skips finished items when re-run.
- Outputs are pulled to a git-ignored `runs/` folder on the laptop. Large
  intermediate data, such as generated images, lives in private Kaggle
  datasets.
- Everything created on Kaggle is private and named with a `fakeradar-`
  prefix. Nothing that already exists on the account is edited or deleted.
- Jobs keep running on Kaggle between working sessions; results are collected
  when work resumes.

## 8. Risks and what is done about each

| Risk | Handling |
|---|---|
| Blur and depth estimates are too noisy to see the law even in real photos. | Gate M with a stop rule. |
| Depth networks use blur as a depth cue, so the two measurements are not independent. | The controlled renders measure this directly. The effect can hide violations but should not create them, so it costs detection rate, not validity. |
| The score reacts to generator texture, not geometry. | Check A3, plus the unlawful renders, which are made from real photos. |
| Real photos that legitimately break the law: phone portrait mode, motion blur, reflections, photos of screens or posters, composites. | They are part of the real calibration set, so they set the threshold. Their share is reported. |
| Few images have visible depth of field. | Coverage is reported for both classes. The test abstains otherwise, and the learned score covers those images. |
| Only open-weight generators are tested. | Stated as a limit. Closed generators are added if the owner approves a source. |
| Calibration photos do not represent the photos seen later. | The audit on held-out phones and platforms, and the conservative variant. |
| Kaggle limits change or a dataset's terms block use. | Phase 0 confirms both, and every dataset has a fallback. |
| Prior work appears, or exists in the unsearched splicing-forensics literature. | Search again before any novelty claim, and narrow the claim to what the search supports. |

## 9. Decisions

**The owner decided**
- Pursue A and B.
- GPU comes from Kaggle, driven from this machine with the owner's token.

**My calls (the owner can overrule any of them)**
- A runs first as a time-boxed go/no-go, and B is the fallback headline.
- The gate thresholds in section 5.
- Experiment code stays in `experiments/lens_law/` until its gate passes, and
  only then moves into `src/fakeradar/`.
- Depth Anything V2 Small is the shipped depth model, because of its licence.
- The simplest blur estimator that passes Gate M under transport is used.
- Code reaches Kaggle embedded in each private notebook, not through a public
  GitHub push. (Changed on 2026-11-14 from "a private dataset": one push is
  atomic, and privacy is the same.)
- This document is written to the repo but not committed until the owner asks.
- Training the existing learned model is in scope, as Phase 4, after the
  gates.

**Left to the builder**
- Patch size, confidence cut, and the optimizer for the fit, tuned on dev.
- The architecture of blur estimator v1.
- Which reference detector and which second depth model, within section 6.
- Exact image counts within the GPU budget, and the notebook layout.

Repo conventions that still apply: ruff rules and line length, tests under
`tests/`, torch imports inside CLI commands, and no resizing of inputs to the
existing forensic branches.

## 10. Prior work this is positioned against

- AIDE (ICLR 2025) and Co-SPY (CVPR 2025): hybrid low-level plus CLIP fusion.
- Jeon and Woo (CIKM 2025): a defocus map fed to a trained classifier, with no
  depth and no law test; they report compression disturbing the edge-based
  blur estimator.
- Sarkar et al. (CVPR 2024): perspective and shadow cues detect generated
  images that fool signal-based detectors.
- PhyBench (2024) found optics the physics category generators handle best.
  Bokeh Diffusion (2025) and arXiv 2510.06215 report poor defocus consistency
  in base models. The outcome of H2 is therefore open.
- QuAD (2026): quality-conditioned likelihood ratios over near-duplicate
  copies, with no formal guarantee. A WACV 2025 workshop paper applies
  post-hoc calibration to synthetic-image detectors. RAW (NeurIPS 2024) and
  MCP (ACL 2025) apply conformal prediction to watermarks and to text.
- Grommelt et al. (2024): GenImage reals are JPEG and fakes are PNG, and
  detectors learn that shortcut.
- NTIRE 2026 challenge: the accuracy bar, led by DINOv3 ensembles.

No paper was found that tests the thin-lens law on generated images (two
searches) or applies conformal false-positive control to unwatermarked
AI-image detection (three searches). These are search results from one day,
not proof of absence.

## 11. What the owner needs to provide

**Required**
1. A Kaggle account that is phone-verified. GPU and internet in notebooks
   need it.
2. A fresh Kaggle API token saved as a file on this laptop at
   `~/.kaggle/kaggle.json`. It is never pasted into the chat, and it can be
   revoked when the work is done.
3. Permission for this scope: private datasets and notebooks only, nothing
   existing touched, and a weekly GPU-hour cap (proposed: 25 of about 30).
4. An OK to install the Kaggle command-line tool and a few analysis packages
   into the laptop's Python.
5. The presentation date, if there is one.

**Only the owner can do, if needed**
6. Accept the licence of any gated model on Hugging Face and supply a read
   token, stored as a file and never pasted into the chat.
7. Approve any paid spend, for example a small batch of images from closed
   commercial generators. The default is none.
8. Say whether the plan and code may be committed, and on which branch.

## 12. Sources for the facts in sections 6 and 7

- Depth Anything V2: https://github.com/DepthAnything/Depth-Anything-V2
- Unsplash dataset: https://unsplash.com/data and https://github.com/unsplash/datasets
- DPDD: https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel
- EBB!: https://aiff22.github.io/pynet-bokeh.html
- FLUX.2 klein 4B: https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
- Z-Image-Turbo: https://www.thundercompute.com/blog/z-image-turbo-comfyui
- FODB: https://arxiv.org/abs/2011.02241
- Kaggle limits (third-party summary): https://fastero.com/blog/kaggle-notebooks
