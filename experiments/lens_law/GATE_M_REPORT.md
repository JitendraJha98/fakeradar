# Gate M report: can the lens law be measured in a single image?

Date: 2026-11-15. Split: development only. The test split was never scored.

**Verdict: no, not well enough.** Gate M failed with both blur estimators and
after both fix rounds the design allows. Under the design's stop rule the
lens-law test (direction A) stops here.

Spec: `docs/superpowers/specs/2026-11-14-lens-law-guaranteed-fpr-design.md`.
Raw gate outputs: `figures/gate_m_v0_dev.json`, `figures/gate_m_v1_dev.json`.

## What was tested

A photo taken through one lens has blur proportional to
`|inverse depth - inverse focus depth|`: a V when blur is plotted against
inverse depth. The pipeline estimates blur per 32-pixel cell, estimates
inverse depth with a monocular depth network, fits the V, and uses the
misfit as the score. Gate M asks whether that score can tell lawful blur
from deliberately broken blur, against a threshold set on real photos.

## Data (development split)

| Set | Size | Role |
|---|---|---|
| iDFD real pairs (sharp shot, defocused shot, measured depth) | 349 scenes | M1, M2 |
| Unsplash photos at f/2.8 or wider | 631 | real-photo threshold |
| Unsplash photos at f/8 or narrower | 248 | sources for controlled renders (lawful, step, position, missing foreground) |

## Results

| Check | Needed | Edge-ratio estimator (v0) | Learned estimator (v1) |
|---|---|---|---|
| M1, real pairs form a V | 80% / 70% | 2.7% / 2.3% | 2.5% / 2.2% |
| M2, defocused shot shows more blur spread | 90% | 29% | **99%** |
| M3, AUROC lawful vs step / position / missing foreground | 0.85 each | 0.76 / 0.69 / 0.65 | 0.85 / 0.78 / 0.66 |
| M4, broken renders flagged at the real-photo 5% threshold | at least 50% | 6.3% | 4.9% |
| M5, AUROC after 0.5x downscale + JPEG 70 | drop of at most 0.10 | 0.59 / 0.58 / 0.62 | 0.61 / 0.59 / 0.65 |

Median lens-fit residual (log-blur units) with v1: lawful renders 0.22,
broken renders 0.27 to 0.39, **real wide-aperture photos 0.40**.

## What the numbers mean

- **The law is visible in real photos.** `figures/vplots_v1_wide_dev.png`
  shows clear V shapes for real wide-aperture photos. The lens model
  explains a median 44% of their blur variation.
- **The instrument is too noisy to test it.** Real photos fit the lens law
  worse than the deliberately broken renders, so a threshold set on real
  photos flags almost nothing. This is the blocker, and it is why M4 fails
  by a wide margin with either estimator.
- **Transport makes it worse, not better.** After a 0.5x downscale and JPEG
  70 the separation falls to near chance. The hope that coarse blur
  structure would survive transport is not supported.
- **M1 could not be tested fairly.** The iDFD defocused shots are about 2.2x
  blurrier than their sharp twins almost uniformly, with little change by
  depth. Pairs with strong depth-dependent defocus (DPDD) could not be
  downloaded by script. This would not change the verdict, because M3, M4
  and M5 fail regardless.
- **The learned estimator fixed two real problems.** All-in-focus shots now
  abstain (4% counted as applicable, was 83%), and M2 passes.

## Fix rounds used

1. **Learned blur estimator (v1).** A 0.92M-parameter network trained on
   1,352 real sharp photos with synthetic defocus. Held-out validation error
   about 14% where confident. Gate still failed.
2. **Second depth model.** Depth Anything V2 Large gave no gain; Depth Pro was
   worse. A multi-scale blur reading lowered the residual slightly but made
   58 to 62% of sharp photos count as applicable, so it was rejected.

Best setting found on development photos: real-photo residual 0.43 to 0.51,
still about twice the lawful-render level.

## What this does and does not show

- It shows that a single-image, monocular-depth measurement of the thin-lens
  law is not precise enough to detect moderate violations.
- It does **not** show whether image generators obey the lens law. That
  question (Phase 2) was never reached.
- A better instrument, such as stereo or dual-pixel depth, might change the
  answer. That is outside this plan.

## Reproduce

From `experiments/lens_law`:

```bash
python -m pytest tests -q                       # 119 tests
python kaggle_jobs.py run <name> jobs/<entry>.py [--gpu] --timeout <s>
python -m lenslaw.report <records.jsonl>        # development split only
```

Every decision made along the way is in the progress ledger,
`.superpowers/sdd/2026-11-14-lens-law-phase0-1/progress.md`.

## Addendum (2026-11-22): check M1 and M2 on DPDD

The owner downloaded DPDD (Abuolaim and Brown, ECCV 2020): 500 Canon scenes,
each shot at wide aperture and again stopped down, so blur varies strongly
with depth. Development-split scenes only (310), scored with both estimators
by `dpdd_m1.py`; the sharp twin carries the next-lower frame number, so pairs
are matched by order (grayscale correlation of matched pairs 0.986 to 0.999).
Summary: `runs/lens_law/dpdd_m1_summary.json`.

| Check | Needed | Edge ratio (v0) | Learned (v1) |
|---|---|---|---|
| M1, depth from the sharp twin | at least 80% | 59.0% (of 295) | 61.6% (of 245) |
| M1, depth from the wide shot itself | at least 70% | 62.0% (of 297) | 63.1% (of 244) |
| M2, wide shot has more blur spread | at least 90% | 98.7% | 96.1% |

Reading: on strongly defocused real photographs the lens law is clearly
measurable: about 60% of pairs fit the V with half or more of the blur
variance explained (median r2 about 0.57 to 0.60), against 2.5% on iDFD's
indoor pairs. M1 still misses its threshold and M3 to M5 fail as before, so
Gate M stays failed and direction A stays stopped. The earlier sentence "M1
could not be tested fairly" is superseded: M1 was tested and narrowly fails.
The accurate conclusion is that the law is measurable in most strongly
defocused photographs but not reliably enough per image to serve as evidence.
