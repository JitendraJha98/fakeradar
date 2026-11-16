# Gate B report: does the calibrated false-accusation rate hold on unseen phones?

**Verdict: Gate B PASSED** (0 of 10 checks missed; the rule allows 1 and no rate above 2x its level).

- Detector: Community Forensics (OwensLab/commfor-model-384). Higher score means more likely generated.
- Photos: 23,105 real photos, 16 calibration phones, 11 audit phones, versions: facebook, instagram, orig, telegram, twitter, whatsapp.
- Held out of calibration: telegram.
- Thresholds come from calibration phones only; rates below are on audit phones only.

## The ten checks

| Version | Level | Measured rate | 95% interval | Check |
|---|---|---|---|---|
| facebook | 5.00% | 5.93% | 3.69% to 8.56% | pass |
| instagram | 5.00% | 5.48% | 2.99% to 8.03% | pass |
| orig | 5.00% | 5.48% | 2.61% to 10.01% | pass |
| twitter | 5.00% | 4.72% | 3.05% to 8.11% | pass |
| whatsapp | 5.00% | 5.55% | 3.24% to 7.98% | pass |
| facebook | 1.00% | 1.28% | 0.51% to 2.23% | pass |
| instagram | 1.00% | 0.89% | 0.51% to 1.60% | pass |
| orig | 1.00% | 0.89% | 0.32% to 2.48% | pass |
| twitter | 1.00% | 1.02% | 0.45% to 1.72% | pass |
| whatsapp | 1.00% | 0.77% | 0.45% to 1.91% | pass |

## Held-out version (telegram), thresholded with the pooled calibration

| Level | Measured rate |
|---|---|
| 5.00% | 3.57% |
| 1.00% | 0.89% |

## Comparisons (audit phones)

Share of real photos flagged per version. The detector's own cut is a score of 0
(probability 0.5). Temperature scaling is not listed because it rescales scores
without moving that cut, so its decisions are identical to the fixed cut.

| Version | Fixed cut (0.5) | Pooled 5% | Per-version 5% | Pooled 1% | Per-version 1% |
|---|---|---|---|---|---|
| facebook | 0.13% | 3.83% | 5.93% | 0.83% | 1.28% |
| instagram | 0.00% | 3.83% | 5.48% | 0.70% | 0.89% |
| orig | 0.38% | 10.01% | 5.48% | 2.93% | 0.89% |
| telegram | 0.00% | 3.57% | 3.57% | 0.89% | 0.89% |
| twitter | 0.06% | 4.34% | 4.72% | 0.83% | 1.02% |
| whatsapp | 0.06% | 4.08% | 5.55% | 0.70% | 0.77% |

## Pass rule (fixed before any score existed)

A check passes if the measured rate is at most 1.5x the level, or if the low end of its 95% interval is at or below the level. Intervals resample calibration and audit phones together (2,000 draws). The gate passes if at most 1 check misses and no rate exceeds 2x its level.

## Detection rate of generated images at these thresholds

Added 2026-11-16, after the gate. Share of generated images flagged, by the detector's own
0.5 cut and by the calibrated thresholds (range over the five platform versions).

| Generator | Condition | Fixed cut (0.5) | Calibrated, 5% level | Calibrated, 1% level |
|---|---|---|---|---|
| SDXL (200 images) | as generated | 96% | 99% to 100% | 98% to 99% |
| SDXL (200 images) | 0.75x + JPEG 85 | 90% | 100% | 94% to 99% |
| SDXL (200 images) | 0.5x + JPEG 70 | 82% | 96% to 98% | 90% to 94% |
| FLUX.2 klein 4B (10 images) | as generated | 70% | 80% to 90% | 80% |
| FLUX.2 klein 4B (10 images) | 0.75x + JPEG 85 | 30% | 70% to 80% | 40% to 60% |
| FLUX.2 klein 4B (10 images) | 0.5x + JPEG 70 | 20% | 80% | 30% to 80% |

What this shows: after a 0.5x downscale and JPEG 70, the detector's own cut catches 82% of
the SDXL images; the thresholds calibrated to a 1% false-accusation rate catch 90% to 94%,
and those calibrated to 5% catch 96% to 98%.

Limits, stated plainly:

- One generator with a usable sample. FLUX.2 klein produced only 10 images before the job
  stopped it as too slow, and Z-Image-Turbo crashed the notebook; 10 images are indicative only.
- SDXL is an older generator (2023) and is very likely among the reference detector's training
  generators, so this is not a test on an unseen generator.
- Transport for generated images is simulated (resize + JPEG), not real platform sharing,
  while the thresholds come from real platform-shared photos.
- Prompts are descriptions of real Unsplash photos; images are 1024 x 1024.
