# Architecture

## Design principles
1. **Never resample the evidence.** Forensic branches receive native-resolution
   crops; only the semantic branch sees a resized global view.
2. **Complementary views over one clever trick.** Spatial gradient statistics,
   spectral statistics, and semantic features fail in *different* ways; the
   gated fusion learns how much to trust each, per training distribution.
3. **Frozen semantics.** The CLIP backbone is never fine-tuned — fine-tuning
   is what destroys its unseen-generator generalization.
4. **Robustness is a training-time property.** The transport-chain augmenter
   (JPEG/WebP/resize/blur/noise) runs on every training image.
5. **Calibrated, abstaining output.** Temperature-scaled probability with an
   explicit `uncertain` band; per-branch scores + gate weights are exposed.

## Branches
| branch | input | front-end | encoder | dim |
|---|---|---|---|---|
| gradient | crop 256² | SRM(12) + NPR(3) + |∇|(1) + coherence(1) = 17ch | ResNet-18 | 512 |
| frequency | crop 256² | radial log-power profile + band stats | MLP | 256 |
| semantic | resized 224² | CLIP ViT-B/16 (frozen) | linear proj | 768 |

**Gradient-field stack.** SRM high-pass residuals suppress content and keep
synthesis noise; the NPR-style down-up residual isolates up-sampling traces;
Scharr gradient magnitude plus *structure-tensor coherence* of the gradient
field expose the anomalous local gradient alignment of synthetic texture.
The stack is always computed in fp32 (autocast disabled): in fp16 the
coherence ratio drifts by O(1), which under CUDA AMP would make the trained
features differ from fp32 inference. Only the ResNet encoder runs in AMP.

**Fusion.** Per-branch projections → learned softmax gate → MLP → logit, with
auxiliary per-branch logits (deep supervision; also reported per image) and a
temperature buffer fitted post-hoc on validation data. The gate is one global
set of weights learned in training, the same for every image: it says how much
the model trusts each branch overall, not which branch fired on a given image.

**Compression equalization.** Some training sets store real photos as JPEG and
generated images as PNG (GenImage does), so "has JPEG traces" becomes a
shortcut for "real". `preprocessing.load_image_equalized` gives every image
that is not already a JPEG (judged from the file header) one JPEG pass at a
chosen quality. Training config key `data.equalize_jpeg_quality` (stored in the
checkpoint; `--resume` refuses a changed value) and `--equalize-jpeg` on
`benchmark`, `robustness` and `calibrate`. Off by default. On GenImage it
raised the fast tier's mean AUROC from 0.902 to 0.937.

**Inference.** Up to `max_crops` native crops from a 3×3 grid (center first,
then corners, then edge midpoints — up to 9); logits are averaged; small
images are the only case that is ever upscaled.

**Calibration.** `fakeradar calibrate val.csv --checkpoint best.pt` fits the
temperature on held-out logits (also runs automatically after training via
`run.calibrate`); the checkpoint stores the fitted value + ECE before/after.
Logits are collected with the inference protocol (mean over the same crop
grid), since averaging K crops shrinks the logit spread. The fit is a bounded
grid search over T ∈ [0.05, 20] (the NLL is convex in 1/T), with a warning if
the optimum hits a bound (tiny or perfectly separable hold-out).

## Guaranteed false-accusation rate
`fakeradar calibrate-fpr REAL.csv --checkpoint best.pt --alpha 0.01` scores a
manifest of **real** photos only (`label` 0; the `generator` column names the
stratum, for example the sharing platform) and sets split-conformal thresholds
(`evaluation/guarantee.py`): the ⌈(n+1)(1−α)⌉-th smallest score per stratum, a
pooled threshold for strata too small for level α. If new real photos are
exchangeable with the calibration photos of their stratum, at most α of them
are flagged, whatever generator made the fakes. `fakeradar scan
--fpr-thresholds FILE [--stratum NAME]` then adds `flagged`, `alpha` and
`stratum` to each result. Audited on 23,105 FODB phone photos through five
platforms: `experiments/fpr_guarantee/GATE_B_REPORT.md`. Per-stratum
thresholds matter: one pooled threshold flagged 10% of camera originals at a
5% target.

## Published models
| zoo name | repo | contents |
|---|---|---|
| `fast-v0` | `JitendraJha98/fakeradar-fast` | gradient + frequency, 47 MB |
| `accurate-v0` | `JitendraJha98/fakeradar-accurate` | + frozen CLIP ViT-B, 393 MB |

Both trained on GenImage Stable Diffusion v1.4 only (95k images, 5 epochs,
compression equalized); weights CC-BY-NC-SA-4.0, inherited from the data.
Measured results, generated from the run files:
[experiments/phase4/RESULTS.md](../experiments/phase4/RESULTS.md).

## Extension points
- `register_branch` — new forensic views (reconstruction-error branch is a
  planned optional heavy branch; excluded from core to keep local-first)
- `PERTURBATIONS` — new robustness tests
- `MODEL_ZOO` — new published tiers
