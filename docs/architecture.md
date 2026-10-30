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
auxiliary per-branch logits (deep supervision + interpretability) and a
temperature buffer fitted post-hoc on validation data.

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

## Extension points
- `register_branch` — new forensic views (reconstruction-error branch is a
  planned optional heavy branch; excluded from core to keep local-first)
- `PERTURBATIONS` — new robustness tests
- `MODEL_ZOO` — new published tiers
