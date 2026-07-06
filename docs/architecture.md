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

**Fusion.** Per-branch projections → learned softmax gate → MLP → logit, with
auxiliary per-branch logits (deep supervision + interpretability) and a
temperature buffer fitted post-hoc on validation data.

**Inference.** Up to `max_crops` native crops (center+corners); logits are
averaged; small images are the only case that is ever upscaled.

## Extension points
- `register_branch` — new forensic views (reconstruction-error branch is a
  planned optional heavy branch; excluded from core to keep local-first)
- `PERTURBATIONS` — new robustness tests
- `MODEL_ZOO` — new published tiers
