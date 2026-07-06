# Benchmark protocol

## Generalization (the number that matters)
Train on **one** generator: GenImage / Stable Diffusion v1.4 (real = ImageNet
"nature" split). Evaluate on all 8 GenImage generators (ADM, BigGAN, GLIDE,
Midjourney, SDv1.4, SDv1.5, VQDM, Wukong) + in-the-wild sets (e.g. Chameleon)
as they are integrated. Metrics: AUROC, AP, acc@0.5, TPR@5%FPR, ECE.

Run:
```bash
fakeradar benchmark data/manifests/genimage_all_val.csv --checkpoint best.pt --out docs/leaderboard.md
```

## Robustness (mandatory in every release)
Same eval set, pushed through: JPEG q∈{90,70,50,30}, downscale {0.75, 0.5},
Gaussian blur σ=1. Command: `fakeradar robustness ... --out docs/robustness.md`

## Leaderboard
| model | train data | mean AUROC (8 gen) | AUROC @ jpeg50 | weights |
|---|---|---|---|---|
| fast-v0 | GenImage SD1.4 | *pending* | *pending* | HF (soon) |
| accurate-v0 | GenImage SD1.4 | *pending* | *pending* | HF (soon) |

Rules: numbers must be reproducible from a manifest + checkpoint; clean and
perturbed columns are both required; failure cases welcome in issues.
