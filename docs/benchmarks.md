# Benchmark protocol

## Generalization (the number that matters)
Train on **one** generator: GenImage / Stable Diffusion v1.4 (real = ImageNet
"nature" split). Evaluate on all 8 GenImage generators (ADM, BigGAN, GLIDE,
Midjourney, SDv1.4, SDv1.5, VQDM, Wukong) + in-the-wild sets (e.g. Chameleon)
as they are integrated. Metrics: AUROC, AP, acc@0.5, TPR@5%FPR, ECE, and
abstention rate (share of `uncertain` verdicts at the detector thresholds).

Splits (`fakeradar prepare-genimage ROOT -g stable_diffusion_v_1_4 -t all`):
- `*_train.csv` — SD1.4 `train` split minus the hold-out
- `*_val.csv` — stratified 5% hold-out of SD1.4 `train`: model selection +
  temperature calibration only
- `*_test.csv` — GenImage `val` split of every generator: reported numbers
  only, never used for selection or calibration

Each generator row scores that generator's AI images against the pooled real
images of the test manifest; `MEAN` is the unweighted mean of per-generator AUROC.

Run:
```bash
fakeradar benchmark data/manifests/genimage_stable_diffusion_v_1_4_test.csv     --checkpoint best.pt --out docs/leaderboard.md --scores-csv runs/fast_v0/test_scores.csv
```

## Robustness (mandatory in every release)
Same eval set, pushed through: JPEG q∈{90,70,50,30}, downscale {0.75, 0.5},
Gaussian blur σ=1. Command: `fakeradar robustness ... --out docs/robustness.md
--scores-csv ...`

## Raw scores
`--scores-csv` writes every per-image probability (`path,label,generator,prob_ai`,
plus `perturbation` for robustness). Commit these next to any published table —
every number must be recomputable from them (paper/INTEGRITY_CHECKLIST.md).

## Leaderboard
<!-- leaderboard:start -->
| model | train data | mean AUROC (8 gen) | AUROC @ jpeg50 | weights |
|---|---|---|---|---|
| fast-v0 | GenImage SD1.4, 95k images, 5 epochs | 0.9367 ± 0.0014 | 0.8385 ± 0.0068 | [HF](https://huggingface.co/JitendraJha98/fakeradar-fast) (CC-BY-NC-SA-4.0) |
| accurate-v0 (3 seed) | GenImage SD1.4, 95k images, 5 epochs | 0.9416 ± 0.0032 | 0.9093 ± 0.0050 | [HF](https://huggingface.co/JitendraJha98/fakeradar-accurate) (CC-BY-NC-SA-4.0) |
| Community Forensics (reference, re-evaluated) | thousands of generators | 0.9954 | 0.9327 | [HF](https://huggingface.co/OwensLab/commfor-model-384) (MIT) |
<!-- leaderboard:end -->

Rules: numbers must be reproducible from a manifest + checkpoint; clean and
perturbed columns are both required; failure cases welcome in issues.
