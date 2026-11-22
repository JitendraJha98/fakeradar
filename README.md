# 🛰️ fakeradar

**Local-first AI-generated image detection.** Gradient-field forensics + frequency analysis + frozen foundation-model semantics, fused into one calibrated score — on your own machine. No uploads, no API keys, no cloud.

[![CI](https://github.com/JitendraJha98/fakeradar/actions/workflows/ci.yml/badge.svg)](https://github.com/JitendraJha98/fakeradar/actions/workflows/ci.yml)
[![CodeQL](https://github.com/JitendraJha98/fakeradar/actions/workflows/codeql.yml/badge.svg)](https://github.com/JitendraJha98/fakeradar/actions/workflows/codeql.yml)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/JitendraJha98/fakeradar/badge)](https://securityscorecards.dev/viewer/?uri=github.com/JitendraJha98/fakeradar)
[![codecov](https://codecov.io/gh/JitendraJha98/fakeradar/branch/main/graph/badge.svg)](https://codecov.io/gh/JitendraJha98/fakeradar)
[![PyPI](https://img.shields.io/pypi/v/fakeradar.svg)](https://pypi.org/project/fakeradar/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

---

## Why fakeradar

Most detectors fail in one of three ways. fakeradar is engineered against all three:

1. **They collapse on unseen generators.** Detectors trained on one model family memorize its artifacts. fakeradar fuses *complementary* forensic views — a gradient-field branch (our signature), a frequency-spectrum branch, and an optional frozen CLIP branch — so no single fingerprint is a point of failure, and the frozen semantic features carry generalization to generators never seen in training.
2. **They die on JPEG.** One WhatsApp forward destroys most published detectors. fakeradar trains *through* a simulated transport chain (JPEG/WebP re-encode, resize, blur, noise) and reports robustness tables, not just clean-set accuracy.
3. **They silently destroy evidence.** Resizing everything to 224×224 erases the high-frequency traces detection depends on. fakeradar scores **native-resolution crops** and aggregates — the forensic signal is never resampled away.

And one product decision on top: **everything runs locally.** The `fast` tier is ~12M params, CPU-friendly, fully offline, ONNX-exportable. Images being checked are often sensitive — they shouldn't have to leave your machine.

## Quickstart

**Try it in your browser:** [huggingface.co/spaces/JitendraJha98/fakeradar](https://huggingface.co/spaces/JitendraJha98/fakeradar). The fast model runs on your own device; the image is never uploaded.

Not on PyPI yet (planned); install from the repository:

```bash
pip install "fakeradar[hub] @ git+https://github.com/JitendraJha98/fakeradar"     # fast tier + model download
pip install "fakeradar[all] @ git+https://github.com/JitendraJha98/fakeradar"     # + CLIP branch, REST API, ONNX, demo, Hub

fakeradar scan photo.jpg holiday_album/ --pretrained fast-v0
```

```text
┌────────────────┬────────┬────────────┬────────────────────────────────┐
│ image          │  P(AI) │ verdict    │ branches                       │
├────────────────┼────────┼────────────┼────────────────────────────────┤
│ photo.jpg      │  0.912 │ likely_ai  │ gradient:0.94  frequency:0.87  │
└────────────────┴────────┴────────────┴────────────────────────────────┘
```

Python API:

```python
from fakeradar import Detector

det = Detector(tier="fast", checkpoint="runs/fast_v0/best.pt")
r = det.predict("photo.jpg")
print(r.prob_ai, r.verdict, r.per_branch)
```

Local REST API: `fakeradar serve --checkpoint best.pt` → `POST /v1/detect`
(multipart `file`; uploads over 25 MB → 413, undecodable → 400).

> **Weights:** `fast-v0` is on the Hugging Face Hub ([JitendraJha98/fakeradar-fast](https://huggingface.co/JitendraJha98/fakeradar-fast), CC-BY-NC-SA-4.0 because the training data is): `fakeradar scan photo.jpg --pretrained fast-v0`. The accurate tier ([JitendraJha98/fakeradar-accurate](https://huggingface.co/JitendraJha98/fakeradar-accurate), about 390 MB, needs `fakeradar[semantic]`) is markedly more robust to compression and resizing: `--tier accurate --pretrained accurate-v0`. It was trained on one generator (Stable Diffusion v1.4); read the model card's limits before relying on it, and set your own threshold on real photos with `fakeradar calibrate-fpr` rather than trusting the fixed 0.5 cut.

## Architecture

```mermaid
flowchart LR
    I[Image] --> C[Native-res crops<br/>no resizing]
    I --> S[Global view 224²]
    C --> G[Gradient-field branch<br/>SRM · NPR residual · ∇ magnitude · coherence]
    C --> F[Frequency branch<br/>radial power spectrum]
    S --> V[Frozen CLIP ViT<br/>optional, accurate tier]
    G --> X[Gated fusion + calibration]
    F --> X
    V --> X
    X --> O[P AI + per-branch scores + verdict]
```

| Tier | Branches | Params (trainable) | Offline | Target hardware |
|---|---|---|---|---|
| `fast` | gradient + frequency | ~12M | ✅ fully | any CPU / laptop |
| `accurate` | + frozen CLIP ViT-B | ~13M (+86M frozen) | after 1 download | GPU preferred, CPU works |

Design details: [docs/architecture.md](docs/architecture.md).

## Train your own

```bash
# train / val (5% hold-out of SD1.4 train) / test (GenImage val, all generators)
fakeradar prepare-genimage /data/GenImage --generators stable_diffusion_v_1_4 --test-generators all
fakeradar train configs/train_fast.yaml            # ~2-4h on one A100
fakeradar train configs/train_fast.yaml --resume runs/fast_v0/last.pt   # after an interruption

M=data/manifests/genimage_stable_diffusion_v_1_4_test.csv
fakeradar benchmark  $M --checkpoint runs/fast_v0/best.pt --out docs/leaderboard.md --scores-csv runs/fast_v0/test_scores.csv
fakeradar robustness $M --checkpoint runs/fast_v0/best.pt --out docs/robustness.md --scores-csv runs/fast_v0/robustness_scores.csv
```

Model selection and calibration use only the hold-out carved from the training
split; GenImage's own `val` split is the untouched test set. Training ends with
automatic temperature calibration on that hold-out — fit on the same
multi-crop logit the detector outputs — so the published probability is
calibrated, not just a ranking score. Re-calibrate on new held-out data anytime:

```bash
fakeradar calibrate data/manifests/genimage_stable_diffusion_v_1_4_val.csv --checkpoint runs/fast_v0/best.pt
```

`configs/train_accurate.yaml` trains the accurate tier the same way (needs
`fakeradar[semantic]`). In a training config, `detector:` may also be just
`fast` or `accurate`, so a pip install needs no `configs/` directory.

## Benchmarks

Protocol: train on **one** generator (GenImage SD v1.4), evaluate on **all eight** + in-the-wild sets, clean **and** perturbed. Full protocol in [docs/benchmarks.md](docs/benchmarks.md).

<!-- results:start -->
fast-v0, trained on GenImage SD v1.4 only; AUROC, mean ± std over 3 seeds;
compression history equalized. Full tables: [experiments/phase4/RESULTS.md](experiments/phase4/RESULTS.md).

| test generator | AUROC |
|---|---|
| stable_diffusion_v_1_4 | 0.9995 ± 0.0001 |
| stable_diffusion_v_1_5 | 0.9994 ± 0.0001 |
| wukong | 0.9955 ± 0.0004 |
| VQDM | 0.9257 ± 0.0057 |
| Midjourney | 0.9233 ± 0.0078 |
| glide | 0.8932 ± 0.0001 |
| BigGAN | 0.8977 ± 0.0076 |
| ADM | 0.8591 ± 0.0229 |
| **mean of 8** | **0.9367 ± 0.0014** |
<!-- results:end -->

We publish robustness tables alongside every release. A detector that only reports clean-set accuracy is hiding something.

## Local & edge deployment

```bash
fakeradar export-onnx runs/fast_v0/best.pt fakeradar_fast.onnx
```

The fast tier exports to a single ONNX graph (dynamic batch) for onnxruntime on
CPU, with int8 quantization on the roadmap. Export needs torch>=2.9; when
onnxruntime is installed the command verifies the exported graph produces the
same probabilities as eager PyTorch and prints the max difference.

## Limitations — read this

- Output is a **probabilistic signal, not proof**. Never treat one score as evidence for accusing a person or authenticating media in legal, journalistic, or academic-integrity settings.
- Detection is an **arms race**: new generators and adversarial post-processing can defeat any detector, including this one. Robustness has known theoretical limits.
- Scores are calibrated on our validation data; heavy out-of-distribution content (screenshots, scans, heavy edits, partial AI inpainting) may land in `uncertain` — that verdict exists on purpose.

## Contributing & community

New generator coverage requests, failure-case reports, and new forensic branches are the most valuable contributions — see [CONTRIBUTING.md](CONTRIBUTING.md) and the [new-generator issue template](.github/ISSUE_TEMPLATE/new_generator_request.md). Project plans: [ROADMAP.md](ROADMAP.md) · [LAUNCH_PLAN.md](LAUNCH_PLAN.md).

## Citation

```bibtex
@software{fakeradar2026,
  author = {Jha, Jitendra},
  title  = {fakeradar: local-first AI-generated image detection},
  year   = {2026},
  url    = {https://github.com/JitendraJha98/fakeradar}
}
```

Research paper (GRAFT) in preparation — see [paper/](paper/).

## License

Apache-2.0 — free for commercial and research use, with an explicit patent grant.
