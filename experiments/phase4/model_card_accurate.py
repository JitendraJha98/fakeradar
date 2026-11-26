"""Write the Hugging Face model card for accurate-v0 from the measured results.

python experiments/phase4/model_card_accurate.py --out runs/hf/fakeradar-accurate/README.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aigibench_table import card_section  # noqa: E402
from model_card import FOOTER  # noqa: E402
from summarize import (  # noqa: E402  # noqa: E402
    ACCURATE,
    BASELINE,
    CONDITIONS,
    GENERATORS,
    MAIN,
    ROOT,
    available,
    benchmark,
    fmt,
    robustness,
)

HEADER = """---
license: cc-by-nc-sa-4.0
library_name: pytorch
pipeline_tag: image-classification
tags:
- ai-generated-image-detection
- image-forensics
- synthetic-media
datasets:
- GenImage
---

# fakeradar accurate-v0

The accurate tier of [fakeradar](https://github.com/JitendraJha98/fakeradar):
the fast tier's gradient-field and frequency branches plus a frozen CLIP
ViT-B/16 branch, fused by a learned gate into a temperature-calibrated P(AI).
About 13M trainable parameters plus the 86M frozen CLIP backbone, which is
stored in the checkpoint (about 390 MB). For a small, fully offline model see
[fakeradar-fast](https://huggingface.co/JitendraJha98/fakeradar-fast).

## Read this first

- **Trained on one generator.** All training images are GenImage Stable
  Diffusion v1.4 against ImageNet photos. Near-perfect on the Stable
  Diffusion family; weaker on other generators, especially ADM, and much
  weaker on 2024-25 generators (AIGIBench, last table).
- **More robust than the fast tier, not more accurate than strong public
  detectors.** It holds up much better after JPEG, resizing and blur (tables
  below), but the public Community Forensics model, trained on thousands of
  generators, scores clearly higher on the same test images.
- **Do not trust the fixed 0.5 cut on new data.** In a spot check, two of
  three DALL-E 2 samples scored as real and one of three real Unsplash
  photographs scored 0.997. Set your own threshold on real photos from your
  setting with `fakeradar calibrate-fpr` (a bounded rate of real photos
  flagged; method audited in the repository's Gate B report).
- **A probabilistic signal, not proof.** Never use it as sole evidence about
  a person or in legal, journalistic or academic-integrity decisions.

## Use

```bash
pip install "fakeradar[semantic]"   # from the repository until a PyPI release exists
fakeradar scan photo.jpg --tier accurate --pretrained accurate-v0
fakeradar calibrate-fpr real_photos.csv --pretrained accurate-v0 --alpha 0.01
```

Or download `accurate_v0.pt` and pass `--checkpoint accurate_v0.pt`. Loading
builds the CLIP backbone with `timm`, which downloads its pretrained weights
once before they are replaced by the checkpoint's.

## Training

Same as fast-v0: GenImage Stable Diffusion v1.4 (Kaggle copy
`vtphatt2/genimage-stable-diffusion-v1-4`, CC-BY-NC-SA-4.0), 95,000 images
plus 5,000 held out, compression history equalized (non-JPEG images
JPEG-encoded once at quality 96), 5 epochs, one Kaggle T4. The CLIP backbone
stays frozen. Seed 42 is the released checkpoint; results below cover seeds
42, 43 and 44.

## Licence

Weights: CC-BY-NC-SA-4.0 (non-commercial, share-alike), inherited from the
training data. Code: Apache-2.0. CLIP weights: OpenAI CLIP, via `timm`.

"""


def card(root: Path = ROOT) -> str:
    acc = available(ACCURATE, root)
    fast = available(MAIN, root)
    base = available(BASELINE, root)
    a_eq = [benchmark(p / "benchmark_equalized.md") for p in acc.values()]
    f_eq = [benchmark(p / "benchmark_equalized.md") for p in fast.values()]
    b_eq = [benchmark(p / "benchmark_equalized.md") for p in base.values()]
    a_rob = [robustness(p / "robustness_equalized.md") for p in acc.values()]
    f_rob = [robustness(p / "robustness_equalized.md") for p in fast.values()]
    b_rob = [robustness(p / "robustness_equalized.md") for p in base.values()]
    lines = [
        "## Measured results",
        "",
        f"AUROC, mean ± sample std over seeds (accurate: {len(acc)}, fast: {len(fast)}).",
        "Test: GenImage `val` split, 1,000 generated images per generator against 2,000",
        "real; robustness on 500 real + 500 generated. Community Forensics is the public",
        "model re-evaluated on the same images, not retrained (not a same-data comparison).",
        "",
        "| Test generator | accurate-v0 | fast-v0 | Community Forensics |",
        "|---|---|---|---|",
    ]
    for g in (*GENERATORS, "MEAN"):
        name = "**Mean of 8**" if g == "MEAN" else g
        lines.append(
            f"| {name} | {fmt([d[g]['auroc'] for d in a_eq])} | "
            f"{fmt([d[g]['auroc'] for d in f_eq])} | {fmt([d[g]['auroc'] for d in b_eq])} |"
        )
    lines += [
        "",
        "| Condition | accurate-v0 | fast-v0 | Community Forensics |",
        "|---|---|---|---|",
    ]
    for c in CONDITIONS:
        lines.append(
            f"| {c} | {fmt([r[c] for r in a_rob])} | {fmt([r[c] for r in f_rob])} | "
            f"{fmt([r[c] for r in b_rob])} |"
        )
    return HEADER + "\n".join(lines) + "\n" + card_section("accurate") + FOOTER


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(card().encode("utf-8"))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
