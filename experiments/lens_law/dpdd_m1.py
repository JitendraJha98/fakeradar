"""Gate M check M1 on DPDD aperture pairs (runs locally, reads the zip directly).

    python dpdd_m1.py <path to dd_dp_dataset_canon.zip> ../../runs/lens_law/dpdd_m1.jsonl

For each development-split scene (assign_split on the file name): blur from the
wide-aperture shot (`*_c/source`), inverse depth from its all-in-focus twin
(`*_c/target`) and from the wide shot itself, with both blur estimators. M1
counts a pair as fitting when the thin-lens fit removes at least half the blur
variance (r2_lens >= 0.5), among pairs whose wide shot is applicable.
This completes the Gate M report; direction A stays stopped regardless.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lenslaw.blur_net import BlurNet  # noqa: E402
from lenslaw.depth import inverse_depth, load_depth  # noqa: E402
from lenslaw.score import load_rgb, score_arrays  # noqa: E402
from lenslaw.splits import assign_split  # noqa: E402

WEIGHTS = HERE.parents[1] / "runs" / "lens_law" / "phase1-train-blur" / "blur_net.pt"


def to_pil(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("zip", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args(argv)

    zf = zipfile.ZipFile(args.zip)
    # The sharp twin is a separate exposure with its own frame number, so pairs
    # are matched by sorted order within each split folder (counts must agree).
    twin = {}
    for split in ("train", "val", "test"):
        src = sorted(
            n for n in zf.namelist() if f"/{split}_c/source/" in n and n.lower().endswith(".png")
        )
        tgt = sorted(
            n for n in zf.namelist() if f"/{split}_c/target/" in n and n.lower().endswith(".png")
        )
        if len(src) != len(tgt):
            raise ValueError(f"{split}: {len(src)} wide shots but {len(tgt)} sharp shots")
        twin.update(zip(src, tgt, strict=True))
    dev = [n for n in sorted(twin) if assign_split(Path(n).stem, salt="dpdd") == "dev"]
    net = BlurNet()
    net.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    net.eval()
    depth = load_depth("cpu")

    done = set()
    if args.out.exists():
        done = {
            json.loads(line)["scene"]
            for line in args.out.read_text(encoding="utf-8").splitlines()
            if line
        }
    with open(args.out, "a", encoding="utf-8") as out:
        for i, name in enumerate(dev):
            scene = Path(name).stem
            if scene in done:
                continue
            wide = load_rgb(Image.open(io.BytesIO(zf.read(name))))
            sharp = load_rgb(Image.open(io.BytesIO(zf.read(twin[name]))))
            d_sharp, d_wide = (
                inverse_depth(depth, to_pil(sharp)),
                inverse_depth(depth, to_pil(wide)),
            )
            for estimator, model in (("v0", None), ("v1", net)):
                for variant, rgb, d in (
                    ("wide_depth_narrow", wide, d_sharp),
                    ("wide_depth_self", wide, d_wide),
                    ("narrow_depth_self", sharp, d_sharp),
                ):
                    rec = {"scene": scene, "estimator": estimator, "variant": variant}
                    rec.update(score_arrays(rgb, d, blur_model=model))
                    out.write(json.dumps(rec) + "\n")
            out.flush()
            print(f"{i + 1}/{len(dev)} {scene}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
