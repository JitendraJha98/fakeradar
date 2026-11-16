"""Score generated images with the reference detector, clean and after transport.

Reads the images pulled from the private generation notebook and appends one
JSON line per (image, condition): generator, photo_id, condition, score.
Re-running resumes where it stopped.

    python score_generated_local.py ../../runs/lens_law/phase3-generate ../../runs/fpr_guarantee/generated_records.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lens_law"))
sys.path.insert(0, str(HERE.parents[1] / "data" / "_tools" / "commfor"))

from fprcal.prep import commfor_preprocess  # noqa: E402
from lenslaw.transport import CONDITIONS, apply_condition  # noqa: E402

MODEL_ID = "OwensLab/commfor-model-384"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "root", type=Path, help="folder with generated_manifest.json and generated/"
    )
    parser.add_argument("out", type=Path)
    args = parser.parse_args(argv)

    from models import ViTClassifier

    model = ViTClassifier.from_pretrained(MODEL_ID, device="cpu").eval()
    manifest = json.loads((args.root / "generated_manifest.json").read_text(encoding="utf-8"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        with open(args.out, encoding="utf-8") as f:
            done = {(r["generator"], r["photo_id"]) for r in map(json.loads, filter(str.strip, f))}
    written = 0
    with open(args.out, "a", encoding="utf-8") as out:
        for item in manifest:
            key = (item["generator"], item["photo_id"])
            path = args.root / item["file"]
            if key in done or not path.exists():
                continue
            img = Image.open(path).convert("RGB")
            batch = torch.stack([commfor_preprocess(apply_condition(img, c)) for c in CONDITIONS])
            with torch.no_grad():
                scores = model(batch).flatten().tolist()
            for cond, score in zip(CONDITIONS, scores, strict=True):
                rec = {"generator": key[0], "photo_id": key[1], "condition": cond, "score": score}
                out.write(json.dumps(rec) + "\n")
            out.flush()
            written += 1
            if written % 50 == 0:
                print(f"{written} images scored", flush=True)
    print(f"done: {written} new images, {len(done)} already there")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
