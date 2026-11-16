"""Score every FODB photo with the reference detector, on this machine.

Reads the three FODB archives directly (nothing is extracted or uploaded) and
appends one JSON line per photo: phone, platform, scene, role, score (the
detector's logit; higher means more likely generated), width, height.
Re-running resumes where it stopped.

    python score_fodb_local.py ../../data/fodb ../../runs/fpr_guarantee/fodb_records.jsonl
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lens_law"))
sys.path.insert(0, str(HERE.parents[1] / "data" / "_tools" / "commfor"))

from fprcal.prep import commfor_preprocess, parse_fodb_name  # noqa: E402
from lenslaw.splits import assign_split  # noqa: E402

BATCH = 16
MODEL_ID = "OwensLab/commfor-model-384"


def role_of(phone: str) -> str:
    """60% of phones calibrate, 40% audit; fixed by a hash of the phone id."""
    return "cal" if assign_split(phone, salt="fodb-phones") == "dev" else "audit"


def decode(item: tuple[str, bytes]):
    """Decode one photo. A file that cannot be decoded returns (name, None, None)."""
    name, data = item
    try:
        with Image.open(io.BytesIO(data)) as im:
            size = im.size
            return name, size, commfor_preprocess(im)
    except OSError:
        return name, None, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("archives", type=Path, help="folder holding part01.zip .. part03.zip")
    parser.add_argument("out", type=Path, help="records.jsonl to append to")
    parser.add_argument("--limit", type=int, default=0, help="score at most this many (0 = all)")
    parser.add_argument("--pattern", default="part*.zip", help="which archives to read")
    args = parser.parse_args(argv)

    from models import ViTClassifier  # the reference detector's own model class

    model = ViTClassifier.from_pretrained(MODEL_ID, device="cpu").eval()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        with open(args.out, encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    skipped, written, t0 = 0, 0, time.time()

    bad_path = args.out.with_suffix(".unreadable.txt")
    if bad_path.exists():
        done |= set(bad_path.read_text(encoding="utf-8").splitlines())
    with (
        open(args.out, "a", encoding="utf-8") as out,
        open(bad_path, "a", encoding="utf-8") as unreadable,
        ThreadPoolExecutor(4) as pool,
    ):

        def flush(batch: list[tuple[str, bytes]]) -> None:
            nonlocal written
            decoded = list(pool.map(decode, batch))
            for name, size, _ in decoded:
                if size is None:  # listed, never silently dropped
                    unreadable.write(name + chr(10))
                    unreadable.flush()
            decoded = [d for d in decoded if d[1] is not None]
            if not decoded:
                written += len(batch)
                return
            with torch.no_grad():
                logits = model(torch.stack([t for _, _, t in decoded])).flatten().tolist()
            for (name, size, _), score in zip(decoded, logits, strict=True):
                phone, platform, scene = parse_fodb_name(name)
                rec = {
                    "id": name,
                    "phone": phone,
                    "platform": platform,
                    "scene": scene,
                    "role": role_of(phone),
                    "score": score,
                    "width": size[0],
                    "height": size[1],
                }
                out.write(json.dumps(rec) + "\n")
            out.flush()
            written += len(batch)
            if written % 480 == 0:
                rate = written / (time.time() - t0)
                print(f"{written} scored, {rate:.1f} photos/s", flush=True)

        batch: list[tuple[str, bytes]] = []
        for archive in sorted(args.archives.glob(args.pattern)):
            with zipfile.ZipFile(archive) as zf:
                for info in zf.infolist():
                    if info.is_dir() or parse_fodb_name(info.filename) is None:
                        skipped += not info.is_dir()
                        continue
                    if info.filename in done:
                        continue
                    batch.append((info.filename, zf.read(info)))
                    if len(batch) == BATCH:
                        flush(batch)
                        batch = []
                    if args.limit and written + len(batch) >= args.limit:
                        break
            if args.limit and written + len(batch) >= args.limit:
                break
        if batch:
            flush(batch)

    print(f"done: {written} new, {len(done)} already there, {skipped} non-photo files skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
