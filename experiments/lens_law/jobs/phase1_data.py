"""Kaggle entry (CPU, internet): fetch the Unsplash photos Phase 1 needs.

Two sets, chosen by EXIF aperture and a hash of the photo id (never by eye):
  wide  - f/2.8 or wider: real photos where depth of field may be visible
  sharp - f/8 or narrower: all-in-focus photos used for the controlled renders

Images stay inside this private notebook's output. The manifest holds photo
ids and the split only; photographer names are used for the split and dropped.
"""

import concurrent.futures
import csv
import hashlib
import io
import json
import os
import urllib.request
import zipfile

from lenslaw.splits import assign_split
from PIL import Image

N_WIDE, N_SHARP = 1000, 400
OUT = "/kaggle/working"
IMG_DIR = os.path.join(OUT, "unsplash")
os.makedirs(IMG_DIR, exist_ok=True)


def fetch(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def aperture(row) -> float | None:
    try:
        value = float(str(row.get("exif_aperture_value", "")).lower().replace("f/", ""))
    except ValueError:
        return None
    return value if 0.7 <= value <= 64 else None


zf = zipfile.ZipFile(io.BytesIO(fetch("https://unsplash.com/data/lite/latest", timeout=900)))
name = next(n for n in zf.namelist() if os.path.basename(n).startswith("photos."))
with zf.open(name) as f:
    rows = list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"), delimiter="\t"))


def by_hash(row) -> str:
    return hashlib.sha256(row["photo_id"].encode()).hexdigest()


wide = sorted((r for r in rows if (aperture(r) or 99) <= 2.8), key=by_hash)[:N_WIDE]
sharp = sorted((r for r in rows if (aperture(r) or 0) >= 8), key=by_hash)[:N_SHARP]
todo = [("wide", r) for r in wide] + [("sharp", r) for r in sharp]


def download(item):
    kind, row = item
    path = os.path.join(IMG_DIR, row["photo_id"] + ".jpg")
    for _ in range(3):
        try:
            data = fetch(row["photo_image_url"] + "?w=1536&q=90&fm=jpg")
            with Image.open(io.BytesIO(data)) as im:
                size = im.size
            with open(path, "wb") as f:
                f.write(data)
            return kind, row, size, None
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
    return kind, row, None, err


with concurrent.futures.ThreadPoolExecutor(16) as pool:
    results = list(pool.map(download, todo))

failures = [(r["photo_id"], err) for _, r, size, err in results if size is None]
with open(os.path.join(OUT, "unsplash_manifest.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["photo_id", "set", "split", "aperture", "focal_length", "width", "height"])
    for kind, row, size, _ in results:
        if size is None:
            continue
        w.writerow(
            [
                row["photo_id"],
                kind,
                assign_split(row["photographer_username"]),
                aperture(row),
                row.get("exif_focal_length", ""),
                size[0],
                size[1],
            ]
        )

report = {
    "source": "Unsplash Lite dataset (https://unsplash.com/data), images not redistributed",
    "rows_in_dataset": len(rows),
    "wide_requested": len(wide),
    "sharp_requested": len(sharp),
    "downloaded": sum(size is not None for _, _, size, _ in results),
    "failures": failures[:20],
    "n_failures": len(failures),
}
print(json.dumps(report, indent=1))
with open(os.path.join(OUT, "data_report.json"), "w") as f:
    json.dump(report, f, indent=1)
