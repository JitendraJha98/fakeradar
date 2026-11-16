"""Kaggle entry (CPU, internet): sharp photos for training blur estimator v1.

Only development-split photographers, only photos shot at f/8 or narrower,
and none of the photos already used as render sources. A tenth of the
photographers are held out for validation. Images stay in this private
notebook's output; the manifest holds photo ids only.
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

N_TRAIN = 1500
OUT = "/kaggle/working"
IMG_DIR = os.path.join(OUT, "unsplash_train")
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


used = set()
for base, _dirs, files in os.walk("/kaggle/input"):
    if "unsplash_manifest.csv" in files:
        with open(os.path.join(base, "unsplash_manifest.csv"), encoding="utf-8") as f:
            used = {r["photo_id"] for r in csv.DictReader(f)}
assert used, "the phase1-data manifest must be attached so its photos can be excluded"

zf = zipfile.ZipFile(io.BytesIO(fetch("https://unsplash.com/data/lite/latest", timeout=900)))
name = next(n for n in zf.namelist() if os.path.basename(n).startswith("photos."))
with zf.open(name) as f:
    rows = list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"), delimiter="\t"))

pool_rows = [
    r
    for r in rows
    if (aperture(r) or 0) >= 8
    and r["photo_id"] not in used
    and assign_split(r["photographer_username"]) == "dev"
]
pool_rows.sort(key=lambda r: hashlib.sha256(r["photo_id"].encode()).hexdigest())
todo = pool_rows[:N_TRAIN]


def download(row):
    path = os.path.join(IMG_DIR, row["photo_id"] + ".jpg")
    for _ in range(3):
        try:
            data = fetch(row["photo_image_url"] + "?w=1536&q=90&fm=jpg")
            with Image.open(io.BytesIO(data)) as im:
                im.verify()
            with open(path, "wb") as f:
                f.write(data)
            return row, True
        except Exception:  # noqa: BLE001
            pass
    return row, False


with concurrent.futures.ThreadPoolExecutor(16) as pool:
    results = list(pool.map(download, todo))

with open(os.path.join(OUT, "train_manifest.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["photo_id", "role"])
    for row, ok in results:
        if ok:
            held_out = assign_split(
                row["photographer_username"], salt="blurnet-val", dev_fraction=0.9
            )
            w.writerow([row["photo_id"], "train" if held_out == "dev" else "val"])

report = {
    "eligible": len(pool_rows),
    "requested": len(todo),
    "downloaded": sum(ok for _, ok in results),
    "excluded_render_or_real_photos": len(used),
}
print(json.dumps(report, indent=1))
with open(os.path.join(OUT, "train_data_report.json"), "w") as f:
    json.dump(report, f, indent=1)
