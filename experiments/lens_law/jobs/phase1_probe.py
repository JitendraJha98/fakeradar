"""Kaggle entry (CPU, internet): look at the candidate Phase 1 data sources.

Writes probe_report.json. Downloads nothing large and keeps no images.
"""

import collections
import csv
import io
import json
import os
import urllib.request
import zipfile

from PIL import Image

report = {}

# ---- iDFD (attached Kaggle dataset): folder tree, counts, image sizes ----
root = None
for base, dirs, _files in os.walk("/kaggle/input"):
    if "Data" in dirs:
        root = os.path.join(base, "Data")
        break
tree = collections.OrderedDict()
samples = {}
if root:
    for base, _dirs, files in os.walk(root):
        rel = os.path.relpath(base, root)
        if files:
            exts = collections.Counter(os.path.splitext(f)[1].lower() for f in files)
            tree[rel] = {"n": len(files), "ext": dict(exts), "first": sorted(files)[:3]}
            first = os.path.join(base, sorted(files)[0])
            try:
                with Image.open(first) as im:
                    samples[rel] = {"size": im.size, "mode": im.mode}
            except Exception as e:  # noqa: BLE001
                samples[rel] = f"not an image: {type(e).__name__}"
report["idfd_root"] = root
report["idfd_tree"] = tree
report["idfd_samples"] = samples
for name in ("README.md", "README.txt", "readme.txt", "LICENSE", "license.txt"):
    for base, _dirs, files in os.walk("/kaggle/input"):
        if name in files:
            with open(os.path.join(base, name), encoding="utf-8", errors="replace") as f:
                report[f"idfd_{name}"] = f.read()[:3000]

# ---- DPDD: does the authors' link serve a file to a script? ----
dpdd = "https://ln2.sync.com/dl/c45358c50/r7kpybwk-xw8hhszh-qkj249ap-y8k2344d"
try:
    req = urllib.request.Request(dpdd, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        head = r.read(400)
        report["dpdd_link"] = {
            "status": r.status,
            "content_type": r.headers.get("Content-Type"),
            "content_length": r.headers.get("Content-Length"),
            "looks_like_html": b"<html" in head.lower() or b"<!doctype" in head.lower(),
        }
except Exception as e:  # noqa: BLE001
    report["dpdd_link"] = f"error: {type(e).__name__}: {e}"

# ---- Unsplash Lite: archive contents, columns, how many wide-aperture photos ----
try:
    req = urllib.request.Request(
        "https://unsplash.com/data/lite/latest", headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        blob = r.read()
    report["unsplash_zip_mb"] = round(len(blob) / 1e6, 1)
    zf = zipfile.ZipFile(io.BytesIO(blob))
    report["unsplash_files"] = {i.filename: i.file_size for i in zf.infolist()}
    photos = next(n for n in zf.namelist() if os.path.basename(n).startswith("photos."))
    delim = "\t" if ".tsv" in photos else ","
    with zf.open(photos) as f:
        rows = list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"), delimiter=delim))
    report["unsplash_columns"] = list(rows[0].keys())
    report["unsplash_rows"] = len(rows)

    def aperture(row):
        try:
            return float(str(row.get("exif_aperture_value", "")).lower().replace("f/", ""))
        except ValueError:
            return None

    values = [aperture(r) for r in rows]
    known = [v for v in values if v is not None and 0.7 <= v <= 64]
    report["unsplash_aperture_known"] = len(known)
    report["unsplash_aperture_le_2_8"] = sum(v <= 2.8 for v in known)
    report["unsplash_aperture_ge_8"] = sum(v >= 8 for v in known)
    wide = [r for r, v in zip(rows, values, strict=False) if v is not None and 0.7 <= v <= 2.8]
    report["unsplash_wide_photographers"] = len({r["photographer_username"] for r in wide})
    report["unsplash_wide_with_description"] = sum(
        bool(r.get("ai_description") or r.get("photo_description")) for r in wide
    )
    ex = wide[0]
    report["unsplash_example"] = {
        k: ex.get(k)
        for k in (
            "photo_image_url",
            "photo_width",
            "photo_height",
            "exif_aperture_value",
            "exif_focal_length",
            "ai_description",
        )
    }
    url = ex["photo_image_url"] + "?w=1536&q=90&fm=jpg"
    with urllib.request.urlopen(urllib.request.Request(url), timeout=60) as r:
        data = r.read()
    with Image.open(io.BytesIO(data)) as im:
        report["unsplash_fetch"] = {"bytes": len(data), "size": im.size, "format": im.format}
except Exception as e:  # noqa: BLE001
    report["unsplash_error"] = f"{type(e).__name__}: {e}"

print(json.dumps(report, indent=1)[:6000])
with open("/kaggle/working/probe_report.json", "w") as f:
    json.dump(report, f, indent=1)
