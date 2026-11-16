"""Kaggle entry (CPU): what is in the attached real-photo audit dataset?

Writes probe3_report.json: folder tree with file counts, extensions, a few
names and image sizes per folder, and any readme or licence text. Keeps no
images.
"""

import collections
import json
import os

from PIL import Image

report = {"tree": {}, "texts": {}}
for base, _dirs, files in os.walk("/kaggle/input"):
    if not files:
        continue
    rel = os.path.relpath(base, "/kaggle/input")
    exts = collections.Counter(os.path.splitext(f)[1].lower() for f in files)
    entry = {"n": len(files), "ext": dict(exts), "first": sorted(files)[:4]}
    for name in sorted(files)[:2]:
        try:
            with Image.open(os.path.join(base, name)) as im:
                entry.setdefault("sizes", []).append([im.size, im.format, im.mode])
        except Exception:  # noqa: BLE001
            pass
    report["tree"][rel] = entry
    for name in files:
        if name.lower().startswith(("readme", "license", "licence")) or name.lower().endswith(
            (".txt", ".md")
        ):
            if len(report["texts"]) < 8:
                with open(os.path.join(base, name), encoding="utf-8", errors="replace") as f:
                    report["texts"][os.path.join(rel, name)] = f.read()[:2500]

report["n_folders"] = len(report["tree"])
if len(report["tree"]) > 120:  # keep the report readable: show the first folders and totals
    keys = sorted(report["tree"])
    report["tree"] = {k: report["tree"][k] for k in keys[:120]}
print(json.dumps(report, indent=1)[:5000])
with open("/kaggle/working/probe3_report.json", "w") as f:
    json.dump(report, f, indent=1)
