"""Kaggle entry (GPU, internet): generated images for the detection-rate step.

Prompts are the text descriptions of real Unsplash photos already used in
this project, so content is matched to the real photos. Three open-weight
generators are tried in turn; one failing does not stop the others. Images
are saved as PNG in this private notebook's output with a manifest.
"""

import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import time
import urllib.request
import zipfile

N_PROMPTS = 200
SIZE = 1024
OUT = "/kaggle/working"
report = {"generators": {}}

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "-U",
        "git+https://github.com/huggingface/diffusers",
    ],
    check=False,
)
import torch  # noqa: E402

# ---- prompts: descriptions of the wide-aperture photos in the phase1-data manifest ----
ids = []
for base, _dirs, files in os.walk("/kaggle/input"):
    if "unsplash_manifest.csv" in files:
        with open(os.path.join(base, "unsplash_manifest.csv"), encoding="utf-8") as f:
            ids = [r["photo_id"] for r in csv.DictReader(f) if r["set"] == "wide"]
req = urllib.request.Request(
    "https://unsplash.com/data/lite/latest", headers={"User-Agent": "Mozilla/5.0"}
)
with urllib.request.urlopen(req, timeout=900) as r:
    zf = zipfile.ZipFile(io.BytesIO(r.read()))
name = next(n for n in zf.namelist() if os.path.basename(n).startswith("photos."))
with zf.open(name) as f:
    rows = {
        r["photo_id"]: r
        for r in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"), delimiter="\t")
    }
prompts = []
for pid in sorted(ids, key=lambda p: hashlib.sha256(p.encode()).hexdigest()):
    text = (
        rows.get(pid, {}).get("ai_description") or rows.get(pid, {}).get("photo_description") or ""
    ).strip()
    if len(text) >= 12:
        prompts.append((pid, f"A photograph of {text}"))
    if len(prompts) == N_PROMPTS:
        break
report["n_prompts"] = len(prompts)
print("prompts:", len(prompts), prompts[:2], flush=True)


def load(kind: str):
    if kind == "sdxl":
        from diffusers import DiffusionPipeline

        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        return pipe, {"num_inference_steps": 30}
    if kind == "zimage":
        from diffusers import ZImagePipeline

        pipe = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo", torch_dtype=torch.bfloat16
        )
        return pipe, {"num_inference_steps": 9, "guidance_scale": 0.0}
    from diffusers import Flux2KleinPipeline

    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-4B", torch_dtype=torch.bfloat16
    )
    return pipe, {"num_inference_steps": 4, "guidance_scale": 1.0}


manifest = []
for kind in ("sdxl", "klein", "zimage"):
    entry = {"made": 0, "seconds": 0}
    report["generators"][kind] = entry
    try:
        pipe, kwargs = load(kind)
        pipe.enable_model_cpu_offload()
        os.makedirs(os.path.join(OUT, "generated", kind), exist_ok=True)
        t0 = time.time()
        for i, (pid, prompt) in enumerate(prompts):
            seed = int(hashlib.sha256(f"{kind}/{pid}".encode()).hexdigest()[:8], 16)
            image = pipe(
                prompt=prompt,
                height=SIZE,
                width=SIZE,
                generator=torch.Generator("cpu").manual_seed(seed),
                **kwargs,
            ).images[0]
            if max(image.convert("L").getextrema()) < 8:  # an all-black image means the run broke
                entry["black_images"] = entry.get("black_images", 0) + 1
                continue
            path = os.path.join("generated", kind, f"{pid}.png")
            image.save(os.path.join(OUT, path))
            manifest.append(
                {"generator": kind, "photo_id": pid, "file": path, "prompt": prompt, "seed": seed}
            )
            entry["made"] += 1
            if i % 25 == 0:
                print(f"{kind} {i}/{len(prompts)} {int(time.time() - t0)}s", flush=True)
            if i == 9 and time.time() - t0 > 900:  # slower than 90 s per image: not worth the quota
                entry["stopped"] = "too slow"
                break
        entry["seconds"] = int(time.time() - t0)
        del pipe
        torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001
        entry["error"] = f"{type(e).__name__}: {e}"[:500]
    print(kind, entry, flush=True)
    with open(os.path.join(OUT, "generated_manifest.json"), "w") as f:
        json.dump(manifest, f)
    with open(os.path.join(OUT, "gen_report.json"), "w") as f:
        json.dump(report, f, indent=1)
print(json.dumps(report, indent=1))
