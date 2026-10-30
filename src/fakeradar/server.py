"""Local REST API. Designed for localhost / trusted networks (no auth).

NOTE: no `from __future__ import annotations` here — stringified annotations
break FastAPI's resolution of the function-local `UploadFile` import on
Python 3.14 (PEP 649). PEP 604 unions work natively on py3.10+.
"""

import io
import threading
from pathlib import Path

DEFAULT_MAX_UPLOAD_MB = 25


def create_app(
    tier: str = "fast",
    checkpoint: Path | None = None,
    untrained_ok: bool = False,
    pretrained: str | None = None,
    device: str = "auto",
    max_upload_mb: float = DEFAULT_MAX_UPLOAD_MB,
):
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from PIL import Image, UnidentifiedImageError

    from . import __version__
    from .detector import Detector

    det = Detector(
        tier=tier,
        checkpoint=checkpoint,
        pretrained=pretrained,
        device=device,
        allow_random_init=untrained_ok and checkpoint is None and pretrained is None,
    )
    max_bytes = int(max_upload_mb * 1024 * 1024)
    lock = threading.Lock()  # one inference at a time: bounded memory, no shared-cache races
    app = FastAPI(title="fakeradar", version=__version__)

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "tier": det.config.tier, "trained": det.trained}

    # Plain `def`: FastAPI runs it in its threadpool, so a slow CPU inference
    # doesn't block the event loop (and /healthz) for other clients.
    @app.post("/v1/detect")
    def detect(file: UploadFile = File(...)):
        payload = file.file.read(max_bytes + 1)
        if len(payload) > max_bytes:
            raise HTTPException(status_code=413, detail=f"Upload exceeds {max_upload_mb:g} MB.")
        try:
            img = Image.open(io.BytesIO(payload)).convert("RGB")
        except Image.DecompressionBombError as e:
            raise HTTPException(status_code=413, detail="Image has too many pixels.") from e
        except (UnidentifiedImageError, OSError) as e:
            raise HTTPException(
                status_code=400, detail="Uploaded file is not a decodable image."
            ) from e
        with lock:
            result = det.predict(img).to_dict()
        result["source"] = file.filename
        return result

    return app
