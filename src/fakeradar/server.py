"""Local REST API. Designed for localhost / trusted networks (no auth).

NOTE: no `from __future__ import annotations` here — stringified annotations
break FastAPI's resolution of the function-local `UploadFile` import on
Python 3.14 (PEP 649). PEP 604 unions work natively on py3.10+.
"""

import io
from pathlib import Path


def create_app(tier: str = "fast", checkpoint: Path | None = None, untrained_ok: bool = False):
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from PIL import Image, UnidentifiedImageError

    from . import __version__
    from .detector import Detector

    det = Detector(
        tier=tier,
        checkpoint=checkpoint,
        allow_random_init=untrained_ok and checkpoint is None,
    )
    app = FastAPI(title="fakeradar", version=__version__)

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "tier": det.config.tier, "trained": det.trained}

    @app.post("/v1/detect")
    async def detect(file: UploadFile = File(...)):
        payload = await file.read()
        try:
            img = Image.open(io.BytesIO(payload)).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            raise HTTPException(
                status_code=400, detail="Uploaded file is not a decodable image."
            ) from e
        result = det.predict(img).to_dict()
        result["source"] = file.filename
        return result

    return app
