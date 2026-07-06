"""Local REST API. Designed for localhost / trusted networks (no auth)."""

from __future__ import annotations

import io
from pathlib import Path


def create_app(tier: str = "fast", checkpoint: Path | None = None, untrained_ok: bool = False):
    from fastapi import FastAPI, File, UploadFile
    from PIL import Image

    from .detector import Detector

    det = Detector(
        tier=tier,
        checkpoint=checkpoint,
        allow_random_init=untrained_ok and checkpoint is None,
    )
    app = FastAPI(title="fakeradar", version="0.1.0")

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "tier": det.config.tier, "trained": det.trained}

    @app.post("/v1/detect")
    async def detect(file: UploadFile = File(...)):
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
        result = det.predict(img).to_dict()
        result["source"] = file.filename
        return result

    return app
