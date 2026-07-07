import io

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from fakeradar.server import create_app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(tier="fast", checkpoint=None, untrained_ok=True))


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["tier"] == "fast" and body["trained"] is False


def test_detect_roundtrip(client):
    buf = io.BytesIO()
    Image.effect_noise((300, 300), 40).convert("RGB").save(buf, format="PNG")
    r = client.post("/v1/detect", files={"file": ("photo.png", buf.getvalue(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["prob_ai"] <= 1.0
    assert body["source"] == "photo.png"
    assert body["verdict"] in {"likely_real", "uncertain", "likely_ai"}


def test_detect_rejects_non_image(client):
    r = client.post("/v1/detect", files={"file": ("junk.png", b"not an image", "image/png")})
    assert r.status_code == 400
    assert "not a decodable image" in r.json()["detail"]
