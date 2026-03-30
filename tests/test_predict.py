"""Tests for the /predict endpoint — concurrent task execution, response format."""
import json

import pytest
import httpx

from src.main import app


@pytest.fixture
def client():
    """Async test client against the FastAPI app (no real server needed)."""
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def _entries(*task_types):
    """Build an entries JSON for the given task types."""
    entries = {}
    if "clip" in task_types:
        entries["clip"] = {"visual": {"modelName": "ViT-B-32__openai"}}
    if "clip-text" in task_types:
        entries["clip"] = {"textual": {"modelName": "ViT-B-32__openai"}}
    if "facial-recognition" in task_types:
        entries["facial-recognition"] = {"detection": {}, "recognition": {}}
    if "ocr" in task_types:
        entries["ocr"] = {"detection": {}, "recognition": {}}
    return json.dumps(entries)


# --- Basic endpoint tests ---

@pytest.mark.asyncio
async def test_ping(client):
    resp = await client.get("/ping")
    assert resp.status_code == 200
    assert resp.text == "pong"


@pytest.mark.asyncio
async def test_root(client):
    resp = await client.get("/")
    assert resp.status_code == 200
    assert resp.json()["message"] == "Immich ML"


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["stub_mode"] is True


# --- Predict: single tasks ---

@pytest.mark.asyncio
async def test_predict_clip_visual(client, test_image_bytes):
    resp = await client.post(
        "/predict",
        data={"entries": _entries("clip")},
        files={"image": ("test.jpg", test_image_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "clip" in data
    embedding = json.loads(data["clip"])
    assert len(embedding) == 512
    assert "imageHeight" in data
    assert "imageWidth" in data


@pytest.mark.asyncio
async def test_predict_clip_text(client):
    resp = await client.post(
        "/predict",
        data={"entries": _entries("clip-text"), "text": "a photo of a cat"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "clip" in data
    embedding = json.loads(data["clip"])
    assert len(embedding) == 512


@pytest.mark.asyncio
async def test_predict_faces(client, test_image_bytes):
    resp = await client.post(
        "/predict",
        data={"entries": _entries("facial-recognition")},
        files={"image": ("test.jpg", test_image_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "facial-recognition" in data
    faces = data["facial-recognition"]
    assert isinstance(faces, list)
    # Stub mode returns 1 fake face
    assert len(faces) == 1
    assert "boundingBox" in faces[0]
    assert "embedding" in faces[0]
    assert "score" in faces[0]


@pytest.mark.asyncio
async def test_predict_ocr(client, test_image_bytes):
    resp = await client.post(
        "/predict",
        data={"entries": _entries("ocr")},
        files={"image": ("test.jpg", test_image_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "ocr" in data
    ocr = data["ocr"]
    assert "text" in ocr
    assert "box" in ocr
    assert "boxScore" in ocr
    assert "textScore" in ocr


# --- Predict: concurrent tasks ---

@pytest.mark.asyncio
async def test_predict_all_three_tasks(client, test_image_bytes):
    """All 3 tasks in one request — tests asyncio.gather path."""
    entries = json.dumps({
        "clip": {"visual": {"modelName": "ViT-B-32__openai"}},
        "facial-recognition": {"detection": {}, "recognition": {}},
        "ocr": {"detection": {}, "recognition": {}},
    })
    resp = await client.post(
        "/predict",
        data={"entries": entries},
        files={"image": ("test.jpg", test_image_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "clip" in data
    assert "facial-recognition" in data
    assert "ocr" in data
    assert "imageHeight" in data
    assert "imageWidth" in data


# --- Error handling ---

@pytest.mark.asyncio
async def test_predict_no_image_or_text(client):
    resp = await client.post("/predict", data={"entries": _entries("clip")})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_predict_invalid_json(client, test_image_bytes):
    resp = await client.post(
        "/predict",
        data={"entries": "not json"},
        files={"image": ("test.jpg", test_image_bytes, "image/jpeg")},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_predict_empty_tasks(client, test_image_bytes):
    """Empty entries dict — should return 200 with just image dimensions."""
    resp = await client.post(
        "/predict",
        data={"entries": "{}"},
        files={"image": ("test.jpg", test_image_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "imageHeight" in data
    assert "clip" not in data
