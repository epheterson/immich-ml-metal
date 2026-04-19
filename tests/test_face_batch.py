"""Tests for batched face embedding — alignment, batching, edge cases.

These tests mock the ONNX model so they run without real weights.
"""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.models.face_embed import get_face_embeddings_batch, ARCFACE_INPUT_SIZE


def _fake_img(w=640, h=480):
    """Create a fake BGR image array."""
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def _face_with_bbox(x1, y1, x2, y2, score=0.9):
    return {
        "boundingBox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "score": score,
    }


def _face_with_landmarks(landmarks, score=0.9):
    return {
        "boundingBox": {"x1": 100, "y1": 100, "x2": 200, "y2": 200},
        "landmarks": landmarks,
        "score": score,
    }


@pytest.fixture
def mock_model():
    """Mock the recognition model to return fake 512-dim embeddings."""
    model = MagicMock()

    def fake_get_feat(imgs):
        n = len(imgs) if isinstance(imgs, list) else imgs.shape[0]
        return np.random.randn(n, 512).astype(np.float32)

    model.get_feat = fake_get_feat
    return model


# --- Core batching ---

def test_empty_faces():
    assert get_face_embeddings_batch(_fake_img(), [], "buffalo_l") == []


def test_single_face_bbox(mock_model):
    img = _fake_img()
    faces = [_face_with_bbox(100, 100, 200, 200)]
    with patch("src.models.face_embed.get_recognition_model", return_value=mock_model):
        results = get_face_embeddings_batch(img, faces)
    assert len(results) == 1
    assert results[0] is not None
    assert results[0].shape == (512,)
    assert abs(np.linalg.norm(results[0]) - 1.0) < 1e-5  # normalized


def test_multiple_faces_bbox(mock_model):
    img = _fake_img()
    faces = [
        _face_with_bbox(10, 10, 100, 100),
        _face_with_bbox(200, 50, 350, 250),
        _face_with_bbox(400, 100, 550, 300),
    ]
    with patch("src.models.face_embed.get_recognition_model", return_value=mock_model):
        results = get_face_embeddings_batch(img, faces)
    assert len(results) == 3
    assert all(r is not None for r in results)
    assert all(r.shape == (512,) for r in results)


def test_result_order_preserved(mock_model):
    """Results must be in the same order as input faces."""
    img = _fake_img()
    faces = [
        _face_with_bbox(10, 10, 50, 50),
        _face_with_bbox(200, 200, 400, 400),
    ]
    with patch("src.models.face_embed.get_recognition_model", return_value=mock_model):
        results = get_face_embeddings_batch(img, faces)
    assert len(results) == 2
    # Each face should get a different embedding (random, so extremely unlikely to match)
    assert not np.array_equal(results[0], results[1])


# --- Edge cases ---

def test_empty_crop_returns_none(mock_model):
    """Face with zero-area bbox should return None, not crash."""
    img = _fake_img()
    faces = [_face_with_bbox(100, 100, 100, 100)]  # zero-width
    with patch("src.models.face_embed.get_recognition_model", return_value=mock_model):
        results = get_face_embeddings_batch(img, faces)
    assert len(results) == 1
    assert results[0] is None


def test_mixed_success_and_failure(mock_model):
    """One good face, one bad — good gets embedding, bad gets None."""
    img = _fake_img()
    faces = [
        _face_with_bbox(50, 50, 200, 200),   # valid
        _face_with_bbox(100, 100, 100, 100),  # zero-area
        _face_with_bbox(300, 100, 500, 300),  # valid
    ]
    with patch("src.models.face_embed.get_recognition_model", return_value=mock_model):
        results = get_face_embeddings_batch(img, faces)
    assert len(results) == 3
    assert results[0] is not None  # valid
    assert results[1] is None      # zero-area
    assert results[2] is not None  # valid


def test_all_faces_fail():
    """All faces fail alignment — returns list of Nones, no crash."""
    img = _fake_img(w=10, h=10)
    faces = [
        _face_with_bbox(0, 0, 0, 0),
        _face_with_bbox(5, 5, 5, 5),
    ]
    # Don't even need to mock the model — should never reach inference
    results = get_face_embeddings_batch(img, faces)
    assert len(results) == 2
    assert all(r is None for r in results)


def test_bbox_clamped_to_image(mock_model):
    """Bbox extending beyond image edges should be clamped, not crash."""
    img = _fake_img(w=200, h=200)
    faces = [_face_with_bbox(-50, -50, 250, 250)]  # extends beyond all edges
    with patch("src.models.face_embed.get_recognition_model", return_value=mock_model):
        results = get_face_embeddings_batch(img, faces)
    assert len(results) == 1
    assert results[0] is not None


# --- Embedding normalization ---

def test_embeddings_are_unit_normalized(mock_model):
    img = _fake_img()
    faces = [_face_with_bbox(50, 50, 200, 200)]
    with patch("src.models.face_embed.get_recognition_model", return_value=mock_model):
        results = get_face_embeddings_batch(img, faces)
    norm = np.linalg.norm(results[0])
    assert abs(norm - 1.0) < 1e-5
