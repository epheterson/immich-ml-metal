import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "models" / "face_embed.py"
SPEC = importlib.util.spec_from_file_location("face_embed_test_module", MODULE_PATH)
face_embed = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(face_embed)


def _set_home(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(face_embed.Path, "home", classmethod(lambda cls: tmp_path))


def _fake_find_model(model_dir: Path):
    model_file = model_dir / "model.onnx"
    return model_file if model_file.exists() else None


def test_uses_existing_valid_model_pack(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)

    model_dir = tmp_path / ".insightface" / "models" / "buffalo_l"
    model_dir.mkdir(parents=True)
    model_file = model_dir / "model.onnx"
    model_file.touch()

    monkeypatch.setattr(face_embed, "_find_recognition_model", _fake_find_model)

    def should_not_download(*args, **kwargs):
        raise AssertionError("unexpected download")

    result = face_embed._ensure_recognition_model_pack("buffalo_l", should_not_download)

    assert result == model_file


def test_recovers_from_invalid_pack_with_leftover_onnx(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)

    model_dir = tmp_path / ".insightface" / "models" / "buffalo_l"
    model_dir.mkdir(parents=True)
    stale_file = model_dir / "stale.onnx"
    stale_file.write_text("incomplete")

    monkeypatch.setattr(face_embed, "_find_recognition_model", _fake_find_model)

    def fake_download(sub_dir, name, force, root):
        assert sub_dir == "models"
        assert name == "buffalo_l"
        assert force is True
        assert root == str(tmp_path / ".insightface")
        assert stale_file.exists()
        (model_dir / "model.onnx").touch()

    result = face_embed._ensure_recognition_model_pack("buffalo_l", fake_download)

    assert result == model_dir / "model.onnx"
    assert result.exists()


def test_downloads_when_model_pack_missing(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)

    model_dir = tmp_path / ".insightface" / "models" / "buffalo_l"
    monkeypatch.setattr(face_embed, "_find_recognition_model", _fake_find_model)

    def fake_download(sub_dir, name, force, root):
        assert sub_dir == "models"
        assert name == "buffalo_l"
        assert force is False
        assert root == str(tmp_path / ".insightface")
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "model.onnx").touch()

    result = face_embed._ensure_recognition_model_pack("buffalo_l", fake_download)

    assert result == model_dir / "model.onnx"


def test_raises_if_redownload_still_missing_model(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)

    model_dir = tmp_path / ".insightface" / "models" / "buffalo_l"
    model_dir.mkdir(parents=True)

    monkeypatch.setattr(face_embed, "_find_recognition_model", lambda _: None)

    def fake_download(sub_dir, name, force, root):
        assert force is True
        model_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError):
        face_embed._ensure_recognition_model_pack("buffalo_l", fake_download)
