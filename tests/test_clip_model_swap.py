"""Regression test for the CLIP model-swap race (see problem.log 2026-08-25):
a slow-loading model instance held by one in-flight request must survive a
concurrent get_clip_model() switch to a different model, instead of having
its _model/_processor nulled out from under it mid-inference.
"""
import threading

from src.models.clip import MLXClip
import src.models.clip as clip_mod


def _fake_load(self):
    """Stand-in for the real _load_model — no mlx/open_clip, just deterministic state."""
    self._model = object()
    self._processor = object()
    self._loaded = True


def test_unload_does_not_null_live_instance(monkeypatch):
    monkeypatch.setattr(MLXClip, "_load_model", _fake_load)
    clip = MLXClip("test-model")
    model_ref, processor_ref = clip._model, clip._processor

    clip.unload()

    assert clip._model is model_ref
    assert clip._processor is processor_ref
    assert clip._loaded is True


def test_concurrent_switch_does_not_corrupt_in_flight_instance(monkeypatch):
    monkeypatch.setattr(MLXClip, "_load_model", _fake_load)
    clip_mod._current_model = None
    clip_mod._current_model_name = None

    # Simulates a request that already captured a reference to the current
    # model — exactly what clip.py's encode_image()/_encode_image_fallback()
    # hold onto across preprocessing, before a concurrent request switches models.
    slow = clip_mod.get_clip_model("slow-model")

    t = threading.Thread(target=lambda: clip_mod.get_clip_model("fast-model"))
    t.start()
    t.join(timeout=2)

    assert slow._model is not None
    assert slow._processor is not None
