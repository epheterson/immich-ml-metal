"""Tests for model memory management — busy guards, unload decisions, tracking."""
import time

from src.main import (
    _mark_model_busy,
    _track_model_use,
    _should_unload,
    _model_busy,
    _model_last_used,
    MODEL_MEMORY_FLOOR_MB,
)


def _cleanup():
    """Reset global state between tests."""
    _model_busy.discard("clip")
    _model_busy.discard("face")
    _model_last_used.pop("clip", None)
    _model_last_used.pop("face", None)


def setup_function():
    _cleanup()


def teardown_function():
    _cleanup()


# --- Busy guard ---

def test_busy_model_never_unloads():
    _mark_model_busy("clip")
    assert "clip" in _model_busy
    # Even with 0MB available, busy model should not unload
    assert not _should_unload("clip", time.monotonic(), 0)


def test_track_clears_busy():
    _mark_model_busy("clip")
    _track_model_use("clip")
    assert "clip" not in _model_busy
    assert "clip" in _model_last_used


# --- Pressure strategy ---

def test_pressure_recently_used_not_unloaded():
    """Model used just now should not unload even under pressure."""
    _track_model_use("clip")
    assert not _should_unload("clip", time.monotonic(), 0)


def test_pressure_idle_and_low_memory_unloads():
    """Model idle for >30s with low memory should unload."""
    _model_last_used["clip"] = time.monotonic() - 60
    assert _should_unload("clip", time.monotonic(), 0)


def test_pressure_idle_but_enough_memory_stays():
    """Model idle for >30s but plenty of memory — don't unload."""
    _model_last_used["clip"] = time.monotonic() - 60
    assert not _should_unload("clip", time.monotonic(), MODEL_MEMORY_FLOOR_MB + 100)


# --- Timeout strategy ---

def test_timeout_strategy(monkeypatch):
    monkeypatch.setattr("src.main.MODEL_UNLOAD_STRATEGY", "timeout")
    monkeypatch.setattr("src.main.MODEL_IDLE_TIMEOUT", 10)
    _model_last_used["clip"] = time.monotonic() - 15
    assert _should_unload("clip", time.monotonic(), 9999)


def test_timeout_strategy_not_yet(monkeypatch):
    monkeypatch.setattr("src.main.MODEL_UNLOAD_STRATEGY", "timeout")
    monkeypatch.setattr("src.main.MODEL_IDLE_TIMEOUT", 10)
    _track_model_use("clip")
    assert not _should_unload("clip", time.monotonic(), 9999)


# --- Never strategy ---

def test_never_strategy(monkeypatch):
    monkeypatch.setattr("src.main.MODEL_UNLOAD_STRATEGY", "never")
    _model_last_used["clip"] = time.monotonic() - 9999
    assert not _should_unload("clip", time.monotonic(), 0)


# --- Multiple models ---

def test_independent_model_tracking():
    """CLIP and face have independent busy/tracking state."""
    _mark_model_busy("clip")
    _track_model_use("face")
    assert "clip" in _model_busy
    assert "face" not in _model_busy
    assert "face" in _model_last_used
    assert "clip" not in _model_last_used
