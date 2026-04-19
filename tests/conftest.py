"""Shared fixtures for immich-ml-metal tests."""
import os
import pytest

# Force STUB_MODE for unit tests — no real models needed.
os.environ["STUB_MODE"] = "true"


@pytest.fixture
def test_image_bytes():
    """A minimal valid JPEG for testing (red 100x100)."""
    from PIL import Image
    import io
    img = Image.new("RGB", (100, 100), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def test_image_large_bytes():
    """A larger JPEG for more realistic testing (640x480)."""
    from PIL import Image
    import io
    img = Image.new("RGB", (640, 480), color=(128, 180, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()
