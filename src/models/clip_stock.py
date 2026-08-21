"""
CLIP with Immich's own ONNX exports, for embedding parity with Docker.

The mlx path in clip.py is faster and produces embeddings that agree with these
to better than 0.999 cosine, which is close enough that search results are
equivalent. Close is not the same as identical, though, and a library whose
embeddings were built here cannot be moved back to a Docker install and topped
up without a subtle discontinuity in the index. The Stock position exists so it
can.

This runs the same files the Docker service runs: huggingface.co/immich-app/
<model>, visual and textual towers as ONNX, with the preprocessing described by
the model's own preprocess_cfg.json rather than assumed. The native engine does
exactly this in ZooCLIP.swift; this is the Python side of the same idea, and the
file layout below is deliberately the same list.
"""

import json
import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_model = None
_model_name: Optional[str] = None
_lock = threading.Lock()

# What a model needs locally. Same list as the native engine's, so a model that
# works on one engine works on the other.
_FILES = [
    "config.json",
    "visual/model.onnx",
    "visual/preprocess_cfg.json",
    "textual/model.onnx",
    "textual/tokenizer.json",
    "textual/tokenizer_config.json",
]


class _StockCLIP:
    def __init__(self, name: str, cache_dir: Path):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        repo = f"immich-app/{name}"
        local: dict[str, str] = {}
        for rel in _FILES:
            local[rel] = hf_hub_download(
                repo_id=repo, filename=rel, cache_dir=str(cache_dir)
            )

        # Some exports keep the weights in a sibling .onnx_data file that the
        # graph references by relative path, so the session has to be opened
        # from the directory the file actually lives in.
        visual_dir = Path(local["visual/model.onnx"]).parent
        try:
            hf_hub_download(
                repo_id=repo,
                filename="visual/model.onnx_data",
                cache_dir=str(cache_dir),
            )
        except Exception:
            pass  # Most exports are self-contained; absence is normal.

        providers = ["CPUExecutionProvider"]
        self.visual = ort.InferenceSession(
            local["visual/model.onnx"], providers=providers
        )
        self.textual = ort.InferenceSession(
            local["textual/model.onnx"], providers=providers
        )
        self.tokenizer = Tokenizer.from_file(local["textual/tokenizer.json"])

        cfg = json.loads(Path(local["visual/preprocess_cfg.json"]).read_text())
        size = cfg["size"]
        # The key is an int for square inputs and a pair for everything else.
        self.size = (size, size) if isinstance(size, int) else tuple(size[-2:])
        self.mean = np.array(cfg["mean"], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(cfg["std"], dtype=np.float32).reshape(3, 1, 1)
        # open_clip's own default when the export does not say (transform.py).
        self.resize_mode = cfg.get("resize_mode", "shortest")

        tok_cfg = json.loads(Path(local["textual/tokenizer_config.json"]).read_text())
        self.context_length = int(tok_cfg.get("model_max_length", 77))
        _ = visual_dir  # kept for clarity about why the download above matters

    def _preprocess(self, img) -> np.ndarray:
        from PIL import Image

        target_h, target_w = self.size
        if self.resize_mode == "squash":
            resized = img.resize((target_w, target_h), Image.Resampling.BICUBIC)
        else:
            # "shortest": scale so the short side meets the target, then centre
            # crop. Doing this the other way round crops the subject out.
            scale = max(target_w / img.width, target_h / img.height)
            new = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
            resized = img.resize(new, Image.Resampling.BICUBIC)
            left = (resized.width - target_w) // 2
            top = (resized.height - target_h) // 2
            resized = resized.crop((left, top, left + target_w, top + target_h))

        arr = np.asarray(resized.convert("RGB"), dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        arr = (arr - self.mean) / self.std
        return np.expand_dims(arr, axis=0).astype(np.float32)

    def encode_image(self, img) -> list[float]:
        inputs = {self.visual.get_inputs()[0].name: self._preprocess(img)}
        out = self.visual.run(None, inputs)[0]
        return _normalize(np.asarray(out[0], dtype=np.float32))

    def encode_text(self, text: str) -> list[float]:
        ids = self.tokenizer.encode(text).ids[: self.context_length]
        # CLIP's text tower takes a fixed-width context, zero padded. The index
        # dtype is read from the model rather than assumed: Immich's exports are
        # not consistent about int32 against int64, and onnxruntime rejects the
        # wrong one outright instead of coercing.
        spec = self.textual.get_inputs()[0]
        dtype = np.int32 if "int32" in spec.type else np.int64
        padded = np.zeros((1, self.context_length), dtype=dtype)
        padded[0, : len(ids)] = ids
        name = spec.name
        out = self.textual.run(None, {name: padded})[0]
        return _normalize(np.asarray(out[0], dtype=np.float32))


def _normalize(vec: np.ndarray) -> list[float]:
    norm = float(np.linalg.norm(vec))
    if norm == 0:
        return vec.astype(np.float32).tolist()
    return (vec / norm).astype(np.float32).tolist()


def get_model(name: str, cache_dir: Path) -> _StockCLIP:
    """Build (once) the stock CLIP towers, swapping if the model changed."""
    global _model, _model_name
    with _lock:
        if _model is not None and _model_name == name:
            return _model
        if _model is not None:
            logger.info("Switching stock CLIP model: %s -> %s", _model_name, name)
            _model = None
        logger.info("Loading stock CLIP model: %s", name)
        _model = _StockCLIP(name, cache_dir)
        _model_name = name
        return _model


def unload_model() -> None:
    global _model, _model_name
    with _lock:
        _model = None
        _model_name = None


def encode_image(image_bytes: bytes, name: str, cache_dir: Path) -> list[float]:
    import io

    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return get_model(name, cache_dir).encode_image(img)


def encode_text(text: str, name: str, cache_dir: Path) -> list[float]:
    return get_model(name, cache_dir).encode_text(text)


class StockCLIPModel:
    """The interface the service expects from a CLIP model: bytes in, vector
    out. Wraps the towers so callers need not know which path they are on."""

    def __init__(self, name: str, cache_dir: Path):
        self._name = name
        self._cache_dir = cache_dir

    def encode_image(self, image_bytes: bytes) -> list[float]:
        return encode_image(image_bytes, self._name, self._cache_dir)

    def encode_text(self, text: str) -> list[float]:
        return encode_text(text, self._name, self._cache_dir)


def get_stock_clip_model(name: str, cache_dir: Path) -> StockCLIPModel:
    # Build the towers now rather than on first encode, so a bad model name or
    # a failed download surfaces where the caller can report it.
    get_model(name, cache_dir)
    return StockCLIPModel(name, cache_dir)
