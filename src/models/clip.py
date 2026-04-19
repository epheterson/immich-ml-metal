"""
CLIP model implementation using MLX for Apple Silicon acceleration.

Supports dynamic model loading based on Immich requests.
Thread-safe for both loading and inference.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image
from typing import Optional
import io
import gc
import logging
import threading

logger = logging.getLogger(__name__)

# Model name mapping: Immich name -> MLX repo (or None to use open_clip fallback)
MODEL_MAP = {
    # OpenAI CLIP models -> MLX
    "ViT-B-32__openai": "mlx-community/clip-vit-base-patch32",
    "ViT-B-16__openai": "mlx-community/clip-vit-base-patch16",
    "ViT-L-14__openai": "mlx-community/clip-vit-large-patch14",
    # LAION CLIP models -> MLX
    "ViT-B-32__laion2b-s34b-b79k": "mlx-community/clip-vit-base-patch32-laion2b",
    "ViT-B-32__laion2b_s34b_b79k": "mlx-community/clip-vit-base-patch32-laion2b",
    # SigLIP models -> None (use open_clip fallback)
    "ViT-B-16-SigLIP__webli": None,
    "ViT-B-16-SigLIP2__webli": None,
    "ViT-SO400M-16-SigLIP2-384__webli": None,
    # Default fallback
    "default": "mlx-community/clip-vit-base-patch32",
}

# open_clip model name mappings for fallback
OPENCLIP_MAP = {
    "ViT-B-32__openai": ("ViT-B-32-quickgelu", "openai"),
    "ViT-B-16__openai": ("ViT-B-16", "openai"),
    "ViT-L-14__openai": ("ViT-L-14", "openai"),
    "ViT-B-32__laion2b-s34b-b79k": ("ViT-B-32", "laion2b_s34b_b79k"),
    "ViT-B-32__laion2b_s34b_b79k": ("ViT-B-32", "laion2b_s34b_b79k"),
    "ViT-B-16-SigLIP__webli": ("ViT-B-16-SigLIP", "webli"),
    "ViT-B-16-SigLIP2__webli": ("ViT-B-16-SigLIP2", "webli"),
    "ViT-SO400M-16-SigLIP2-384__webli": ("ViT-SO400M-16-SigLIP2-384", "webli"),
}


class MLXClip:
    """CLIP model using MLX for Apple Silicon acceleration."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._loaded = False
        self._repo_id = MODEL_MAP.get(model_name, MODEL_MAP.get("default"))
        # Use the global metal_lock — Vision framework also touches Metal
        # and concurrent MLX + Vision Metal access crashes the process.
        from src.gpu_lock import metal_lock

        self._inference_lock = metal_lock
        self._load_model()

    def _load_model(self):
        """Load the MLX CLIP model, or fallback to open_clip."""
        self._repo_id = MODEL_MAP.get(self.model_name)

        if self._repo_id is None and self.model_name not in OPENCLIP_MAP:
            logger.warning(
                f"Unknown model '{self.model_name}', using MLX default (ViT-B-32)"
            )
            self._repo_id = MODEL_MAP["default"]

        if self._repo_id is None:
            logger.info(
                f"No MLX version for {self.model_name}, using open_clip fallback"
            )
            self._load_fallback()
            return

        try:
            from mlx_clip import mlx_clip

            logger.info(f"Loading MLX CLIP model: {self.model_name} -> {self._repo_id}")
            self._model = mlx_clip(self._repo_id)
            self._loaded = True
            logger.info(f"Successfully loaded CLIP model via MLX: {self.model_name}")

        except ImportError:
            logger.warning("mlx_clip not available, falling back to open_clip with MPS")
            self._load_fallback()
        except Exception as e:
            logger.error(f"MLX model loading failed: {e}", exc_info=True)
            logger.info("Falling back to open_clip")
            self._load_fallback()

    def _load_fallback(self):
        """Fallback to open_clip with MPS acceleration."""
        try:
            import torch
            import open_clip
        except ImportError as e:
            logger.error(f"open_clip not available and MLX failed: {e}")
            raise RuntimeError(
                "Neither mlx_clip nor open_clip available. "
                "Install one with: pip install open-clip-torch"
            ) from e

        if self.model_name in OPENCLIP_MAP:
            arch, pretrained = OPENCLIP_MAP[self.model_name]
        elif "__" in self.model_name:
            arch, pretrained = self.model_name.split("__", 1)
            if (
                pretrained == "openai"
                and "quickgelu" not in arch.lower()
                and "siglip" not in arch.lower()
            ):
                arch = arch + "-quickgelu"
        else:
            arch = "ViT-B-32-quickgelu"
            pretrained = "openai"

        logger.info(f"Loading open_clip model: {arch} / {pretrained}")

        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                arch, pretrained=pretrained
            )
            tokenizer = open_clip.get_tokenizer(arch)
        except Exception as e:
            logger.warning(f"Failed to load {arch}/{pretrained}: {e}")
            logger.info("Falling back to ViT-B-32-quickgelu/openai")
            arch, pretrained = "ViT-B-32-quickgelu", "openai"
            model, _, preprocess = open_clip.create_model_and_transforms(
                arch, pretrained=pretrained
            )
            tokenizer = open_clip.get_tokenizer(arch)

        if torch.backends.mps.is_available():
            self._device = torch.device("mps")
            model = model.to(self._device)
            logger.info("Using MPS (Metal) acceleration")
        else:
            self._device = torch.device("cpu")
            logger.warning("MPS not available, using CPU")

        model.eval()

        self._model = model
        self._processor = preprocess
        self._tokenizer = tokenizer
        self._use_fallback = True
        self._loaded = True

        logger.info(
            f"Successfully loaded CLIP model via open_clip: {arch}/{pretrained}"
        )

    def encode_image(self, image_bytes: bytes) -> np.ndarray:
        """
        Generate CLIP embedding for an image.
        Thread-safe — only GPU inference is serialized. Image decode and
        preprocessing run outside the lock. A reference to the model is
        captured before preprocessing and verified after acquiring the lock
        so a concurrent model switch cannot cause a mismatch. If the model
        was swapped mid-flight, preprocessing is re-run once against the
        new model.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        # Decode outside lock — this is CPU work, not GPU
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        if hasattr(self, "_use_fallback") and self._use_fallback:
            return self._encode_image_fallback(image)

        # MLX path — preprocess the PIL image directly (no temp file needed).
        # Capture model reference before preprocessing so we can detect
        # if the model was swapped between preprocessing and inference.
        # One retry: if a concurrent request switched models between
        # preprocessing and lock acquisition, re-preprocess with the
        # current model and try once more.
        for attempt in range(2):
            model_ref = self._model
            processed = model_ref.img_processor([image])

            with self._inference_lock:
                if self._model is not model_ref:
                    if attempt == 0:
                        logger.warning(
                            "CLIP model changed during preprocessing, retrying"
                        )
                        continue
                    raise RuntimeError(
                        "CLIP model changed during preprocessing after retry"
                    )
                output = self._model.model(**{"pixel_values": processed})
                embedding = output.image_embeds[0]
                # Force Metal evaluation inside the lock — MLX arrays are lazy,
                # and Metal work must complete before releasing the lock so
                # Vision framework calls don't collide with in-flight Metal ops.
                if isinstance(embedding, mx.array):
                    embedding = np.array(embedding)

            embedding = embedding / np.linalg.norm(embedding)
            return embedding.flatten().astype(np.float32)

        # Should never reach here — range(2) always runs and either
        # returns or raises. Defensive guard.
        raise RuntimeError("CLIP encode_image failed to produce an embedding")

    def _encode_image_fallback(self, image: Image.Image) -> np.ndarray:
        """Encode image using open_clip fallback.

        Preprocessing (resize/normalize) runs outside the lock since it's
        CPU-only. Only the MPS/GPU inference is serialized. Model reference
        is captured before preprocessing and verified after lock acquisition.
        One retry on model swap, same as encode_image.
        """
        import torch

        for attempt in range(2):
            model_ref = self._model
            image_tensor = self._processor(image).unsqueeze(0).to(self._device)

            with self._inference_lock:
                if self._model is not model_ref:
                    if attempt == 0:
                        logger.warning(
                            "CLIP model changed during preprocessing (fallback), retrying"
                        )
                        continue
                    raise RuntimeError(
                        "CLIP model changed during preprocessing after retry"
                    )
                with torch.no_grad():
                    embedding = self._model.encode_image(image_tensor)
                    embedding = embedding / embedding.norm(dim=-1, keepdim=True)

            # .cpu() triggers MPS device sync — safe outside the lock
            # because MPS uses its own command queue (unlike MLX which
            # shares the Metal command buffer with Vision framework).
            return embedding.squeeze().cpu().numpy().astype(np.float32)

        raise RuntimeError("CLIP encode_image_fallback failed to produce an embedding")

    def encode_text(self, text: str) -> np.ndarray:
        """
        Generate CLIP embedding for text.
        Thread-safe — only GPU inference is serialized, matching encode_image.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        if hasattr(self, "_use_fallback") and self._use_fallback:
            return self._encode_text_fallback(text)

        with self._inference_lock:
            embedding = self._model.text_encoder(text)
            if isinstance(embedding, mx.array):
                embedding = np.array(embedding)
            embedding = embedding / np.linalg.norm(embedding)
            return embedding.flatten().astype(np.float32)

    def _encode_text_fallback(self, text: str) -> np.ndarray:
        """Encode text using open_clip fallback.

        Tokenization runs outside the lock since it's CPU-only.
        Only the MPS/GPU inference is serialized, matching _encode_image_fallback.
        One retry on model swap, same as encode_image paths.
        """
        import torch

        for attempt in range(2):
            model_ref = self._model
            tokens = self._tokenizer([text]).to(self._device)

            with self._inference_lock:
                if self._model is not model_ref:
                    if attempt == 0:
                        logger.warning(
                            "CLIP model changed during tokenization (text fallback), retrying"
                        )
                        continue
                    raise RuntimeError(
                        "CLIP model changed during tokenization after retry"
                    )
                with torch.no_grad():
                    embedding = self._model.encode_text(tokens)
                    embedding = embedding / embedding.norm(dim=-1, keepdim=True)

            return embedding.squeeze().cpu().numpy().astype(np.float32)

        raise RuntimeError("CLIP encode_text_fallback failed to produce an embedding")

    def unload(self):
        """Unload model and free memory."""
        logger.info(f"Unloading CLIP model: {self.model_name}")
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._loaded = False

        gc.collect()

        try:
            mx.clear_cache()
        except AttributeError:
            try:
                mx.metal.clear_cache()
            except Exception:
                pass


# Global model cache with thread safety
_current_model: Optional[MLXClip] = None
_current_model_name: Optional[str] = None
_model_lock = threading.Lock()


def get_clip_model(model_name: str = "ViT-B-32__openai") -> MLXClip:
    """
    Get CLIP model, loading or switching as needed (thread-safe).

    If a different model is requested, unloads current model first to free memory.
    """
    global _current_model, _current_model_name

    normalized_name = model_name.replace("::", "__")

    with _model_lock:
        if _current_model is not None and _current_model_name != normalized_name:
            logger.info(
                f"Switching CLIP model: {_current_model_name} -> {normalized_name}"
            )
            _current_model.unload()
            _current_model = None
            _current_model_name = None

        if _current_model is None:
            logger.info(f"Loading CLIP model: {normalized_name}")
            _current_model = MLXClip(normalized_name)
            _current_model_name = normalized_name

        return _current_model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    import sys

    logger.info("Testing CLIP model loading...")
    logger.info(
        f"Supported MLX models: {[k for k, v in MODEL_MAP.items() if v is not None and k != 'default']}"
    )
    logger.info(f"Supported open_clip models: {list(OPENCLIP_MAP.keys())}")

    logger.info("\n--- Testing MLX model ---")
    clip = get_clip_model("ViT-B-32__openai")
    text_emb = clip.encode_text("a photo of a cat")
    logger.info(f"Text embedding shape: {text_emb.shape}")
    logger.info(f"Text embedding norm: {np.linalg.norm(text_emb):.4f}")

    if len(sys.argv) > 1:
        with open(sys.argv[1], "rb") as f:
            img_emb = clip.encode_image(f.read())
        logger.info(f"Image embedding shape: {img_emb.shape}")
        similarity = np.dot(text_emb, img_emb)
        logger.info(f"Text-image similarity: {similarity:.4f}")

    logger.info("\n✅ CLIP tests passed!")
