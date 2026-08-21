"""
Model implementations for immich-ml-metal.

- clip: CLIP image/text embeddings (MLX/open_clip)
- face_detect: Face detection (Apple Vision framework)
- face_embed: Face embeddings (InsightFace ArcFace)
- ocr: Text recognition (Apple Vision framework)
"""

from .clip import get_clip_model as get_clip_model_mlx, MLXClip


def get_clip_model(model_name):
    """Return whichever CLIP this install is configured for.

    Same seam as faces and OCR. Stock runs Immich's own ONNX exports, whose
    embeddings agree with the mlx path to better than 0.999 cosine but are not
    identical, which is the entire reason the position exists.
    """
    from ..config import settings

    if getattr(settings, "stock_ml", False):
        from .clip_stock import get_stock_clip_model

        return get_stock_clip_model(model_name, settings.cache_dir)
    return get_clip_model_mlx(model_name)
from .face_detect import detect_faces as detect_faces_vision


def detect_faces(image_bytes, min_score=None):
    """Detect faces with whichever detector this install is configured for.

    One seam rather than a branch at each call site: the health check and the
    recognition path must never disagree about which detector is live, or the
    check passes against a detector that is not the one doing the work.
    """
    from ..config import settings

    if getattr(settings, "stock_faces", False):
        from .face_detect_stock import detect_faces as detect_faces_stock

        return detect_faces_stock(
            image_bytes,
            min_score if min_score is not None else 0.0,
            settings.face_model,
        )
    return detect_faces_vision(image_bytes)
from .face_embed import get_face_embedding, get_face_embeddings_batch, get_recognition_model
from .ocr import recognize_text as recognize_text_vision


def recognize_text(image_bytes, min_confidence=0.5, use_language_correction=True):
    """Read text with whichever recogniser this install is configured for.

    Same seam as detect_faces, for the same reason: one place decides, so the
    health check and the work can never disagree about which one is live.
    """
    from ..config import settings

    if getattr(settings, "stock_ml", False):
        from .ocr_stock import recognize_text as recognize_text_stock

        return recognize_text_stock(
            image_bytes,
            settings.models_dir,
            settings.ocr_min_detection_score,
            min_confidence,
        )
    return recognize_text_vision(
        image_bytes,
        min_confidence=min_confidence,
        use_language_correction=use_language_correction,
    )

__all__ = [
    "get_clip_model",
    "MLXClip", 
    "detect_faces",
    "get_face_embedding",
    "get_face_embeddings_batch",
    "get_recognition_model",
    "recognize_text",
]