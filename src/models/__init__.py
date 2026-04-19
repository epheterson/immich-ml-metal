"""
Model implementations for immich-ml-metal.

- clip: CLIP image/text embeddings (MLX/open_clip)
- face_detect: Face detection (Apple Vision framework)
- face_embed: Face embeddings (InsightFace ArcFace)
- ocr: Text recognition (Apple Vision framework)
"""

from .clip import get_clip_model, MLXClip
from .face_detect import detect_faces
from .face_embed import get_face_embedding, get_face_embeddings_batch, get_recognition_model
from .ocr import recognize_text

__all__ = [
    "get_clip_model",
    "MLXClip", 
    "detect_faces",
    "get_face_embedding",
    "get_face_embeddings_batch",
    "get_recognition_model",
    "recognize_text",
]