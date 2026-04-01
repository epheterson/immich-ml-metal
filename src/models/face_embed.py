"""
Face embedding generation using InsightFace ArcFace model.

Uses CoreML execution provider when available for Apple Silicon acceleration.
Thread-safe for both loading and inference.
"""

import numpy as np
from PIL import Image
import cv2
import io
from typing import Optional
from pathlib import Path
import logging
import threading
import gc

logger = logging.getLogger(__name__)

# Expected model characteristics for ArcFace recognition models
ARCFACE_INPUT_SIZE = 112
ARCFACE_EMBEDDING_DIM = 512

# Global model cache with thread safety
_recognition_model = None
_current_model_name: Optional[str] = None
_model_lock = threading.Lock()
_inference_lock = threading.Lock()


def get_recognition_model(model_name: str = "buffalo_l"):
    """
    Get or create the face recognition model (thread-safe).
    
    Loads the recognition model directly from insightface model zoo.
    If a different model is requested, unloads the current one first.
    
    Args:
        model_name: InsightFace model name (buffalo_s, buffalo_m, buffalo_l)
    """
    global _recognition_model, _current_model_name
    
    with _model_lock:
        if _recognition_model is not None and _current_model_name != model_name:
            logger.info(f"Switching face model: {_current_model_name} -> {model_name}")
            unload_recognition_model()
        
        if _recognition_model is None:
            logger.info(f"Loading face recognition model: {model_name}")
            _load_model(model_name)
            _current_model_name = model_name
        
        return _recognition_model


def _find_recognition_model(model_dir: Path) -> Optional[Path]:
    """
    Find the recognition model in a model pack directory.
    
    Uses shape validation to identify ArcFace models rather than
    relying on filename patterns which may change between versions.
    
    Returns:
        Path to the recognition model, or None if not found.
    """
    import onnxruntime as ort
    
    onnx_files = list(model_dir.glob("*.onnx"))
    if not onnx_files:
        return None
    
    # Known filename patterns (checked first for speed)
    known_patterns = ["w600k", "w300k", "glintr", "arcface"]
    
    for f in onnx_files:
        if any(pattern in f.name.lower() for pattern in known_patterns):
            # Verify it's actually a recognition model by checking shapes
            if _validate_recognition_model(f):
                return f
    
    # Fallback: check all ONNX files for correct input/output shapes
    for f in onnx_files:
        if _validate_recognition_model(f):
            logger.info(f"Found recognition model by shape validation: {f.name}")
            return f
    
    return None


def _validate_recognition_model(model_path: Path) -> bool:
    """
    Validate that an ONNX model is an ArcFace recognition model.
    
    Checks:
    - Input shape compatible with (batch, 3, 112, 112)
    - Output shape compatible with (batch, 512)
    """
    try:
        import onnxruntime as ort
        
        # Quick session just to check metadata
        sess = ort.InferenceSession(
            str(model_path), 
            providers=["CPUExecutionProvider"]
        )
        
        # Check input shape
        input_info = sess.get_inputs()[0]
        input_shape = input_info.shape
        # Shape could be [batch, 3, 112, 112] or with dynamic batch
        if len(input_shape) != 4:
            return False
        # Check spatial dimensions (indices 2 and 3)
        if input_shape[2] != ARCFACE_INPUT_SIZE or input_shape[3] != ARCFACE_INPUT_SIZE:
            return False
        # Check channels (index 1)
        if input_shape[1] != 3:
            return False
        
        # Check output shape
        output_info = sess.get_outputs()[0]
        output_shape = output_info.shape
        if len(output_shape) != 2:
            return False
        # Check embedding dimension (index 1)
        if output_shape[1] != ARCFACE_EMBEDDING_DIM:
            return False
        
        return True
        
    except Exception as e:
        logger.debug(f"Model validation failed for {model_path}: {e}")
        return False


def _load_model(model_name: str):
    """Internal function to load the model (assumes lock is held)."""
    global _recognition_model
    
    try:
        import onnxruntime as ort
        from insightface.model_zoo import model_zoo
        from insightface.utils.storage import download as download_model_pack
    except ImportError as e:
        logger.error(f"Failed to import required packages: {e}")
        raise RuntimeError(
            "insightface or onnxruntime not available. "
            "Install with: pip install insightface onnxruntime"
        ) from e
    
    available = ort.get_available_providers()
    logger.info(f"ONNX Runtime providers available: {available}")
    
    if "CoreMLExecutionProvider" in available:
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        logger.info("Using CoreML for face recognition")
    else:
        providers = ["CPUExecutionProvider"]
        logger.info("Using CPU for face recognition (CoreML not available)")

    rec_model_path = _ensure_recognition_model_pack(model_name, download_model_pack)
    
    logger.info(f"Loading recognition model: {rec_model_path.name}")
    
    try:
        _recognition_model = model_zoo.get_model(str(rec_model_path), providers=providers)
        logger.info(f"Successfully loaded face recognition model: {model_name}")
    except Exception as e:
        logger.error(f"Failed to load recognition model: {e}", exc_info=True)
        raise RuntimeError(f"Could not load {model_name} recognition model") from e


def _ensure_recognition_model_pack(model_name: str, download_model_pack) -> Path:
    """
    Ensure the face model pack exists and contains a valid ArcFace model.

    If the cached pack is missing or does not contain a usable ArcFace model,
    refresh it once from the upstream InsightFace release.
    """
    insightface_root = Path.home() / ".insightface"
    model_dir = insightface_root / "models" / model_name

    rec_model_path = _find_recognition_model(model_dir) if model_dir.exists() else None
    if rec_model_path is not None:
        return rec_model_path

    if model_dir.exists():
        logger.warning(
            "No valid recognition model found in cached pack %s; refreshing download",
            model_dir,
        )
        force = True
    else:
        logger.info(f"Downloading {model_name} model pack")
        force = False

    try:
        download_model_pack("models", model_name, force=force, root=str(insightface_root))
    except Exception as e:
        logger.error(f"Failed to download model pack: {e}", exc_info=True)
        raise RuntimeError(f"Could not download {model_name} model pack") from e

    rec_model_path = _find_recognition_model(model_dir)
    if rec_model_path is not None:
        return rec_model_path

    available_files = [f.name for f in model_dir.glob("*.onnx")] if model_dir.exists() else []
    logger.error(f"No valid recognition model found in {model_dir}")
    logger.error(f"Available ONNX files: {available_files}")
    raise FileNotFoundError(
        f"No ArcFace recognition model (input: 3x112x112, output: 512-dim) "
        f"found in {model_dir}"
    )


def unload_recognition_model():
    """Unload the current recognition model and free memory."""
    global _recognition_model, _current_model_name
    
    if _recognition_model is not None:
        logger.info(f"Unloading face recognition model: {_current_model_name}")
        _recognition_model = None
        _current_model_name = None
        gc.collect()


def get_face_embedding(
    image_bytes: bytes,
    landmarks: list[list[float]],
    model_name: str = "buffalo_l"
) -> np.ndarray:
    """
    Generate 512-dim face embedding using ArcFace (thread-safe).
    
    Args:
        image_bytes: Raw image data
        landmarks: 5-point landmarks [[x,y], ...] from Vision framework
                  Order: left_eye, right_eye, nose, left_mouth, right_mouth
        model_name: InsightFace model to use
        
    Returns:
        512-dimensional normalized embedding as float32 array
    """
    try:
        from insightface.utils import face_align
    except ImportError as e:
        logger.error("insightface not available")
        raise RuntimeError("Install insightface: pip install insightface") from e
    
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Failed to decode image")
    except Exception as e:
        logger.error(f"Image decoding failed: {e}")
        raise
    
    kps = np.array(landmarks, dtype=np.float32)
    
    try:
        aligned_face = face_align.norm_crop(img_bgr, kps, image_size=ARCFACE_INPUT_SIZE)
    except Exception as e:
        logger.error(f"Face alignment failed: {e}")
        raise
    
    model = get_recognition_model(model_name)
    
    with _inference_lock:
        try:
            embedding = model.get_feat(aligned_face)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    embedding = embedding.flatten()
    embedding = embedding / np.linalg.norm(embedding)
    
    return embedding.astype(np.float32)


def get_face_embedding_from_bbox(
    image_bytes: bytes,
    bbox: dict,
    model_name: str = "buffalo_l"
) -> Optional[np.ndarray]:
    """
    Generate face embedding using bounding box (fallback, thread-safe).
    
    Less accurate than landmark-based alignment but works as fallback.
    
    Args:
        image_bytes: Raw image data
        bbox: Bounding box dict with x1, y1, x2, y2
        model_name: InsightFace model to use
        
    Returns:
        512-dimensional normalized embedding, or None if failed
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            logger.error("Failed to decode image")
            return None
        
        x1, y1 = int(bbox["x1"]), int(bbox["y1"])
        x2, y2 = int(bbox["x2"]), int(bbox["y2"])
        
        # Add 10% padding
        w, h = x2 - x1, y2 - y1
        pad_x, pad_y = int(w * 0.1), int(h * 0.1)
        
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(img_bgr.shape[1], x2 + pad_x)
        y2 = min(img_bgr.shape[0], y2 + pad_y)
        
        face_crop = img_bgr[y1:y2, x1:x2]
        
        if face_crop.size == 0:
            logger.error("Empty face crop")
            return None
        
        face_resized = cv2.resize(face_crop, (ARCFACE_INPUT_SIZE, ARCFACE_INPUT_SIZE))
        
        model = get_recognition_model(model_name)
        
        with _inference_lock:
            embedding = model.get_feat(face_resized)
        
        embedding = embedding.flatten()
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding.astype(np.float32)
        
    except Exception as e:
        logger.error(f"Embedding from bbox failed: {e}", exc_info=True)
        return None


# Alias for backward compatibility
def get_face_recognizer(model_name: str = "buffalo_l"):
    """Alias for get_recognition_model for backward compatibility."""
    return get_recognition_model(model_name)


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    logger.info("Testing face embedding generation...")
    logger.info("(This will download the model on first run)")
    
    model = get_recognition_model("buffalo_l")
    logger.info(f"Model loaded successfully: {type(model).__name__}")
    
    if len(sys.argv) > 1:
        from .face_detect import detect_faces
        
        with open(sys.argv[1], "rb") as f:
            image_bytes = f.read()
        
        faces, w, h = detect_faces(image_bytes)
        logger.info(f"Detected {len(faces)} face(s) in {w}x{h} image")
        
        for i, face in enumerate(faces):
            logger.info(f"\nFace {i + 1}:")
            logger.info(f"  BBox: ({face['boundingBox']['x1']:.0f}, {face['boundingBox']['y1']:.0f}) - "
                       f"({face['boundingBox']['x2']:.0f}, {face['boundingBox']['y2']:.0f})")
            logger.info(f"  Score: {face['score']:.3f}")
            
            if "landmarks" in face:
                embedding = get_face_embedding(image_bytes, face["landmarks"], "buffalo_l")
                logger.info(f"  Embedding (landmark-aligned): {embedding.shape}")
            else:
                embedding = get_face_embedding_from_bbox(image_bytes, face["boundingBox"], "buffalo_l")
                logger.info(f"  Embedding (bbox-cropped): {embedding.shape}")
            
            if embedding is not None:
                logger.info(f"  Embedding norm: {np.linalg.norm(embedding):.4f}")
                logger.info(f"  First 5 values: {embedding[:5]}")
    
    logger.info("\n✅ Face embedding test complete!")
