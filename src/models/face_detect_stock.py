"""
Face detection with Immich's own ONNX detector, for output parity with Docker.

The Apple Vision detector in face_detect.py is faster and runs on the Neural
Engine, but it is a different detector: it finds a slightly different set of
faces and draws slightly different boxes. That is fine until someone needs a
library that can move back to a Docker deployment without re-detecting every
face, which is what the Stock position exists for.

This runs what Immich runs. Immich builds `RetinaFace` from the insightface
model zoo over the detection model in the same `buffalo_l` pack we already
download for recognition, prepares it at 640x640, and filters by a score
threshold (machine-learning/immich_ml/models/facial_recognition/detection.py).
Same library, same model file, same input size, so the boxes match by
construction rather than by resemblance.
"""

import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# insightface builds its own session per model; loading is slow and the service
# is multi-threaded, so build once behind a lock like the recognition model.
_detector = None
_detector_lock = threading.Lock()
_detector_input_size = (640, 640)


def _find_detection_model(model_dir: Path) -> Optional[Path]:
    """Locate the detection model inside a buffalo pack.

    By shape rather than by filename: the pack ships several .onnx files and
    the naming has changed across insightface releases. The detector is the one
    with four outputs per stride (scores, boxes, and keypoints), which no other
    model in the pack has, so the filename is only a fast path.
    """
    onnx_files = sorted(model_dir.glob("*.onnx"))
    if not onnx_files:
        return None

    for f in onnx_files:
        if "det_" in f.name.lower() or "scrfd" in f.name.lower():
            return f

    # Fall back to asking the models what they are, so a rename cannot break us.
    import onnxruntime as ort

    for f in onnx_files:
        try:
            sess = ort.InferenceSession(str(f), providers=["CPUExecutionProvider"])
            inputs = sess.get_inputs()
            # A detector takes one image and returns many tensors; the ArcFace
            # recognition model returns exactly one embedding.
            if len(inputs) == 1 and len(sess.get_outputs()) > 3:
                return f
        except Exception:
            continue
    return None


def get_detector(model_name: str = "buffalo_l", min_score: float = 0.5):
    """Build (once) the same RetinaFace detector Immich uses."""
    global _detector
    with _detector_lock:
        if _detector is not None:
            return _detector

        from insightface.model_zoo import RetinaFace
        from insightface.utils.storage import download as download_model_pack

        # Same pack the recognition model comes from, so this is usually
        # already on disk by the time faces are detected.
        pack_dir = Path(download_model_pack("models", model_name))
        model_path = _find_detection_model(pack_dir)
        if model_path is None:
            raise RuntimeError(
                f"No detection model in the {model_name} pack at {pack_dir}. "
                "Stock face detection needs Immich's own detector."
            )

        logger.info("Loading stock face detector: %s", model_path.name)
        detector = RetinaFace(model_file=str(model_path))
        # ctx_id=-1 is CPU. onnxruntime has no Metal provider, and CoreML
        # rejects this graph's dynamic shapes, so CPU is what Docker uses here
        # too and what keeps the boxes identical.
        # det_thresh, or insightface pre-filters at its own 0.5 before our
        # min_score is ever applied, and a lower configured threshold silently
        # finds fewer faces than Docker would.
        detector.prepare(
            ctx_id=-1, det_thresh=min_score, input_size=_detector_input_size
        )
        _detector = detector
        return _detector


def unload_detector() -> None:
    global _detector
    with _detector_lock:
        _detector = None


def detect_faces(
    image_bytes: bytes, min_score: float = 0.7, model_name: str = "buffalo_l"
) -> tuple[list[dict], int, int]:
    """Detect faces the way Docker does.

    Returns the same shape as the Vision detector in face_detect.py, so the
    caller does not care which one ran: (faces, width, height), each face a
    dict with boundingBox in pixels and a score.
    """
    import cv2

    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        logger.error("Stock face detection could not decode the image")
        return [], 0, 0

    height, width = img_bgr.shape[:2]

    # Deliberately not caught. A detector that cannot load is not an image
    # with no faces in it, and returning the latter marks every asset as
    # processed with zero faces while /health, which calls this same seam,
    # reports the detector as fine. The caller turns this into a 500, which is
    # the honest answer and the one that gets looked at.
    detector = get_detector(model_name, min_score)

    # insightface returns boxes already in pixels on the original image, plus a
    # score column. It applies its own NMS, matching Immich.
    boxes, keypoints = detector.detect(img_bgr, metric="default")
    if boxes is None or len(boxes) == 0:
        return [], width, height

    faces = []
    for i, row in enumerate(boxes):
        x1, y1, x2, y2, score = row[:5]
        if score < min_score:
            continue
        face = {
            "boundingBox": {
                # Clamped and integral, like the Vision path: Immich stores
                # these against the image, and a box a pixel outside it is a
                # crash in whatever crops from it later.
                "x1": max(0, int(round(float(x1)))),
                "y1": max(0, int(round(float(y1)))),
                "x2": min(width, int(round(float(x2)))),
                "y2": min(height, int(round(float(y2)))),
            },
            "score": float(score),
        }
        if keypoints is not None and i < len(keypoints):
            face["landmarks"] = [[float(x), float(y)] for x, y in keypoints[i]]
        faces.append(face)

    return faces, width, height
