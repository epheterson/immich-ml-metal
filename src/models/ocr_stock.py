"""
Text recognition with Immich's own ONNX models, for output parity with Docker.

Apple's Vision framework (ocr.py) is faster and generally reads text better on
this hardware. It is also a completely different recogniser: different boxes,
different strings, different confidence scale. That is fine until someone needs
a library that can move back to a Docker deployment without re-running OCR over
everything, which is what the Stock position exists for.

This runs what Immich runs: RapidOCR's PP-OCRv5 detection and recognition
models through onnxruntime, with the same preprocessing, the same DBPostProcess
parameters and the same reading order as
machine-learning/immich_ml/models/ocr/{detection,recognition}.py. The heavy
parts (CTC decoding, the character dictionary, model downloads) come from the
`rapidocr` package Immich itself depends on, rather than being reimplemented
here, so the two stay in step as it is updated.
"""

import logging
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

_detector = None
_recognizer = None
_lock = threading.Lock()

# Immich's values. Not defaults: DBPostProcess ships different ones, and these
# decide which text regions survive, so a mismatch changes the output.
_MAX_RESOLUTION = 736
_DB_THRESH = 0.3
_DB_UNCLIP_RATIO = 1.6
_DB_MAX_CANDIDATES = 1000

class _OcrOptions(dict):
    """What rapidocr's TextRecognizer expects: a mapping that also answers to
    attribute access. Mirrors immich_ml/models/ocr/schemas.py OcrOptions."""

    def __init__(self, lang_type=None, **options):
        super().__init__(**options)
        from rapidocr.utils.typings import EngineType, LangRec

        self.engine_type = EngineType.ONNXRUNTIME
        self.lang_type = lang_type if lang_type is not None else LangRec.CH
        self.font_path = None


_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_STD_INV = np.float32(1.0) / (np.array([0.5, 0.5, 0.5], dtype=np.float32) * 255.0)


def _model_path(task: str, models_dir: Path) -> Path:
    """Where a downloaded PP-OCRv5 model lives, and fetch it if it is missing.

    Uses rapidocr's own resolver and downloader, so the file and its checksum
    are whichever ones Immich would have used for the same task.
    """
    from rapidocr.inference_engine.base import FileInfo, InferSession
    from rapidocr.utils.download_file import DownloadFile, DownloadFileInput
    from rapidocr.utils.typings import (
        EngineType,
        LangDet,
        LangRec,
        OCRVersion,
        TaskType,
    )
    from rapidocr.utils.typings import ModelType as RapidModelType

    if task == "det":
        info = FileInfo(
            engine_type=EngineType.ONNXRUNTIME,
            ocr_version=OCRVersion.PPOCRV5,
            task_type=TaskType.DET,
            lang_type=LangDet.CH,
            model_type=RapidModelType.MOBILE,
        )
    else:
        info = FileInfo(
            engine_type=EngineType.ONNXRUNTIME,
            ocr_version=OCRVersion.PPOCRV5,
            task_type=TaskType.REC,
            lang_type=LangRec.CH,
            model_type=RapidModelType.MOBILE,
        )

    model_info = InferSession.get_model_url(info)
    url = model_info["model_dir"]
    target = models_dir / "ocr-stock" / f"{task}-{Path(url).name}"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading stock OCR %s model", task)
        DownloadFile.run(
            DownloadFileInput(
                file_url=url,
                sha256=model_info["SHA256"],
                save_path=target,
                logger=logger,
            )
        )
    return target


def _get_models(models_dir: Path):
    """Build both sessions once. Loading is slow and the service is threaded."""
    global _detector, _recognizer
    with _lock:
        if _detector is not None and _recognizer is not None:
            return _detector, _recognizer

        import onnxruntime as ort
        from rapidocr.ch_ppocr_det.utils import DBPostProcess
        from rapidocr.ch_ppocr_rec import TextRecognizer

        det_path = _model_path("det", models_dir)
        rec_path = _model_path("rec", models_dir)

        # CPU, like Docker. CoreML rejects these graphs' dynamic shapes, and
        # the point of this path is matching output rather than being quick.
        session = ort.InferenceSession(
            str(det_path), providers=["CPUExecutionProvider"]
        )
        postprocess = DBPostProcess(
            thresh=_DB_THRESH,
            box_thresh=0.5,
            max_candidates=_DB_MAX_CANDIDATES,
            unclip_ratio=_DB_UNCLIP_RATIO,
            use_dilation=True,
            score_mode="fast",
        )
        # rapidocr's recogniser wants a config that answers both as a mapping
        # and as an object (cfg["rec_batch_num"] and cfg.engine_type), and it
        # takes an already-built session rather than a path. Immich solves this
        # with a dict subclass carrying the attributes; same shape here, same
        # values, so the batching and the input geometry match.
        rec_session = ort.InferenceSession(
            str(rec_path), providers=["CPUExecutionProvider"]
        )
        recognizer = TextRecognizer(_OcrOptions(
            session=rec_session,
            rec_batch_num=6,
            rec_img_shape=(3, 48, 320),
        ))
        _detector = (session, postprocess)
        _recognizer = recognizer
        return _detector, _recognizer


def unload_models() -> None:
    global _detector, _recognizer
    with _lock:
        _detector = None
        _recognizer = None


def _transform(img) -> np.ndarray:
    """Immich's detection preprocessing, exactly.

    The 32-pixel rounding matters: the detector is fully convolutional and a
    size that is not a multiple of 32 shifts every box it returns.
    """
    import cv2
    from PIL import Image

    if img.height < img.width:
        ratio = float(_MAX_RESOLUTION) / img.height
    else:
        ratio = float(_MAX_RESOLUTION) / img.width
    ratio = min(ratio, 1.0)

    resize_h = int(round(int(img.height * ratio) / 32) * 32)
    resize_w = int(round(int(img.width * ratio) / 32) * 32)
    resized = img.resize((resize_w, resize_h), resample=Image.Resampling.LANCZOS)

    arr = cv2.cvtColor(np.array(resized, dtype=np.float32), cv2.COLOR_RGB2BGR)
    arr -= _MEAN
    arr *= _STD_INV
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, axis=0)


def _sorted_boxes(boxes: np.ndarray) -> np.ndarray:
    """Immich's reading order: top to bottom, then left to right within a line.

    Lines are grouped by a 10-pixel jump in y, which is what makes two boxes
    side by side come back in the order a person would read them rather than in
    detector order.
    """
    if len(boxes) == 0:
        return boxes
    y_order = np.argsort(boxes[:, 0, 1], kind="stable")
    sorted_y = boxes[y_order, 0, 1]
    line_ids = np.empty(len(boxes), dtype=np.int32)
    line_ids[0] = 0
    np.cumsum(np.abs(np.diff(sorted_y)) >= 10, out=line_ids[1:])
    sort_key = line_ids * 1e6 + boxes[y_order, 0, 0]
    return boxes[y_order[np.argsort(sort_key, kind="stable")]]


def recognize_text(
    image_bytes: bytes,
    models_dir: Path,
    min_detection_score: float = 0.5,
    min_recognition_score: float = 0.5,
) -> dict[str, Any]:
    """Read text the way Docker does.

    Returns the same shape as the Vision path in ocr.py, so the caller does not
    care which recogniser ran.
    """
    import cv2
    from PIL import Image
    import io

    empty: dict[str, Any] = {"text": [], "box": [], "boxScore": [], "textScore": []}

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        logger.error("Stock OCR could not decode the image: %s", e)
        return empty

    if img.width < 32 or img.height < 32:
        return empty

    try:
        (session, postprocess), recognizer = _get_models(models_dir)
    except Exception as e:
        logger.error("Stock OCR models unavailable: %s", e)
        return empty

    postprocess.box_thresh = min_detection_score

    out = session.run(None, {"x": _transform(img)})[0]
    boxes, box_scores = postprocess(out, (img.height, img.width))
    if boxes is None or len(boxes) == 0:
        return empty

    boxes = _sorted_boxes(np.asarray(boxes, dtype=np.float32))

    # Crop each detected quadrilateral and hand the crops to the recogniser in
    # one batch, which is what Immich does and what keeps the per-image cost
    # from scaling with the number of lines.
    from rapidocr.ch_ppocr_rec import TextRecInput

    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    crops = [_crop_quad(img_bgr, box) for box in boxes]
    crops = [c for c in crops if c is not None and c.size > 0]
    if not crops:
        return empty

    result = recognizer(TextRecInput(img=crops))
    texts = list(getattr(result, "txts", []) or [])
    scores = list(getattr(result, "scores", []) or [])

    kept_text, kept_box, kept_box_score, kept_text_score = [], [], [], []
    for i, text in enumerate(texts):
        score = float(scores[i]) if i < len(scores) else 0.0
        if not text or score < min_recognition_score:
            continue
        kept_text.append(text)
        kept_box.append([int(v) for v in np.asarray(boxes[i]).flatten().tolist()])
        kept_box_score.append(float(box_scores[i]) if i < len(box_scores) else 0.0)
        kept_text_score.append(score)

    return {
        "text": kept_text,
        "box": kept_box,
        "boxScore": kept_box_score,
        "textScore": kept_text_score,
    }


def _crop_quad(img_bgr: np.ndarray, box: np.ndarray) -> Optional[np.ndarray]:
    """Rectify one detected quadrilateral to an upright crop.

    Text regions come back as four corners, not rectangles, so a rotated line
    has to be warped rather than sliced or the recogniser reads it at an angle.
    """
    import cv2

    pts = np.asarray(box, dtype=np.float32).reshape(4, 2)
    width = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
    height = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
    if width < 1 or height < 1:
        return None
    dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(pts, dst)
    crop = cv2.warpPerspective(img_bgr, matrix, (width, height))
    # Tall, narrow crops are vertical text; the recogniser expects it upright.
    if height > width * 1.5:
        crop = np.rot90(crop)
    return np.ascontiguousarray(crop)
