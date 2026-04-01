"""
immich-ml-metal: Metal/ANE-optimized ML service for Immich.

Drop-in replacement for Immich's ML service, optimized for Apple Silicon.
"""

from fastapi import FastAPI, Form, File, UploadFile, HTTPException, Request
from fastapi.responses import ORJSONResponse, PlainTextResponse, JSONResponse
from typing import Optional
from functools import partial
import json
import numpy as np
from PIL import Image
import io
import os
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time as _time
import threading as _threading
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field

from .config import settings

# Configure logging based on settings
settings.configure_logging()
logger = logging.getLogger(__name__)

# Use real models unless STUB_MODE is set
STUB_MODE = os.getenv("STUB_MODE", "false").lower() == "true"

# --- Model memory management ---
# Models stay loaded for fast inference. Under memory pressure, idle models
# are unloaded to prevent swap spirals. Re-loaded automatically on next request.
#
# MODEL_UNLOAD_STRATEGY:
#   "pressure" (default) — only unload when available RAM is low AND model is idle
#   "timeout"            — unload after MODEL_IDLE_TIMEOUT seconds of inactivity
#   "never"              — keep models loaded permanently
MODEL_UNLOAD_STRATEGY = os.getenv("MODEL_UNLOAD_STRATEGY", "pressure")
MODEL_IDLE_TIMEOUT = int(os.getenv("MODEL_IDLE_TIMEOUT", "120"))
MODEL_MEMORY_FLOOR_MB = int(os.getenv("MODEL_MEMORY_FLOOR_MB", "500"))
# Minimum idle time before a model is eligible for pressure-based unloading
_PRESSURE_IDLE_MIN_SEC = 30
_model_last_used: dict[str, float] = {}
_model_busy: set[str] = set()  # models currently loading or running inference
_idle_monitor_started = False


def _track_model_use(model_type: str) -> None:
    """Record that a model was just used (call AFTER load/inference, not before)."""
    _model_last_used[model_type] = _time.monotonic()
    _model_busy.discard(model_type)


def _mark_model_busy(model_type: str) -> None:
    """Mark a model as actively loading or running inference (unload-protected)."""
    _model_busy.add(model_type)


def _available_memory_mb() -> int:
    """Check available memory (free + inactive pages) via vm_stat."""
    try:
        import subprocess
        import re
        vm = subprocess.check_output(["vm_stat"], timeout=5).decode()
        ps_match = re.search(r"page size of (\d+) bytes", vm)
        page_size = int(ps_match.group(1)) if ps_match else 16384
        free_match = re.search(r"Pages free:\s+(\d+)", vm)
        inactive_match = re.search(r"Pages inactive:\s+(\d+)", vm)
        if not free_match or not inactive_match:
            return 9999
        free = int(free_match.group(1))
        inactive = int(inactive_match.group(1))
        return (free + inactive) * page_size // (1024 * 1024)
    except Exception:
        return 9999


def _should_unload(model_type: str, now: float, avail_mb: int) -> bool:
    """Decide whether a model should be unloaded based on strategy.

    avail_mb is passed in so the caller can check memory once per cycle,
    not once per model.
    """
    if model_type in _model_busy:
        return False  # model is loading or mid-inference, never unload

    last_used = _model_last_used.get(model_type, now)
    idle_sec = now - last_used

    if MODEL_UNLOAD_STRATEGY == "never":
        return False
    elif MODEL_UNLOAD_STRATEGY == "timeout":
        return idle_sec >= MODEL_IDLE_TIMEOUT
    else:  # "pressure" — default
        if idle_sec < _PRESSURE_IDLE_MIN_SEC:
            return False  # actively in use, don't touch
        return avail_mb < MODEL_MEMORY_FLOOR_MB


def _unload_model(model_type: str, reason: str) -> None:
    """Unload a specific model and log the reason."""
    try:
        if model_type == "clip":
            from .models import clip as clip_module
            with clip_module._model_lock:
                if clip_module._current_model is not None:
                    clip_module._current_model.unload()
                    clip_module._current_model = None
                    clip_module._current_model_name = None
                    logger.info("Unloaded CLIP model (%s)", reason)
            _model_last_used.pop("clip", None)
        elif model_type == "face":
            from .models import face_embed as face_module
            if face_module._recognition_model is not None:
                face_module.unload_recognition_model()
                logger.info("Unloaded face model (%s)", reason)
            _model_last_used.pop("face", None)
    except Exception as e:
        logger.warning("Model unload failed for %s: %s", model_type, e)


def _start_idle_monitor() -> None:
    """Start background thread that manages model memory."""
    global _idle_monitor_started
    if _idle_monitor_started or STUB_MODE or MODEL_UNLOAD_STRATEGY == "never":
        return
    _idle_monitor_started = True

    def _monitor():
        while True:
            _time.sleep(30)
            now = _time.monotonic()
            avail_mb = _available_memory_mb()
            for model_type in list(_model_last_used.keys()):
                if _should_unload(model_type, now, avail_mb):
                    idle_sec = now - _model_last_used.get(model_type, now)
                    if MODEL_UNLOAD_STRATEGY == "timeout":
                        reason = "idle %.0fs" % idle_sec
                    else:
                        reason = "memory pressure, %dMB available, idle %.0fs" % (
                            avail_mb, idle_sec)
                    _unload_model(model_type, reason)

    t = _threading.Thread(target=_monitor, daemon=True, name="model-memory-monitor")
    t.start()
    logger.info("Model memory monitor started (strategy=%s)", MODEL_UNLOAD_STRATEGY)


# Dedicated thread pool for ML inference — reuses threads instead of
# creating a new one per asyncio.to_thread() call.
_inference_pool = ThreadPoolExecutor(
    max_workers=settings.max_concurrent_requests, thread_name_prefix="ml-inference"
)


def _run_in_pool(fn, *args):
    """Run a sync function in the inference thread pool."""
    return asyncio.get_running_loop().run_in_executor(_inference_pool, fn, *args)


if STUB_MODE:
    logger.warning("Running in STUB_MODE - returning fake data")

# Semaphore for backpressure - limits queued requests
_request_semaphore: Optional[asyncio.Semaphore] = None


def get_request_semaphore() -> asyncio.Semaphore:
    """Get or create the request semaphore (lazy init for async context)."""
    global _request_semaphore
    if _request_semaphore is None:
        _request_semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
    return _request_semaphore


# Pydantic models for response validation
class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class FaceDetection(BaseModel):
    boundingBox: BoundingBox
    embedding: str  # Stringified array
    score: float


class OCRResult(BaseModel):
    text: list[str]
    box: list[int]  # Flat list of coordinates
    boxScore: list[float]
    textScore: list[float]


class PredictResponse(BaseModel):
    imageHeight: Optional[int] = None
    imageWidth: Optional[int] = None
    clip: Optional[str] = None  # Stringified array
    facial_recognition: Optional[list[FaceDetection]] = Field(None, alias="facial-recognition")
    ocr: Optional[OCRResult] = None


# Lifespan context for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    logger.info("Starting immich-ml-metal service")
    logger.info(f"STUB_MODE: {STUB_MODE}")
    logger.info(f"Settings: host={settings.host}, port={settings.port}")
    logger.info(f"CLIP model: {settings.clip_model}")
    logger.info(f"Face model: {settings.face_model}")
    logger.info(f"Face min score: {settings.face_min_score}")
    logger.info(f"Max concurrent requests: {settings.max_concurrent_requests}")
    logger.info(f"Log level: {settings.log_level}")
    
    _start_idle_monitor()
    yield

    # Cleanup on shutdown - import modules to access current state
    logger.info("Shutting down immich-ml-metal service")
    if not STUB_MODE:
        try:
            from .models import clip as clip_module
            from .models import face_embed as face_module
            
            # Cleanup CLIP model
            with clip_module._model_lock:
                if clip_module._current_model is not None:
                    clip_module._current_model.unload()
                    clip_module._current_model = None
                    clip_module._current_model_name = None
            
            # Cleanup face recognition model
            face_module.unload_recognition_model()
            
            logger.info("Models unloaded successfully")
        except Exception as e:
            logger.error(f"Error during model cleanup: {e}")

    _inference_pool.shutdown(wait=False)


app = FastAPI(
    title="immich-ml-metal",
    description="Metal/ANE-optimized drop-in replacement for Immich ML",
    version="0.1.0",
    lifespan=lifespan
)


# Middleware for request logging (conditional based on settings)
@app.middleware("http")
async def log_requests(request: Request, call_next):
    if settings.log_requests:
        logger.info(f"{request.method} {request.url.path}")
    response = await call_next(request)
    if settings.log_requests:
        logger.debug(f"Response status: {response.status_code}")
    return response


def get_clip(model_name: str = "ViT-B-32__openai"):
    """Get CLIP model, loading on first use or switching if model changed.

    Marks CLIP as busy (unload-protected) during load. Callers must call
    _track_model_use("clip") after inference completes to release the guard.
    """
    if STUB_MODE:
        return None
    from .models.clip import get_clip_model
    _mark_model_busy("clip")
    try:
        return get_clip_model(model_name)
    except Exception:
        _model_busy.discard("clip")  # don't leave a permanent busy flag on load failure
        raise


async def run_face_recognition_async(
    image_bytes: bytes, 
    min_score: float, 
    model_name: str
) -> list[dict]:
    """Run face detection and embedding generation (async wrapper)."""
    return await _run_in_pool(
        _run_face_recognition_sync,
        image_bytes,
        min_score,
        model_name
    )


def _run_face_recognition_sync(
    image_bytes: bytes,
    min_score: float,
    model_name: str
) -> list[dict]:
    """Synchronous face recognition implementation.

    Decodes the image once, filters by min_score, then runs a single
    batched ONNX inference for all qualifying faces.
    """
    import cv2
    from .models.face_detect import detect_faces
    from .models.face_embed import get_face_embeddings_batch
    _mark_model_busy("face")

    try:
        faces, _, _ = detect_faces(image_bytes)

        # Filter by score first — no point aligning faces we'll discard
        scored_faces = [f for f in faces if f["score"] >= min_score]
        if not scored_faces:
            return []

        # Decode image once for all faces
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            logger.error("Failed to decode image for face recognition")
            return []

        # Single batched inference
        embeddings = get_face_embeddings_batch(img_bgr, scored_faces, model_name)

        # Build results, skipping any faces where embedding failed
        results = []
        for face, embedding in zip(scored_faces, embeddings):
            if embedding is not None:
                results.append({
                    "boundingBox": face["boundingBox"],
                    "embedding": str(embedding.tolist()),
                    "score": face["score"]
                })

        return results
    finally:
        _track_model_use("face")


@app.get("/")
async def root():
    """Root endpoint - mirrors Immich ML."""
    return ORJSONResponse({"message": "Immich ML"})


@app.get("/ping")
def ping():
    """Health check endpoint."""
    return PlainTextResponse("pong")


@app.get("/health")
async def health():
    """
    Detailed health check endpoint.
    
    Checks if models can be loaded and basic functionality works.
    """
    health_status = {
        "status": "healthy",
        "stub_mode": STUB_MODE,
        "checks": {}
    }
    
    try:
        if not STUB_MODE:
            # Check CLIP model
            try:
                get_clip(settings.clip_model)
                _track_model_use("clip")
                health_status["checks"]["clip"] = "ok"
            except Exception as e:
                logger.error(f"CLIP health check failed: {e}")
                health_status["checks"]["clip"] = f"error: {str(e)}"
                health_status["status"] = "degraded"
            
            # Check face recognition model
            try:
                from .models.face_embed import get_recognition_model
                get_recognition_model(settings.face_model)
                health_status["checks"]["face_recognition"] = "ok"
            except Exception as e:
                logger.error(f"Face recognition health check failed: {e}")
                health_status["checks"]["face_recognition"] = f"error: {str(e)}"
                health_status["status"] = "degraded"
            
            # Actually test Vision framework with a minimal image
            try:
                from .models.face_detect import detect_faces
                # Create 1x1 test image
                test_img = Image.new("RGB", (1, 1), color=(128, 128, 128))
                buffer = io.BytesIO()
                test_img.save(buffer, format="JPEG")
                detect_faces(buffer.getvalue())
                health_status["checks"]["vision_framework"] = "ok"
            except Exception as e:
                logger.error(f"Vision framework health check failed: {e}")
                health_status["checks"]["vision_framework"] = f"error: {str(e)}"
                health_status["status"] = "degraded"
        else:
            health_status["checks"]["stub_mode"] = "active"
        
        return JSONResponse(content=health_status)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return JSONResponse(
            content={"status": "unhealthy", "error": str(e)},
            status_code=503
        )


@app.post("/predict")
async def predict(
    entries: str = Form(...),
    image: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
):
    """
    Main prediction endpoint - mirrors Immich ML API.
    
    Args:
        entries: JSON string describing requested tasks
        image: Optional image file for visual tasks
        text: Optional text for text encoding tasks
    
    Returns:
        ORJSONResponse with inference results
    """
    # Apply backpressure via semaphore
    semaphore = get_request_semaphore()
    try:
        # Use timeout to avoid indefinite queuing
        async with asyncio.timeout(settings.request_timeout):
            async with semaphore:
                return await _process_predict(entries, image, text)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=503,
            detail="Service overloaded, request timed out waiting in queue"
        )


async def _process_predict(
    entries: str,
    image: Optional[UploadFile],
    text: Optional[str],
) -> ORJSONResponse:
    """Internal predict processing (assumes semaphore is held)."""
    # Parse the entries JSON
    try:
        tasks = json.loads(entries)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid entries JSON: {e}")
        raise HTTPException(status_code=422, detail=f"Invalid entries JSON: {e}")
    
    if image is None and text is None:
        raise HTTPException(
            status_code=400, 
            detail="Either image or text must be provided"
        )
    
    response = {}
    
    # Read and validate image if provided
    image_bytes = None
    img = None
    if image:
        if image.size and image.size > settings.max_image_size:
            raise HTTPException(
                status_code=413,
                detail=f"Image too large. Max size: {settings.max_image_size / 1024 / 1024:.1f}MB"
            )
        
        try:
            image_bytes = await image.read()
            img = Image.open(io.BytesIO(image_bytes))
            response["imageHeight"] = img.height
            response["imageWidth"] = img.width
            
            if img.width * img.height > 100_000_000:  # 100 megapixels
                raise HTTPException(
                    status_code=413,
                    detail="Image resolution too high"
                )
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to read/decode image: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    
    # Build concurrent coroutines for each requested task.
    # CLIP hits GPU/Metal, face detection hits ANE (Vision), face embedding hits
    # CPU/CoreML (ONNX), and OCR hits ANE (Vision). Face detection and OCR share
    # the ANE but macOS schedules Vision requests internally. CLIP and face
    # embedding are on fully independent compute units.
    coroutines = []

    async def _timed(name, coro):
        """Wrap a coroutine with timing instrumentation."""
        t0 = _time.monotonic()
        result = await coro
        elapsed = (_time.monotonic() - t0) * 1000
        if result is not None:
            logger.info("  %s: %.0fms", name, elapsed)
        return result

    async def _run_clip(task_config):
        """Run CLIP embedding (visual or textual). Returns ("clip", value) or None."""
        if "visual" in task_config and image_bytes:
            model_name = task_config["visual"].get("modelName", settings.clip_model)

            if STUB_MODE:
                embedding = np.random.randn(512).astype(np.float32)
                embedding = embedding / np.linalg.norm(embedding)
            else:
                clip = get_clip(model_name)
                try:
                    embedding = await _run_in_pool(
                        clip.encode_image,
                        image_bytes
                    )
                finally:
                    _track_model_use("clip")

            return ("clip", str(embedding.tolist()))

        elif "textual" in task_config and text:
            model_name = task_config["textual"].get("modelName", settings.clip_model)

            if STUB_MODE:
                embedding = np.random.randn(512).astype(np.float32)
                embedding = embedding / np.linalg.norm(embedding)
            else:
                clip = get_clip(model_name)
                try:
                    embedding = await _run_in_pool(
                        clip.encode_text,
                        text
                    )
                finally:
                    _track_model_use("clip")

            return ("clip", str(embedding.tolist()))

        return None

    async def _run_faces(task_config):
        """Run facial recognition. Returns ("facial-recognition", value) or None."""
        if image_bytes is None:
            return None

        detection_config = task_config.get("detection", {})
        recognition_config = task_config.get("recognition", {})

        min_score = detection_config.get("options", {}).get(
            "minScore",
            settings.face_min_score
        )
        model_name = recognition_config.get("modelName", settings.face_model)

        if STUB_MODE:
            fake_embedding = np.random.randn(512).astype(np.float32).tolist()
            faces = [
                {
                    "boundingBox": {
                        "x1": int(img.width * 0.25),
                        "y1": int(img.height * 0.15),
                        "x2": int(img.width * 0.75),
                        "y2": int(img.height * 0.85)
                    },
                    "embedding": str(fake_embedding),
                    "score": 0.99
                }
            ]
        else:
            faces = await run_face_recognition_async(
                image_bytes,
                min_score,
                model_name
            )

        logger.info("  faces: %d detected", len(faces))
        return ("facial-recognition", faces)

    async def _run_ocr(task_config):
        """Run OCR text recognition. Returns ("ocr", value) or None."""
        if image_bytes is None:
            return None

        detection_config = task_config.get("detection", {})
        recognition_config = task_config.get("recognition", {})

        min_detection_score = detection_config.get("options", {}).get("minScore", 0.0)
        min_recognition_score = recognition_config.get("options", {}).get("minScore", 0.0)
        min_score = max(min_detection_score, min_recognition_score)

        if STUB_MODE:
            return ("ocr", {
                "text": ["placeholder", "text"],
                "box": [0, 0, 100, 0, 100, 50, 0, 50, 0, 50, 100, 50, 100, 100, 0, 100],
                "boxScore": [0.95, 0.92],
                "textScore": [0.98, 0.96]
            })
        else:
            from .models.ocr import recognize_text
            ocr_result = await _run_in_pool(
                partial(recognize_text, image_bytes,
                        min_confidence=min_score,
                        use_language_correction=settings.ocr_use_language_correction)
            )
            return ("ocr", ocr_result)

    for task_type, task_config in tasks.items():
        if task_type == "clip":
            coroutines.append(_timed("clip", _run_clip(task_config)))
        elif task_type == "facial-recognition":
            coroutines.append(_timed("faces", _run_faces(task_config)))
        elif task_type == "ocr":
            coroutines.append(_timed("ocr", _run_ocr(task_config)))

    # Run all tasks concurrently — they hit independent compute units.
    # Exceptions propagate: if any task fails the whole request fails,
    # matching the old sequential behaviour.
    t_start = _time.monotonic()
    results = await asyncio.gather(*coroutines)
    total_ms = (_time.monotonic() - t_start) * 1000

    task_names = [t for t in tasks.keys() if t in ("clip", "facial-recognition", "ocr")]
    for result in results:
        if result is not None:
            key, value = result
            response[key] = value

    logger.info("predict: %d task(s) [%s] completed in %.0fms",
                len(task_names), "+".join(task_names), total_ms)

    # Validate response against schema - fail loudly if validation fails
    try:
        validated_response = PredictResponse(**response)
        return ORJSONResponse(validated_response.model_dump(by_alias=True, exclude_none=True))
    except Exception as e:
        logger.error(f"Response validation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: response validation failed"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unexpected errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    # Only expose error details in debug mode (should be off for network-exposed service)
    error_detail = str(exc) if settings.debug_mode else "Internal server error"
    
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": error_detail}
    )


def main():
    """Entry point for running the service directly."""
    import uvicorn
    
    logger.info(f"Starting server on {settings.host}:{settings.port}")
    
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level=settings.log_level.lower(),
        timeout_keep_alive=settings.request_timeout,
    )


if __name__ == "__main__":
    main()