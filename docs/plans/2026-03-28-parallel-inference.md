# Parallel Inference & Performance Optimization

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Maximize throughput by running ML tasks concurrently across Apple Silicon's independent compute units (GPU, ANE, CPU) and batching face embeddings.

**Architecture:** Three independent optimizations that compound: (1) run CLIP/faces/OCR concurrently within a single /predict request via asyncio.gather, (2) batch all face embeddings into a single ONNX inference call instead of per-face, (3) move CLIP image prep outside the inference lock so prep work doesn't block GPU access. All changes are in the ml submodule (epheterson/immich-ml-metal).

**Tech Stack:** Python asyncio, ONNX Runtime (InsightFace), MLX, Apple Vision framework

---

## Hardware Map (Why This Works)

| Task | Compute Unit | Framework |
|------|-------------|-----------|
| CLIP encode | **GPU** (Metal) | MLX / open_clip MPS |
| Face detection | **ANE** | Apple Vision |
| Face embedding | **CPU/CoreML** | InsightFace ONNX |
| OCR | **ANE** | Apple Vision |

These are independent silicon — GPU work doesn't block ANE, ANE doesn't block CPU.
CLIP + face detection + face embedding can all run simultaneously.
Face detection and OCR both use ANE so they'll contend there, but the OS handles that.

---

### Task 1: Concurrent Task Execution (asyncio.gather)

**Files:**
- Modify: `src/main.py:470-572` (the sequential `for task_type` loop in `_process_predict`)

**Context:** Currently `_process_predict` iterates tasks sequentially — CLIP finishes, then faces, then OCR. Each `await asyncio.to_thread(...)` yields to the event loop but the next task doesn't start until the current one returns. With `asyncio.gather()`, all three launch simultaneously.

**Step 1: Refactor _process_predict to collect coroutines and gather them**

Replace the sequential `for task_type, task_config in tasks.items():` loop (lines 470-572) with a pattern that:

1. Builds a list of coroutines for each requested task
2. Runs them concurrently with `asyncio.gather()`
3. Merges results into the response dict

```python
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

    # Build coroutines for each task — they'll run concurrently
    async def _run_clip(task_config):
        if "visual" in task_config and image_bytes:
            model_name = task_config["visual"].get("modelName", settings.clip_model)

            if STUB_MODE:
                embedding = np.random.randn(512).astype(np.float32)
                embedding = embedding / np.linalg.norm(embedding)
            else:
                clip = get_clip(model_name)
                try:
                    embedding = await asyncio.to_thread(
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
                    embedding = await asyncio.to_thread(
                        clip.encode_text,
                        text
                    )
                finally:
                    _track_model_use("clip")

            return ("clip", str(embedding.tolist()))
        return None

    async def _run_faces(task_config):
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

        return ("facial-recognition", faces)

    async def _run_ocr(task_config):
        if image_bytes is None:
            return None

        detection_config = task_config.get("detection", {})
        recognition_config = task_config.get("recognition", {})

        min_detection_score = detection_config.get("options", {}).get("minScore", 0.0)
        min_recognition_score = recognition_config.get("options", {}).get("minScore", 0.0)
        min_score = max(min_detection_score, min_recognition_score)

        if STUB_MODE:
            result = {
                "text": ["placeholder", "text"],
                "box": [0, 0, 100, 0, 100, 50, 0, 50, 0, 50, 100, 50, 100, 100, 0, 100],
                "boxScore": [0.95, 0.92],
                "textScore": [0.98, 0.96]
            }
        else:
            from .models.ocr import recognize_text
            result = await asyncio.to_thread(
                recognize_text,
                image_bytes,
                min_confidence=min_score,
                use_language_correction=settings.ocr_use_language_correction
            )

        return ("ocr", result)

    # Dispatch tasks concurrently
    coroutines = []
    for task_type, task_config in tasks.items():
        if task_type == "clip":
            coroutines.append(_run_clip(task_config))
        elif task_type == "facial-recognition":
            coroutines.append(_run_faces(task_config))
        elif task_type == "ocr":
            coroutines.append(_run_ocr(task_config))

    results = await asyncio.gather(*coroutines)

    for result in results:
        if result is not None:
            key, value = result
            response[key] = value

    # Validate response against schema
    try:
        validated_response = PredictResponse(**response)
        return ORJSONResponse(validated_response.model_dump(by_alias=True, exclude_none=True))
    except Exception as e:
        logger.error(f"Response validation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: response validation failed"
        )
```

**Step 2: Smoke test**

Run: `cd /Users/elp/Repos/immich-apple-silicon/ml && STUB_MODE=true python3 -c "import asyncio; from src.main import app"`
Expected: No import errors

**Step 3: Commit**

```bash
git add src/main.py
git commit -m "perf: concurrent task execution via asyncio.gather

CLIP (GPU), face detection (ANE), face embedding (CPU/CoreML), and OCR
(ANE) now run concurrently instead of sequentially within a single
/predict request. These hit independent compute units on Apple Silicon."
```

---

### Task 2: Face Embedding Batching

**Files:**
- Modify: `src/models/face_embed.py:204-256` (get_face_embedding)
- Modify: `src/main.py:282-314` (_run_face_recognition_sync)

**Context:** Currently each face in an image gets its own `cv2.imdecode` + `model.get_feat()` call. For a group photo with 10 faces, that's 10 full image decodes and 10 separate ONNX inferences. The ONNX model accepts batch input (batch, 3, 112, 112).

**Step 1: Add batch embedding function to face_embed.py**

Add after `get_face_embedding` (after line 256):

```python
def get_face_embeddings_batch(
    img_bgr: np.ndarray,
    faces: list[dict],
    model_name: str = "buffalo_l"
) -> list[Optional[np.ndarray]]:
    """
    Generate embeddings for multiple faces in a single batch inference.

    Takes a pre-decoded BGR image and a list of face dicts (with landmarks
    or boundingBox). Returns embeddings in the same order as input faces.
    Significantly faster than per-face calls for images with multiple faces.

    Args:
        img_bgr: Pre-decoded image in BGR format (cv2)
        faces: List of face dicts with 'landmarks' and/or 'boundingBox'
        model_name: InsightFace model to use

    Returns:
        List of 512-dim normalized embeddings (or None for failed faces)
    """
    from insightface.utils import face_align

    if not faces:
        return []

    model = get_recognition_model(model_name)
    aligned_faces = []
    face_indices = []  # track which input faces succeeded alignment

    for i, face in enumerate(faces):
        try:
            if "landmarks" in face:
                kps = np.array(face["landmarks"], dtype=np.float32)
                aligned = face_align.norm_crop(img_bgr, kps, image_size=ARCFACE_INPUT_SIZE)
            elif "boundingBox" in face:
                bbox = face["boundingBox"]
                x1, y1 = int(bbox["x1"]), int(bbox["y1"])
                x2, y2 = int(bbox["x2"]), int(bbox["y2"])
                w, h = x2 - x1, y2 - y1
                pad_x, pad_y = int(w * 0.1), int(h * 0.1)
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(img_bgr.shape[1], x2 + pad_x)
                y2 = min(img_bgr.shape[0], y2 + pad_y)
                face_crop = img_bgr[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue
                aligned = cv2.resize(face_crop, (ARCFACE_INPUT_SIZE, ARCFACE_INPUT_SIZE))
            else:
                continue

            aligned_faces.append(aligned)
            face_indices.append(i)
        except Exception as e:
            logger.warning(f"Face alignment failed for face {i}: {e}")

    if not aligned_faces:
        return [None] * len(faces)

    # Batch inference — single ONNX call for all faces
    with _inference_lock:
        try:
            batch = np.stack(aligned_faces)
            embeddings_raw = model.get_feat(batch)
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            return [None] * len(faces)

    # Normalize and map back to input order
    results: list[Optional[np.ndarray]] = [None] * len(faces)
    for batch_idx, face_idx in enumerate(face_indices):
        emb = embeddings_raw[batch_idx].flatten()
        emb = emb / np.linalg.norm(emb)
        results[face_idx] = emb.astype(np.float32)

    return results
```

**Step 2: Update _run_face_recognition_sync to use batch function**

Replace the per-face loop in `src/main.py:282-314`:

```python
def _run_face_recognition_sync(
    image_bytes: bytes,
    min_score: float,
    model_name: str
) -> list[dict]:
    """Synchronous face recognition implementation."""
    from .models.face_detect import detect_faces
    from .models.face_embed import get_face_embeddings_batch
    _mark_model_busy("face")

    try:
        faces, _, _ = detect_faces(image_bytes)

        # Filter by score first
        scored_faces = [f for f in faces if f["score"] >= min_score]
        if not scored_faces:
            return []

        # Decode image once for all faces
        nparr = np.frombuffer(image_bytes, np.uint8)
        import cv2
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            logger.error("Failed to decode image for face embedding")
            return []

        # Batch embed all faces in a single ONNX call
        embeddings = get_face_embeddings_batch(img_bgr, scored_faces, model_name)

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
```

**Step 3: Smoke test**

Run: `cd /Users/elp/Repos/immich-apple-silicon/ml && STUB_MODE=true python3 -c "from src.models.face_embed import get_face_embeddings_batch; print('import ok')"`
Expected: `import ok`

**Step 4: Commit**

```bash
git add src/main.py src/models/face_embed.py
git commit -m "perf: batch face embeddings in single ONNX inference

Decode image once, align all faces, run single batched get_feat() call.
For a photo with N faces: N image decodes + N inferences -> 1 decode + 1 inference."
```

---

### Task 3: CLIP Image Prep Outside Inference Lock

**Files:**
- Modify: `src/models/clip.py:233-258` (encode_image method)

**Context:** The entire `encode_image` method runs under `_inference_lock` — including PIL image decode, JPEG encoding to temp file, and temp file cleanup. Only the actual MLX `image_encoder()` call needs the lock. Moving prep work outside means the next request's image can be prepped while the current one is on the GPU.

**Step 1: Restructure encode_image to minimize lock scope**

```python
def encode_image(self, image_bytes: bytes) -> np.ndarray:
    """
    Generate CLIP embedding for an image.
    Thread-safe - only GPU inference is serialized.
    """
    if not self._loaded:
        raise RuntimeError("Model not loaded")

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    if hasattr(self, '_use_fallback') and self._use_fallback:
        with self._inference_lock:
            return self._encode_image_fallback(image)

    # Prep temp file outside lock — this is I/O, not GPU
    buffer_mgr = get_buffer_manager()
    temp_path, _ = buffer_mgr.get_image_path(image)

    try:
        # Only hold lock for actual GPU inference
        with self._inference_lock:
            embedding = self._model.image_encoder(temp_path)

        if isinstance(embedding, mx.array):
            embedding = np.array(embedding)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.flatten().astype(np.float32)
    finally:
        buffer_mgr.release_path(temp_path)
```

Note: The fallback path (open_clip/torch) still needs the full lock because `_encode_image_fallback` uses `self._processor` and `self._model` which aren't thread-safe with torch. Only the MLX path benefits from the narrower lock.

**Step 2: Smoke test**

Run: `cd /Users/elp/Repos/immich-apple-silicon/ml && python3 -c "from src.models.clip import MLXClip; print('import ok')"`
Expected: `import ok`

**Step 3: Commit**

```bash
git add src/models/clip.py
git commit -m "perf: CLIP image prep outside inference lock

Image decode, JPEG encode, and temp file creation now happen outside
the GPU lock. Only the actual MLX image_encoder() call is serialized.
Allows next request's image prep to overlap with current GPU inference."
```

---

### Task 4: Final Commit & PR

**Step 1: Verify all changes work together**

Run the service in stub mode and hit the predict endpoint:
```bash
cd /Users/elp/Repos/immich-apple-silicon/ml
STUB_MODE=true python3 -c "
import asyncio, json
from src.main import app
from httpx import AsyncClient, ASGITransport

async def test():
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as client:
        # Test with all three tasks
        from PIL import Image
        import io
        img = Image.new('RGB', (100, 100), color='red')
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        buf.seek(0)
        entries = json.dumps({
            'clip': {'visual': {'modelName': 'ViT-B-32__openai'}},
            'facial-recognition': {'detection': {}, 'recognition': {}},
            'ocr': {'detection': {}, 'recognition': {}}
        })
        resp = await client.post('/predict', data={'entries': entries}, files={'image': ('test.jpg', buf, 'image/jpeg')})
        print(f'Status: {resp.status_code}')
        data = resp.json()
        print(f'Keys: {sorted(data.keys())}')
        assert 'clip' in data
        assert 'facial-recognition' in data
        assert 'ocr' in data
        print('All tasks returned results concurrently')

asyncio.run(test())
"
```

**Step 2: Push and create PR**

```bash
cd /Users/elp/Repos/immich-apple-silicon/ml
git push origin main
```

Then update the submodule pointer in the parent repo if desired.

---

## Expected Impact

| Scenario | Before | After | Why |
|----------|--------|-------|-----|
| Single image, all 3 tasks | ~serial sum | ~max(CLIP, faces+OCR) | gather() runs GPU + ANE concurrently |
| Group photo, 10 faces | 10 decodes + 10 inferences | 1 decode + 1 inference | Batched ONNX call |
| Back-to-back CLIP requests | Blocked during image prep | Prep overlaps with GPU | Narrower inference lock |

The biggest win is Task 1 (concurrent tasks) for requests that include multiple task types. Task 2 (face batching) is the biggest win for group photos during library imports.
