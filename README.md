# immich-ml-metal

A Metal/ANE-accelerated drop-in replacement for [Immich's](https://immich.app/) machine learning service, built for Apple Silicon Macs.

> **Heads up:** I'm not a software developer. This was architected and largely written by Claude against my requirements, and tested only in my own home setup. Treat it as an experimental community project, not production software. Contributions from actual developers are very welcome.

## What it does

Immich's standard ML container runs well on NVIDIA, Intel, and AMD GPUs but has been hard to run natively on Apple's frameworks (especially since OCR was added). This service speaks the same ML API as Immich's, but routes each task to a native Apple framework:

| Task | Compute unit | Framework |
|------|-------------|-----------|
| CLIP embeddings | GPU (Metal) | MLX, with open_clip/MPS fallback |
| Face detection | ANE | Apple Vision |
| Face recognition | CPU / CoreML | InsightFace ArcFace (ONNX) |
| OCR | ANE | Apple Vision |

## Performance

Within a single `/predict` request, CLIP, face recognition, and OCR are dispatched together via `asyncio.gather`, so the Vision-framework tasks (face detection, OCR) and the ONNX face-embedding step can overlap. Face embeddings for all faces in a photo are batched into a single ONNX call.

One caveat worth knowing: MLX and Apple's Vision framework both submit Metal work, and concurrent access crashes the process, so CLIP's GPU inference is serialized behind a global Metal lock. CLIP and the Vision tasks therefore don't run fully in parallel — but the batching and the overlap of the non-MLX work still add up.

Per-task timing is logged on every request:
```
INFO:   clip: 25ms
INFO:   faces: 3 detected
INFO:   faces: 135ms
INFO:   ocr: 47ms
INFO: predict: 3 task(s) [clip+facial-recognition+ocr] completed in 135ms
```

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4 — Intel not supported)
- macOS Tahoe (26) or later — may work on earlier, untested
- Python 3.11 (3.13 lacks some required wheels)
- Xcode Command Line Tools (`xcode-select --install`) — `insightface` builds from source and needs a compiler
- A running Immich server (this replaces only the ML service)

## Installation

```bash
git clone https://github.com/sebastianfredette/immich-ml-metal.git
cd immich-ml-metal

# Use Python 3.11 — 3.13 doesn't yet have all required wheels
python3.11 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

# First run downloads models
python -m src.main
```

`mlx-clip` is already pinned to a known-good commit in `requirements.txt`. Bump that pin yourself only if you want a newer version.

## Configuration

### Choosing models — do this in Immich, not here

CLIP and face recognition models are selected in **Immich's admin panel** (Administration → Settings → Machine Learning), and Immich sends the chosen model with every request. This service uses whatever each request specifies. The `ML_CLIP_MODEL` / `ML_FACE_MODEL` environment variables below are only *fallback defaults* for the rare request that omits a model — setting them will **not** override what you've configured in Immich. If you want a different model, change it in Immich.

What this service controls is which models are *supported* and how each one is loaded.

### Supported CLIP models

The model name Immich sends is resolved through three lookup tables in `src/models/clip.py` (`MODEL_MAP`, `HF_REPO_MAP`, `OPENCLIP_MAP`), and a given name can end up loaded one of four ways:

1. **Prebuilt MLX** — a ready-made `mlx-community` port exists; loaded directly via MLX (fastest path, no conversion).
2. **Converted MLX** — no prebuilt port, but a HuggingFace source is known. Converted to MLX on **first use** and cached under `~/.cache/immich-ml-metal/mlx-models/<name>` (~1–2GB download, a few minutes — happens once per model per machine, so the first request after selecting such a model is slow).
3. **open_clip fallback** — no MLX path at all (e.g. SigLIP); runs through open_clip on MPS instead. Works, but slower than MLX.
4. **Default** — an unrecognized name silently falls back to `ViT-B-32` (prebuilt MLX).

Note also: if `mlx_clip` can't be imported for any reason, **everything** routes to the open_clip fallback regardless of the above.

| Immich model name | How it loads |
|-------------------|--------------|
| `ViT-B-32__openai` | Prebuilt MLX |
| `ViT-B-16__openai` | Prebuilt MLX |
| `ViT-B-32__laion2b-s34b-b79k` | Prebuilt MLX |
| `ViT-B-32__laion2b_s34b_b79k` | Prebuilt MLX (alias of the above) |
| `ViT-L-14__openai` | Converted MLX on first use |
| `ViT-B-16-SigLIP__webli` | open_clip fallback |
| `ViT-B-16-SigLIP2__webli` | open_clip fallback |
| `ViT-L-16-SigLIP2-256__webli` | open_clip fallback |
| `ViT-SO400M-16-SigLIP2-384__webli` | open_clip fallback |
| anything else | Default (`ViT-B-32`) |

This table reflects the maps at time of writing — `src/models/clip.py` is the source of truth if they've diverged.

### Supported face models

`buffalo_s`, `buffalo_m`, `buffalo_l` (InsightFace ArcFace, via ONNX + CoreML).

### Environment variables

Set via environment, or edit defaults in `src/config.py` (the full list lives there). The ones you're most likely to touch:

| Variable | Default | Description |
|----------|---------|-------------|
| `ML_HOST` | `0.0.0.0` | Bind address |
| `ML_PORT` | `3003` | Port (must match Immich's `MACHINE_LEARNING_URL`) |
| `ML_FACE_MIN_SCORE` | `0.7` | Face detection confidence threshold |
| `ML_OCR_LANGUAGE_CORRECTION` | `true` | Disable for serials/codes |
| `MODEL_UNLOAD_STRATEGY` | `pressure` | `pressure` (unload idle models when RAM low), `timeout`, or `never` |
| `ML_LOG_LEVEL` | `INFO` | `DEBUG`/`INFO`/`WARNING`/`ERROR` |
| `ML_CLIP_MODEL` | `ViT-B-32__openai` | *Fallback* CLIP model (overridden per-request by Immich) |
| `ML_FACE_MODEL` | `buffalo_l` | *Fallback* face model (overridden per-request by Immich) |

## Connecting to Immich

In your Immich `docker-compose.yml` or `.env`, point the ML URL at your Mac:

```yaml
MACHINE_LEARNING_URL=http://192.168.1.100:3003
```

## Running the service

```bash
source .venv/bin/activate
python -m uvicorn src.main:app --host 0.0.0.0 --port 3003 --workers 1
```

### As a persistent service (launchd)

Create `~/Library/LaunchAgents/com.immich.ml.plist`, adjusting the paths:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.immich.ml</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/immich-ml-metal/.venv/bin/python</string>
        <string>-m</string>
        <string>uvicorn</string>
        <string>src.main:app</string>
        <string>--host</string>
        <string>0.0.0.0</string>
        <string>--port</string>
        <string>3003</string>
        <string>--workers</string>
        <string>1</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/path/to/immich-ml-metal</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/path/to/logs/immich-ml.log</string>
    <key>StandardErrorPath</key>
    <string>/path/to/logs/immich-ml-error.log</string>
</dict>
</plist>
```

```bash
launchctl load ~/Library/LaunchAgents/com.immich.ml.plist
launchctl start com.immich.ml
```

A LaunchAgent (under `~/Library/LaunchAgents`) runs in your user's GUI session, which the Apple frameworks need. Note that macOS doesn't rotate these logs.

> **Cold-start note:** CoreML compiles the face model on the first request after each restart (~6–7s). ONNX Runtime's `ModelCacheDirectory` option is currently ignored due to an [upstream bug](https://github.com/microsoft/onnxruntime/issues/23228), so this recompile isn't yet persisted across restarts.

## Verification

```bash
curl http://localhost:3003/ping            # -> pong
curl http://localhost:3003/health           # component health check

curl -X POST http://localhost:3003/predict \
  -F 'entries={"clip":{"textual":{"modelName":"ViT-B-32__openai"}}}' \
  -F 'text=a photo of a cat'
```

In Immich's admin logs you should see the ML service connect.

## Status & contributing

Alpha quality — use at your own risk. Tested on a MacBook Pro M1 (8GB) and an M4 Mac mini (16GB), macOS 26.1, Immich v2.4.1. Not all Immich ML features are guaranteed compatible, and memory use isn't heavily optimized.

Contributions especially welcome from developers who can review code quality, add tests, verify Immich compatibility, or improve docs. No guarantees — this is a hobby project.
