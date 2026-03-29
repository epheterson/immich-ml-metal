# immich-ml-metal

> **⚠️ Important Disclaimer**: I (the repository owner) am not a software developer. This project was architected, designed, and primarily authored by Claude based on my requirements and feedback. While I've tested it in my home environment, please treat this as an **experimental community project** rather than production-ready software. I'm sharing this in hopes it's useful for the Mac community. If it's not helpful for you, please don't feel pressured to use it.

A Metal/ANE-optimized drop-in replacement for [Immich's](https://immich.app/) machine learning service, designed specifically for Apple Silicon Macs. This allows Mac users to run Immich's ML workloads natively on their hardware.

## What This Does

Immich's standard ML container runs well on NVIDIA, Intel, and AMD GPUs. Recently, the community has had trouble running it natively on Apple's ML framework (particularly after OCR was implemented). This project is a drop-in replacement for Immich-ML that uses the same ML API but uses as many native Apple ML frameworks as possible:

- **CLIP Embeddings**: MLX-accelerated (with open_clip fallback) for image/text search
- **Face Detection**: Apple Vision framework (runs on Neural Engine)
- **Face Recognition**: InsightFace ArcFace with CoreML acceleration
- **OCR**: Apple Vision framework text recognition

## Performance: Why This is Fast

Apple Silicon has three independent compute units — GPU (Metal), Neural Engine (ANE), and CPU. This service runs ML tasks across all three concurrently:

| Task | Compute Unit | Framework |
|------|-------------|-----------|
| CLIP embedding | GPU (Metal) | MLX / open_clip MPS |
| Face detection | ANE | Apple Vision |
| Face embedding | CPU / CoreML | InsightFace ONNX |
| OCR | ANE | Apple Vision |

Within a single `/predict` request, CLIP, face recognition, and OCR run simultaneously via `asyncio.gather`. Face embeddings are batched into a single ONNX inference call regardless of how many faces are in the photo.

**Benchmarks (M4, 24GB):**

| Photo | Faces | Latency |
|-------|-------|---------|
| 3.5MB portrait | 0 | 134ms |
| 5.5MB landscape | 0 | 149ms |
| 26MB group photo | 3 | 602ms |
| 23MB group photo | 5 | 589ms |

Per-task timing is logged on every request:
```
INFO:   clip: 25ms
INFO:   faces: 3 detected
INFO:   faces: 135ms
INFO:   ocr: 47ms
INFO: predict: 3 task(s) [clip+facial-recognition+ocr] completed in 135ms
```

## Project Status

** A(I)lpha Quality - Use at Your Own Risk**

- [x] CLIP implementation (MLX + open_clip fallback)
- [x] Face detection (Vision framework)
- [x] Face embeddings (InsightFace + CoreML)
- [x] OCR (Vision framework)
- [x] Basic integration testing with real Immich instance
- [ ] Community testing and validation

**Known Limitations:**
- Only tested in my specific home setup (MacBook Pro M1, macOS 26.1, Immich v2.4.1)
- Not all Immich ML features may be fully compatible
- Memory usage not extensively optimized
- No load testing performed
- Could literally only work on my machine(tm)

## Requirements

- **macOS Tahoe+** (Might work on earlier OSs, but you'd have to test it yourself)
- **Apple Silicon Mac** (M1/M2/M3/M4 - Intel Macs not supported)
- **Python 3.11** (Not working on 3.13)
- **Immich server** already running (this replaces just the ML service)

## Installation

```bash
# Clone the repository
git clone https://github.com/sebastianfredette/immich-ml-metal.git
cd immich-ml-metal

# Pin mlx-clip to a specific commit (for stability)
# Get the current commit hash:
git ls-remote https://github.com/harperreed/mlx_clip.git HEAD
# Edit requirements.txt and replace the tail end of the mlx_clip.git address with the new hash

# Create and activate virtual environment
# Please ensure python 3.11 is used — 3.13 doesn't yet have all required wheels
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# First run will download models
python -m src.main
```

## Configuration

Configure via environment variables or edit `src/config.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `ML_HOST` | `0.0.0.0` | Bind address |
| `ML_PORT` | `3003` | Port number (must match Immich config) |
| `ML_MODELS_DIR` | `./models` | Model storage directory |
| `ML_CLIP_MODEL` | `ViT-B-32__openai` | CLIP model name |
| `ML_CLIP_BUFFER_RAM_MB` | `256` | Max RAM for image encode queue (MB) |
| `ML_FACE_MODEL` | `buffalo_l` | Face recognition model (buffalo_s/m/l) |
| `ML_FACE_MIN_SCORE` | `0.7` | Face detection confidence threshold |
| `ML_OCR_LANGUAGE_CORRECTION` | `true` | Language correction for OCR (disable for codes/serials) |
| `ML_USE_COREML` | `true` | Enable CoreML acceleration |
| `ML_USE_ANE` | `true` | Enable Apple Neural Engine |
| `ML_MAX_CONCURRENT_REQUESTS` | `4` | Max queued requests before backpressure |
| `MODEL_UNLOAD_STRATEGY` | `pressure` | `pressure`: unload when RAM is low + idle. `timeout`: unload after idle timeout. `never`: keep loaded. |
| `MODEL_IDLE_TIMEOUT` | `120` | Seconds before unloading idle models (only used with `timeout` strategy) |
| `MODEL_MEMORY_FLOOR_MB` | `500` | Available RAM threshold that triggers model unloading (only used with `pressure` strategy) |
| `ML_LOG_LEVEL` | `INFO` | Logging verbosity (DEBUG/INFO/WARNING/ERROR) |
| `ML_LOG_REQUESTS` | `true` | Log individual requests (disable for high volume) |
| `ML_DEBUG_MODE` | `false` | Expose error details (keep false when network-exposed) |

### Model Choices

**CLIP Model Mapping**:
- OpenAI CLIP models -> MLX
  - `ViT-B-32__openai` -> `mlx-community/clip-vit-base-patch32`
  - `ViT-B-16__openai`-> `mlx-community/clip-vit-base-patch16`
  - `ViT-L-14__openai`-> `mlx-community/clip-vit-large-patch14`
    
- LAION CLIP models -> MLX
  - `ViT-B-32__laion2b-s34b-b79k`-> `mlx-community/clip-vit-base-patch32-laion2b`
  - `ViT-B-32__laion2b_s34b_b79k`-> `mlx-community/clip-vit-base-patch32-laion2b`
    
- SigLIP models -> None (uses open_clip fallback)
  - `ViT-B-16-SigLIP__webli`
  - `ViT-B-16-SigLIP2__webli`
  - `ViT-SO400M-16-SigLIP2-384__webli`
    
- Default fallback: `mlx-community/clip-vit-base-patch32`

**Face Models**:
- `buffalo_s`
- `buffalo_m`
- `buffalo_l` - **default**

## Connecting to Immich

In your Immich `docker-compose.yml` or `.env`:

```yaml
# Point to your Mac's IP address (not localhost unless Immich is also native)
MACHINE_LEARNING_URL=http://192.168.1.100:3003
```

## Actually Running the Service

```bash
source .venv/bin/activate
uvicorn src.main:app --host 0.0.0.0 --port 3003 --workers 1
```

### Running as a Service (macOS launchd):

Create `~/Library/LaunchAgents/com.immich.ml.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.immich.ml</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/your/.venv/bin/python</string>
        <string>-m</string>
        <string>src.main</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/path/to/immich-ml-metal</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/immich-ml.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/immich-ml-error.log</string>
</dict>
</plist>
```

Then:
```bash
launchctl load ~/Library/LaunchAgents/com.immich.ml.plist
launchctl start com.immich.ml
```

## Verification

Test the service is working:

```bash
# Health check
curl http://localhost:3003/ping
# Should return: pong

# Service info
curl http://localhost:3003/
# Should return: {"message":"Immich ML"}

# Detailed health (checks all components)
curl http://localhost:3003/health
# Should return: {"status":"healthy","checks":{...}}

# Test CLIP text encoding
curl -X POST http://localhost:3003/predict \
  -F 'entries={"clip":{"textual":{"modelName":"ViT-B-32__openai"}}}' \
  -F 'text=a photo of a cat'
```

In Immich, you should see the ML service connect in the admin logs.

## Contributing

Given this is primarily an AI-assisted project, contributions are **very welcome**, especially from actual developers who can:

- Review and improve the code quality
- Add proper tests
- Verify Immich compatibility  
- Optimize performance
- Add missing features
- Improve documentation

## Support

This is a hobby project with no guarantees. I barely know how to use Git.

## Final Notes

This project exists because I wanted to run Immich ML on my Mac with passable hardware acceleration. **It works for me**, but may not work for everyone. Use at your own risk, and please contribute improvements if you can!

---

**tl;dr**: AI-written Immich ML service for Apple Silicon. Alpha quality. Use with caution. Please help.