"""Metal GPU lock for serializing MLX inference.

MLX uses Metal command buffers with lazy evaluation. If an MLX array
hasn't been evaluated when a concurrent Vision framework call submits
its own Metal work, the command buffers collide and the process
crashes with an assertion.

This lock serializes MLX inference (CLIP) so Metal evaluation
completes before the lock releases. Vision framework (face detection,
OCR) and CoreML (face embeddings via ONNX+CoreML EP) are internally
thread-safe and do NOT need this lock — they use separate Metal
command queues per handler.

Import: from src.gpu_lock import metal_lock
"""

import threading

metal_lock = threading.Lock()
