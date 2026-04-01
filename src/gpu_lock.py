"""Global Metal GPU lock for serializing all Metal-touching operations.

MLX, Vision framework, and CoreML all use Metal internally. Concurrent
access crashes with command buffer assertions. This module provides a
single lock that all GPU-touching code should acquire.

Import: from src.gpu_lock import metal_lock
"""
import threading

metal_lock = threading.Lock()
