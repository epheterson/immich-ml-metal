"""One thread for every MLX call.

From mlx 0.31.2 on, MLX streams belong to the thread that created them. A
model loaded on one thread and run on another fails with "There is no
Stream(gpu, 0)" (0.31.2) or "There is no Stream(cpu, 1)" (0.32.x), even with
one request at a time (#38, #103). Running loading, inference and cache
clearing all on this one thread keeps every array on the stream it was made
on. CLIP inference was already serialized by metal_lock, so this costs no
throughput.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
_mlx_thread_ident = None


def _remember_thread():
    global _mlx_thread_ident
    _mlx_thread_ident = threading.get_ident()


_executor.submit(_remember_thread).result()


def run_on_mlx_thread(fn, *args, **kwargs):
    """Run fn on the MLX thread and return its result (or raise its error).
    Called from the MLX thread itself, it runs inline instead of deadlocking."""
    if threading.get_ident() == _mlx_thread_ident:
        return fn(*args, **kwargs)
    return _executor.submit(fn, *args, **kwargs).result()
