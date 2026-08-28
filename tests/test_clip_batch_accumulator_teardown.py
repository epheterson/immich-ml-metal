"""Regression test for the _BatchAccumulator teardown race flagged in PR #7
review: unload() calls stop() while a request that captured this MLXClip
instance before a model switch may still be between preprocessing and
submit(). Before this fix, a submit() landing right after the background
thread drained its final batch and exited left that caller blocked on
event.wait() forever — nothing was left running to service the queue.

_BatchAccumulator now tracks liveness under the same lock the background
thread uses to decide whether to exit, so submit() either enqueues into a
thread that's guaranteed to still be looping, or (once the thread has
actually exited) runs the forward pass inline instead of queueing it.
"""
import threading

import numpy as np
import torch

from src.models.clip import _BatchAccumulator

DIM = 4


class _IdentityModel:
    """encode_image returns its input unchanged — enough to exercise the
    accumulator's batching/inline plumbing without a real CLIP model."""

    def encode_image(self, x):
        return x


def _make_accumulator():
    return _BatchAccumulator(_IdentityModel(), torch.device("cpu"), threading.Lock())


def _tensor(value: float = 1.0):
    return torch.full((1, DIM), value)


def test_submit_after_stop_falls_back_inline_instead_of_hanging():
    acc = _make_accumulator()
    acc.stop()  # background thread has fully drained and exited
    assert acc._alive is False

    result = acc.submit(_tensor(3.0))

    assert isinstance(result, np.ndarray)
    assert result.shape == (DIM,)
    assert np.allclose(result, 1.0 / DIM ** 0.5)  # uniform vector, L2-normalized


def test_submit_still_batches_while_alive():
    acc = _make_accumulator()
    result = acc.submit(_tensor(5.0))
    acc.stop()

    assert isinstance(result, np.ndarray)
    assert result.shape == (DIM,)


def test_concurrent_submit_races_stop_without_hanging():
    """Reproduces the maintainer-flagged race directly: many stragglers call
    submit() at the same moment unload() calls stop(). None may ever hang,
    regardless of whether they land before or after the thread's exit."""
    acc = _make_accumulator()
    n = 20
    results = [None] * n
    errors = [None] * n

    def worker(i):
        try:
            results[i] = acc.submit(_tensor(float(i + 1)))
        except Exception as e:  # pragma: no cover - only on real regression
            errors[i] = e

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    acc.stop()  # races the workers' submit() calls, same as unload() would
    for t in threads:
        t.join(timeout=2)

    assert all(not t.is_alive() for t in threads), "a submit() call hung after stop()"
    assert all(e is None for e in errors), errors
    assert all(r is not None and r.shape == (DIM,) for r in results)
