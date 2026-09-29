"""Batched inference must hand every caller its own embedding back.

_BatchAccumulator stacks concurrent requests into one forward pass and returns
row i to caller i. If that mapping ever skewed, every photo would be stored
with another photo's embedding: no exception, no crash, just smart search that
quietly returns the wrong pictures. Nothing else would notice.

A fake model whose output row depends only on its own input row makes any
mix-up visible without needing real weights.
"""

import threading

import numpy as np
import torch

from src.models.clip import _BatchAccumulator

N = 12


class _RowWiseModel:
    """Embeds each image independently and records every batch it sees."""

    def __init__(self):
        self.batch_sizes = []

    def encode_image(self, stacked):
        self.batch_sizes.append(stacked.shape[0])
        return stacked.reshape(stacked.shape[0], -1)[:, :16].clone()


def _expected(tensor):
    v = tensor.reshape(-1)[:16].numpy().astype(np.float32)
    return v / np.linalg.norm(v)


def _submit_all(acc, tensors):
    results = [None] * len(tensors)
    start = threading.Barrier(len(tensors))

    def worker(i):
        start.wait()
        results[i] = acc.submit(tensors[i])

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(tensors))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    return results


def test_every_caller_gets_its_own_embedding_from_a_shared_batch():
    model = _RowWiseModel()
    acc = _BatchAccumulator(model, torch.device("cpu"), threading.Lock())
    try:
        # Different directions, not just different scales: the accumulator
        # normalises each embedding, so inputs that are multiples of one another
        # all come out identical and a swapped row would pass unnoticed.
        tensors = [
            torch.rand(1, 3, 4, 4, generator=torch.Generator().manual_seed(i))
            for i in range(N)
        ]
        # Guard the guard: every expected embedding must differ from the rest.
        expected = [_expected(t) for t in tensors]
        for a in range(N):
            for b in range(a + 1, N):
                assert not np.allclose(expected[a], expected[b], atol=1e-3)
        results = _submit_all(acc, tensors)
    finally:
        acc.stop()

    # Positive control: if nothing was batched, this test proves nothing.
    assert (
        max(model.batch_sizes) > 1
    ), f"requests never shared a pass: {model.batch_sizes}"
    for i, (got, t) in enumerate(zip(results, tensors)):
        assert got is not None, f"caller {i} never got a result"
        assert np.allclose(
            got, _expected(t), atol=1e-6
        ), f"caller {i} got another caller's embedding"


def test_a_failed_batch_reaches_every_caller_in_it():
    """One bad forward pass must fail each waiter, not leave them hanging."""

    class Broken(_RowWiseModel):
        def encode_image(self, stacked):
            raise RuntimeError("model fell over")

    acc = _BatchAccumulator(Broken(), torch.device("cpu"), threading.Lock())
    errors = []
    start = threading.Barrier(4)

    def worker():
        start.wait()
        try:
            acc.submit(torch.ones(1, 3, 4, 4))
        except RuntimeError as e:
            errors.append(str(e))

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    acc.stop()
    assert not any(t.is_alive() for t in threads), "a caller was left waiting forever"
    assert len(errors) == 4 and all("fell over" in e for e in errors)
