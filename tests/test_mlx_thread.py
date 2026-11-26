"""MLX work runs on one thread, whichever thread asks (#38, #103)."""
import threading

import numpy as np
import pytest

from src.mlx_thread import run_on_mlx_thread

mx = pytest.importorskip("mlx.core")


def test_runs_on_one_thread_from_any_caller():
    seen = set()

    def ident():
        seen.add(threading.get_ident())

    threads = [threading.Thread(target=run_on_mlx_thread, args=(ident,)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    run_on_mlx_thread(ident)
    assert len(seen) == 1
    assert threading.get_ident() not in seen


def test_nested_call_runs_inline():
    assert run_on_mlx_thread(lambda: run_on_mlx_thread(lambda: 7)) == 7


def test_array_made_on_one_caller_evaluates_from_another():
    made = run_on_mlx_thread(lambda: mx.array([1.0, 2.0]) * 2)
    out = {}

    def other():
        out["v"] = run_on_mlx_thread(lambda: np.array(made + 1))

    t = threading.Thread(target=other)
    t.start()
    t.join()
    assert out["v"].tolist() == [3.0, 5.0]


def test_errors_reach_the_caller():
    def boom():
        raise ValueError("x")

    with pytest.raises(ValueError):
        run_on_mlx_thread(boom)
