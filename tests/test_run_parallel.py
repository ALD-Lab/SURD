"""surd.run_parallel gives the same numbers with and without pymp."""

import io
from contextlib import redirect_stdout

import matplotlib.pyplot as plt
import numpy as np
import pytest

import surd
import surd.core

from helpers import lagged_histogram

NBINS = 8
NLAG = 1


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(11)
    X = rng.standard_normal((3, 20_000))
    X[0, 1:] += 0.8 * X[1, :-1] * X[2, :-1]
    X[1, 1:] += 0.6 * X[2, :-1]
    return X


def _run_parallel(X):
    fig, axs = plt.subplots(3, 2)
    with redirect_stdout(io.StringIO()):
        out = surd.run_parallel(X, 3, NLAG, NBINS, axs)
    plt.close(fig)
    return out


def test_serial_path_matches_surd(data, monkeypatch):
    monkeypatch.setattr(surd.core, "pymp", None)
    I_R, I_S, MI, leak = _run_parallel(data)
    ref = surd.surd(lagged_histogram(data, 0, NLAG, NBINS))
    assert I_R == pytest.approx(ref[0])
    assert I_S == pytest.approx(ref[1])
    assert MI == pytest.approx(ref[2])
    assert leak == pytest.approx(ref[3])


def test_parallel_and_serial_paths_agree(data, monkeypatch):
    if surd.core.pymp is None:
        pytest.skip("pymp is not installed")
    parallel = _run_parallel(data)
    monkeypatch.setattr(surd.core, "pymp", None)
    serial = _run_parallel(data)
    for a, b in zip(parallel, serial):
        assert a == pytest.approx(b, abs=0.0)


def test_run_parallel_draws_one_row_per_target(data, monkeypatch):
    monkeypatch.setattr(surd.core, "pymp", None)
    fig, axs = plt.subplots(3, 2)
    with redirect_stdout(io.StringIO()):
        surd.run_parallel(data, 3, NLAG, NBINS, axs)
    for i in range(3):
        assert len(axs[i, 0].patches) == 11  # 7 R/U bars and 4 S bars
        assert len(axs[i, 1].patches) == 1
    plt.close(fig)
