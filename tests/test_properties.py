"""Bounds and consistency properties on random and synthetic data."""

import io
from contextlib import redirect_stdout

import matplotlib.pyplot as plt
import numpy as np
import pytest

import surd

from helpers import lagged_histogram


@pytest.mark.parametrize("shape", [(3, 3), (4, 4, 4), (3, 5, 2, 4), (2,) * 5])
def test_bounds_on_random_distributions(shape):
    rng = np.random.default_rng(sum(shape))
    for _ in range(5):
        p = rng.random(shape) ** 3
        I_R, I_S, MI, leak = surd.surd(p)
        assert 0.0 <= leak <= 1.0
        assert all(v >= 0.0 for v in I_R.values())
        assert all(v >= 0.0 for v in I_S.values())
        assert all(v >= -1e-12 for v in MI.values())
        nsrc = len(shape) - 1
        assert len(I_R) == 2 ** nsrc - 1
        assert len(I_S) == 2 ** nsrc - 1 - nsrc
        assert len(MI) == 2 ** nsrc - 1


def test_leak_is_conditional_entropy_over_entropy():
    rng = np.random.default_rng(1)
    p = rng.random((4, 3, 5))
    I_R, I_S, MI, leak = surd.surd(p)
    H = surd.it_tools.entropy_nvars(p, (0,))
    Hc = surd.it_tools.cond_entropy(p, (0,), (1, 2))
    assert leak == pytest.approx(Hc / H)
    # The mutual information of the full source set closes the balance.
    assert MI[(1, 2)] == pytest.approx(H - Hc)


def test_surd_normalizes_its_input_in_place():
    p = np.ones((2, 2, 2))
    surd.surd(p)
    assert p.sum() == pytest.approx(1.0)


def test_surd_hd_matches_surd_with_all_combinations():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((3, 20_000))
    X[0, 1:] += 0.7 * X[1, :-1] - 0.5 * X[2, :-1]
    Y = np.vstack([X[0, 1:], X[:, :-1]])
    nbins = 6
    hist, _ = np.histogramdd(Y.T, nbins)
    I_R, I_S, MI, _ = surd.surd(hist)
    I_R_hd, I_S_hd, MI_hd = surd.surd_hd(Y, nbins, max_combs=3)
    assert I_R_hd == pytest.approx(I_R, abs=1e-9)
    assert I_S_hd == pytest.approx(I_S, abs=1e-9)
    assert MI_hd == pytest.approx(MI, abs=1e-9)


def test_surd_hd_truncates_synergy_order():
    rng = np.random.default_rng(3)
    Y = rng.standard_normal((5, 5_000))
    I_R, I_S, MI = surd.surd_hd(Y, 4, max_combs=2)
    assert max(len(k) for k in I_S) == 2
    assert max(len(k) for k in MI) == 2
    # Redundancies are still reported for every subset of sources.
    assert max(len(k) for k in I_R) == 4


def test_nice_print_reports_normalized_terms(capsys):
    p = np.zeros((2, 2, 2))
    for a in (0, 1):
        for b in (0, 1):
            p[a ^ b, a, b] = 0.25
    surd.nice_print(*surd.surd(p))
    out = capsys.readouterr().out
    assert "Redundant (R):" in out
    assert "Unique (U):" in out
    assert "(1, 2)      : 1.0000" in out
    assert "Information Leak:  0.00%" in out


def test_plot_returns_normalized_bars():
    rng = np.random.default_rng(4)
    p = rng.random((3, 4, 4))
    I_R, I_S, MI, leak = surd.surd(p)
    fig, axs = plt.subplots(1, 2)
    bars = surd.plot(I_R, I_S, leak, axs, nvars=2, threshold=-0.01)
    plt.close(fig)
    assert set(bars) == {"R12", "U1", "U2", "S12"}
    assert sum(bars.values()) == pytest.approx(1.0)
    assert len(axs[0].patches) == 4
    assert axs[1].patches[0].get_height() == pytest.approx(leak)


def test_plot_nlabels_keeps_the_largest_bars():
    rng = np.random.default_rng(5)
    p = rng.random((3, 3, 3, 3))
    I_R, I_S, MI, leak = surd.surd(p)
    fig, axs = plt.subplots(1, 2)
    surd.plot_nlabels(I_R, I_S, leak, axs, nvars=3, nlabels=4)
    plt.close(fig)
    assert len(axs[0].patches) == 4


def test_run_matches_surd_on_each_target():
    rng = np.random.default_rng(6)
    X = rng.standard_normal((2, 10_000))
    X[1, 1:] += 0.9 * X[0, :-1]
    nbins = 8
    fig, axs = plt.subplots(2, 2)
    with redirect_stdout(io.StringIO()):
        I_R, I_S, MI, leak = surd.run(X, 2, 1, nbins, axs)
    plt.close(fig)
    # run returns the decomposition of the last target.
    ref = surd.surd(lagged_histogram(X, 1, 1, nbins))
    assert I_R == pytest.approx(ref[0])
    assert I_S == pytest.approx(ref[1])
    assert leak == pytest.approx(ref[3])


def test_run_multiple_lags_returns_bounded_leak():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((2, 6_000))
    X[0, 2:] += 0.8 * X[1, :-2]
    fig, axs = plt.subplots(2, 2)
    with redirect_stdout(io.StringIO()):
        I_R, I_S, MI, leak = surd.run_multiple_lags(X, 2, 2, 5, 2, axs)
    plt.close(fig)
    assert 0.0 <= leak <= 1.0
    assert max(len(k) for k in I_R) == 4
    assert max(len(k) for k in I_S) == 2
