"""Inputs of the exactness tests of ``surd.states``.

The same function builds the histograms for the reference fixture
(``tests/data/states_reference.npz``, produced with the original
SURD-states code) and for the test that compares ``surd.states`` to it,
so both sides see identical arrays.
"""

import numpy as np


def build_cases(analytic_eqs):
    """Dictionary name -> histogram ``p[target, source1, ...]``.

    ``analytic_eqs`` is the module with the benchmark ``source`` system
    (``surd.states.analytic_eqs`` or the original ``surd_states.analytic_eqs``).
    """
    cases = {}

    # Logic gates, exact distributions on fair independent bits.
    def gate(f):
        p = np.zeros((2, 2, 2))
        for a in (0, 1):
            for b in (0, 1):
                p[f(a, b), a, b] += 0.25
        return p

    cases["xor"] = gate(lambda a, b: a ^ b)
    cases["and"] = gate(lambda a, b: a & b)
    cases["copy1"] = gate(lambda a, b: a)
    dup = np.zeros((2, 2, 2))
    for b in (0, 1):
        dup[b, b, b] = 0.5
    cases["duplicated"] = dup

    # Random histograms with two and three sources, with empty bins.
    rng = np.random.default_rng(2025)
    h = rng.random((5, 4, 4))
    h[h < 0.3] = 0.0
    cases["random_2src"] = h * 1000.0
    h = rng.random((4, 3, 3, 3))
    h[h < 0.3] = 0.0
    cases["random_3src"] = h * 1000.0
    # Counts, as np.histogramdd returns them.
    cases["counts_2src"] = rng.poisson(3.0, size=(6, 5, 5)).astype(float)

    # Benchmark system of example E01 (source), short record, nlag 1.
    np.random.seed(10)
    qs = analytic_eqs.source(20_000)
    X = np.array([q[2_000:] for q in qs])
    nbins = 8
    max_abs = np.floor(np.percentile(X, 99.99))
    width = 2 * max_abs / (nbins - 1)
    edges = np.linspace(-max_abs, max_abs + width, nbins + 1)
    for i in range(3):
        Y = np.vstack([X[i, 1:], X[:, :-1]])
        hist, _ = np.histogramdd(Y.T, bins=[edges] * 4)
        cases[f"benchmark_source_target{i + 1}"] = hist

    return cases
