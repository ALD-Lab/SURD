"""Small utilities shared by the tests."""

import numpy as np


def gate_histogram(gate):
    """Exact joint distribution p[target, source1, source2] of a logic gate.

    Both sources are independent fair bits. ``gate(a1, a2)`` gives the target,
    which is the future of the gate output, as in example E01. The histogram
    has shape (2, 2, 2) and sums to one.
    """
    p = np.zeros((2, 2, 2))
    for a1 in (0, 1):
        for a2 in (0, 1):
            p[gate(a1, a2), a1, a2] += 0.25
    return p


def duplicated_source_histogram():
    """Target is a copy of a bit that enters twice as a source (E01, case 1).

    The joint distribution lives on (t, a1, a2) = (b, b, b).
    """
    p = np.zeros((2, 2, 2))
    for b in (0, 1):
        p[b, b, b] = 0.5
    return p


def normalized(I_R, I_S, MI):
    """Terms divided by the mutual information of the full source set.

    This is the normalization used by ``surd.nice_print`` and ``surd.plot``.
    """
    scale = max(MI.values())
    return ({k: v / scale for k, v in I_R.items()},
            {k: v / scale for k, v in I_S.items()})


def lagged_histogram(X, target, nlag, nbins):
    """Histogram with the future of X[target] first and all present X after.

    Same arrangement as ``surd.run``: ``Y = [X[i, nlag:], X[:, :-nlag]]``.
    """
    Y = np.vstack([X[target, nlag:], X[:, :-nlag]])
    hist, _ = np.histogramdd(Y.T, nbins)
    return hist


def dominant(terms):
    """Key with the largest value in a dictionary of contributions."""
    return max(terms, key=terms.get)
