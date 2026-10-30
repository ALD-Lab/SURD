"""Ties between a subset and its supersets go to the subset."""

import numpy as np
import pytest

import surd


def _pmf(f, nvars):
    """Exact table of a binary target f(sources) with nvars uniform binary sources."""
    p = np.zeros((2,) * (nvars + 1))
    for s in np.ndindex(*(2,) * nvars):
        p[(f(s),) + s] += 1
    return p / p.sum()


@pytest.mark.parametrize("nvars", [3, 4, 5, 6])
def test_copy_is_unique(nvars):
    # The target is source 2. Every superset of source 2 carries the same
    # information, which must stay with source 2.
    I_R, I_S, MI, leak = surd.surd(_pmf(lambda s: s[1], nvars))
    assert I_R[(2,)] == pytest.approx(1.0, abs=1e-9)
    assert max(I_S.values()) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("nvars", [3, 6])
def test_xor_is_synergy(nvars):
    I_R, I_S, MI, leak = surd.surd(_pmf(lambda s: s[0] ^ s[1], nvars))
    assert I_S[(1, 2)] == pytest.approx(1.0, abs=1e-9)
    assert sum(I_R.values()) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("nvars", [2, 6])
def test_and_gate(nvars):
    # Same values with or without independent extra sources.
    I_R, I_S, MI, leak = surd.surd(_pmf(lambda s: s[0] & s[1], nvars))
    assert I_R[(1, 2)] == pytest.approx(0.3113, abs=1e-4)
    assert I_S[(1, 2)] == pytest.approx(0.5, abs=1e-9)


def test_assign_matches_surd():
    # surd.assign on the histogram specific information reproduces surd.surd.
    rng = np.random.default_rng(3)
    h = rng.random((4, 3, 3, 3)) ** 4
    expected = surd.surd(h.copy())
    p = h + 1e-14
    p /= p.sum()
    from itertools import combinations
    from surd import it_tools as it
    inds = range(1, 4)
    p_s = p.sum(axis=tuple(inds), keepdims=True)
    Is = {}
    for n in inds:
        for j in combinations(inds, n):
            noj = tuple(set(inds) - set(j))
            p_a = p.sum(axis=(0, *noj), keepdims=True)
            p_as = p.sum(axis=noj, keepdims=True)
            Is[j] = ((p_as / p_s) * (it.mylog(p_as / p_a) - it.mylog(p_s))).sum(axis=j).ravel()
    I_R, I_S, MI = surd.assign(Is, p_s.ravel())
    for a, b in zip((I_R, I_S, MI), expected[:3]):
        assert a.keys() == b.keys()
        for k in a:
            assert a[k] == pytest.approx(b[k], abs=1e-12)
