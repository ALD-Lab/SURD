"""Benchmark systems of example E02 with unambiguous dominant terms.

E02 runs 5e7 steps and 50 bins. Here a short record and a coarse histogram
are enough to recover the dominant contribution of each case, which is what
the examples establish:

- mediator Q3 -> Q2 -> Q1: unique Q2 -> Q1 and unique Q3 -> Q2;
- confounder Q3 -> Q1, Q3 -> Q2: synergy of Q2 with Q3 on Q2;
- synergistic collider [Q2, Q3] -> Q1: synergy S23 on Q1;
- redundant collider Q2 = Q3 -> Q1: redundancy R23 on Q2 and Q3.
"""

import numpy as np
import pytest

import surd
from surd import analytic_eqs as cases

from helpers import dominant, lagged_histogram, normalized

NSTEPS = 100_000
TRANSIENT = 1000
NBINS = 12
NLAG = 1


def _simulate(system):
    np.random.seed(10)
    qs = system(NSTEPS)
    return np.array([q[TRANSIENT:] for q in qs])


def _decompose(X, target):
    hist = lagged_histogram(X, target, NLAG, NBINS)
    I_R, I_S, MI, leak = surd.surd(hist)
    R, S = normalized(I_R, I_S, MI)
    return R, S, leak


@pytest.fixture(scope="module")
def mediator():
    return _simulate(cases.mediator)


@pytest.fixture(scope="module")
def confounder():
    return _simulate(cases.confounder)


@pytest.fixture(scope="module")
def synergistic():
    return _simulate(cases.synergistic_collider)


@pytest.fixture(scope="module")
def redundant():
    return _simulate(cases.redundant_collider)


def test_mediator_q2_drives_q1(mediator):
    R, S, _ = _decompose(mediator, target=0)
    assert dominant(R) == (2,)
    assert R[(2,)] > 0.9
    assert max(S.values()) < 0.05


def test_mediator_q3_drives_q2(mediator):
    R, S, _ = _decompose(mediator, target=1)
    assert R[(3,)] > 0.9
    assert R[(1,)] < 0.05
    assert R[(2,)] < 0.05


def test_mediator_q3_is_self_driven(mediator):
    R, S, _ = _decompose(mediator, target=2)
    assert dominant(R) == (3,)
    assert R[(3,)] > 0.8
    assert R[(3,)] > max(S.values())


def test_confounder_q2_needs_q2_and_q3_together(confounder):
    R, S, _ = _decompose(confounder, target=1)
    assert dominant(S) == (2, 3)
    assert S[(2, 3)] > 0.5
    assert R[(1,)] < 0.05
    assert S[(1, 2)] < 0.05


def test_confounder_q3_is_self_driven(confounder):
    R, S, _ = _decompose(confounder, target=2)
    assert dominant(R) == (3,)
    assert R[(3,)] > 0.6
    assert R[(3,)] > max(S.values())


def test_synergistic_collider_q1_is_synergy_of_q2_q3(synergistic):
    R, S, _ = _decompose(synergistic, target=0)
    assert dominant(S) == (2, 3)
    assert S[(2, 3)] > 0.5
    assert R[(2,)] < 0.05
    assert R[(3,)] < 0.05


def test_redundant_collider_q2_q3_are_redundant(redundant):
    for target in (1, 2):
        R, S, _ = _decompose(redundant, target=target)
        assert dominant(R) == (2, 3)
        assert R[(2, 3)] > 0.9
        assert R[(2,)] < 0.01
        assert R[(3,)] < 0.01


def test_redundant_collider_sources_are_identical(redundant):
    # Q3 is defined as Q2, so the decompositions of both targets coincide.
    R2, S2, leak2 = _decompose(redundant, target=1)
    R3, S3, leak3 = _decompose(redundant, target=2)
    assert R2 == pytest.approx(R3)
    assert S2 == pytest.approx(S3)
    assert leak2 == pytest.approx(leak3)
