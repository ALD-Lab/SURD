"""Logic gates with known decompositions (example E01).

The distributions are built exactly, so the expected values hold to the
regularization of 1e-14 that ``surd.surd`` adds to every bin.
"""

import numpy as np
import pytest

import surd

from helpers import duplicated_source_histogram, gate_histogram, normalized

TOL = 1e-6


def test_duplicated_source_is_redundant():
    I_R, I_S, MI, leak = surd.surd(duplicated_source_histogram())
    R, S = normalized(I_R, I_S, MI)
    assert R[(1, 2)] == pytest.approx(1.0, abs=TOL)
    assert R[(1,)] == pytest.approx(0.0, abs=TOL)
    assert R[(2,)] == pytest.approx(0.0, abs=TOL)
    assert S[(1, 2)] == pytest.approx(0.0, abs=TOL)
    assert leak == pytest.approx(0.0, abs=TOL)
    # One bit of information in total.
    assert MI[(1, 2)] == pytest.approx(1.0, abs=TOL)


def test_copied_source_is_unique():
    I_R, I_S, MI, leak = surd.surd(gate_histogram(lambda a1, a2: a1))
    R, S = normalized(I_R, I_S, MI)
    assert R[(1,)] == pytest.approx(1.0, abs=TOL)
    assert R[(2,)] == pytest.approx(0.0, abs=TOL)
    assert R[(1, 2)] == pytest.approx(0.0, abs=TOL)
    assert S[(1, 2)] == pytest.approx(0.0, abs=TOL)
    assert leak == pytest.approx(0.0, abs=TOL)


def test_copied_second_source_is_unique_to_it():
    I_R, I_S, MI, _ = surd.surd(gate_histogram(lambda a1, a2: a2))
    R, S = normalized(I_R, I_S, MI)
    assert R[(2,)] == pytest.approx(1.0, abs=TOL)
    assert R[(1,)] == pytest.approx(0.0, abs=TOL)
    assert R[(1, 2)] == pytest.approx(0.0, abs=TOL)
    assert S[(1, 2)] == pytest.approx(0.0, abs=TOL)


def test_xor_is_synergistic():
    I_R, I_S, MI, leak = surd.surd(gate_histogram(lambda a1, a2: a1 ^ a2))
    R, S = normalized(I_R, I_S, MI)
    assert S[(1, 2)] == pytest.approx(1.0, abs=TOL)
    assert R[(1,)] == pytest.approx(0.0, abs=TOL)
    assert R[(2,)] == pytest.approx(0.0, abs=TOL)
    assert R[(1, 2)] == pytest.approx(0.0, abs=TOL)
    assert leak == pytest.approx(0.0, abs=TOL)
    # Each source alone says nothing about an XOR output.
    assert MI[(1,)] == pytest.approx(0.0, abs=TOL)
    assert MI[(2,)] == pytest.approx(0.0, abs=TOL)
    assert MI[(1, 2)] == pytest.approx(1.0, abs=TOL)


def test_xor_with_an_empty_middle_bin():
    # E01 histograms the XOR case with three bins. The middle bin is empty
    # and the result is the same as with two bins.
    p = np.zeros((3, 3, 3))
    for a1 in (0, 1):
        for a2 in (0, 1):
            p[2 * (a1 ^ a2), 2 * a1, 2 * a2] = 0.25
    I_R, I_S, MI, leak = surd.surd(p)
    R, S = normalized(I_R, I_S, MI)
    assert S[(1, 2)] == pytest.approx(1.0, abs=TOL)
    assert sum(R.values()) == pytest.approx(0.0, abs=TOL)
    assert leak == pytest.approx(0.0, abs=TOL)


def test_and_gate_has_no_unique_information():
    # AND is symmetric in its inputs, so neither source can hold unique
    # information. The information splits between redundant and synergistic.
    I_R, I_S, MI, leak = surd.surd(gate_histogram(lambda a1, a2: a1 & a2))
    R, S = normalized(I_R, I_S, MI)
    assert R[(1,)] == pytest.approx(0.0, abs=TOL)
    assert R[(2,)] == pytest.approx(0.0, abs=TOL)
    assert R[(1, 2)] > 0.0
    assert S[(1, 2)] > 0.0
    assert R[(1, 2)] + S[(1, 2)] == pytest.approx(1.0, abs=TOL)
    assert leak == pytest.approx(0.0, abs=TOL)


def test_independent_sources_leave_everything_to_the_leak():
    # Target is a fresh fair bit, independent of both sources.
    p = np.full((2, 2, 2), 0.125)
    I_R, I_S, MI, leak = surd.surd(p)
    assert all(abs(v) < TOL for v in I_R.values())
    assert all(abs(v) < TOL for v in I_S.values())
    assert all(abs(v) < TOL for v in MI.values())
    assert leak == pytest.approx(1.0, abs=TOL)


def test_three_source_parity_is_triple_synergy():
    # Parity of three fair bits: only the three sources together inform.
    p = np.zeros((2, 2, 2, 2))
    for a in np.ndindex(2, 2, 2):
        p[(a[0] ^ a[1] ^ a[2],) + a] = 0.125
    I_R, I_S, MI, leak = surd.surd(p)
    R, S = normalized(I_R, I_S, MI)
    assert S[(1, 2, 3)] == pytest.approx(1.0, abs=TOL)
    assert sum(R.values()) == pytest.approx(0.0, abs=TOL)
    assert S[(1, 2)] == pytest.approx(0.0, abs=TOL)
    assert leak == pytest.approx(0.0, abs=TOL)


def test_sum_of_terms_equals_total_mutual_information():
    for gate in (lambda a, b: a, lambda a, b: a ^ b, lambda a, b: a & b,
                 lambda a, b: a | b):
        I_R, I_S, MI, _ = surd.surd(gate_histogram(gate))
        total = sum(I_R.values()) + sum(I_S.values())
        assert total == pytest.approx(MI[(1, 2)], abs=TOL)
