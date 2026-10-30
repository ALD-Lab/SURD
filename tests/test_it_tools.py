"""Information-theoretic tools on exact discrete distributions."""

import numpy as np
import pytest

from surd import it_tools as it

TOL = 1e-12


def test_mylog_is_log2_and_zero_on_invalid_entries():
    x = np.array([1.0, 2.0, 0.5, 0.0, np.nan, np.inf])
    out = it.mylog(x)
    assert out[:3] == pytest.approx([0.0, 1.0, -1.0])
    assert out[3:] == pytest.approx([0.0, 0.0, 0.0])


def test_entropy_of_uniform_distribution():
    assert it.entropy(np.full(8, 1 / 8)) == pytest.approx(3.0, abs=TOL)
    assert it.entropy(np.array([1.0, 0.0])) == pytest.approx(0.0, abs=TOL)


def test_joint_and_conditional_entropy_of_a_copy():
    # X1 is a copy of X0, X2 is an independent fair bit.
    p = np.zeros((2, 2, 2))
    for b in (0, 1):
        for c in (0, 1):
            p[b, b, c] = 0.25
    assert it.entropy_nvars(p, (0,)) == pytest.approx(1.0, abs=TOL)
    assert it.entropy_nvars(p, (0, 1)) == pytest.approx(1.0, abs=TOL)
    assert it.entropy_nvars(p, (0, 2)) == pytest.approx(2.0, abs=TOL)
    assert it.cond_entropy(p, (0,), (1,)) == pytest.approx(0.0, abs=TOL)
    assert it.cond_entropy(p, (0,), (2,)) == pytest.approx(1.0, abs=TOL)
    assert it.mutual_info(p, (0,), (1,)) == pytest.approx(1.0, abs=TOL)
    assert it.mutual_info(p, (0,), (2,)) == pytest.approx(0.0, abs=TOL)


def test_conditional_mutual_information_of_xor():
    # X0 = X1 xor X2: the pair is informative only once one of them is known.
    p = np.zeros((2, 2, 2))
    for a in (0, 1):
        for b in (0, 1):
            p[a ^ b, a, b] = 0.25
    assert it.mutual_info(p, (0,), (1,)) == pytest.approx(0.0, abs=TOL)
    assert it.cond_mutual_info(p, (0,), (1,), (2,)) == pytest.approx(1.0, abs=TOL)


def test_myhistogram_is_normalized():
    rng = np.random.default_rng(0)
    h = it.myhistogram(rng.standard_normal((500, 2)), 5)
    assert h.shape == (5, 5)
    assert h.sum() == pytest.approx(1.0)
    assert (h > 0).all()


def test_transfer_entropy_to_a_copied_source():
    # Future of variable 1 (axis 0) is a copy of present variable 2 (axis 2);
    # its own past (axis 1) is an independent bit.
    p = np.zeros((2, 2, 2))
    for b in (0, 1):
        for c in (0, 1):
            p[c, b, c] = 0.25
    TE = it.transfer_entropy(p, target_var=1)
    assert TE.shape == (2,)
    assert TE[1] == pytest.approx(1.0, abs=TOL)
