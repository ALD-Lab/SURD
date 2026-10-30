"""surd.neural on small systems with a known answer. Short trainings, a few
seconds each on one CPU thread."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import surd  # noqa: E402
import surd.neural as sn  # noqa: E402

FAST = dict(max_steps=1500, kfold=2)


@pytest.fixture(autouse=True)
def _one_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def _gates(N, seed, f, nvars=3):
    rng = np.random.default_rng(seed)
    S = rng.integers(0, 2, (nvars, N)).astype(float)
    return np.vstack([f(S), S])


def test_copy_is_unique_to_its_source():
    Y = _gates(3000, 0, lambda S: S[1])
    I_R, I_S, MI, leak = sn.surd(Y, 2, **FAST)
    total = max(MI.values())
    assert total == pytest.approx(1.0, abs=0.05)
    assert I_R[(2,)] / total > 0.9
    assert leak < 0.05


def test_xor_is_synergy_of_its_sources():
    Y = _gates(3000, 1, lambda S: np.logical_xor(S[0], S[1]).astype(float))
    I_R, I_S, MI, leak = sn.surd(Y, 2, **FAST)
    total = max(MI.values())
    assert I_S[(1, 2)] / total > 0.9


def test_continuous_mediator_unique():
    # T is a noisy function of source 1; sources 2 and 3 are noise.
    rng = np.random.default_rng(2)
    S = rng.standard_normal((3, 4000))
    T = np.tanh(2 * S[0]) + 0.2 * rng.standard_normal(4000)
    I_R, I_S, MI, leak = sn.surd(np.vstack([T, S]), 8, **FAST)
    total = max(MI.values())
    assert I_R[(1,)] / total > 0.85


def test_null_is_small():
    Y = _gates(3000, 3, lambda S: S[0])
    assert sn.null_mi(Y, 2, **FAST) < 0.05


def test_keys_match_histogram_path():
    Y = _gates(2000, 4, lambda S: S[0])
    I_R, I_S, MI, _ = sn.surd(Y, 2, **FAST)
    hist, _ = np.histogramdd(Y.T, 2)
    R2, S2, M2, _ = surd.surd(hist)
    assert I_R.keys() == R2.keys() and I_S.keys() == S2.keys() and MI.keys() == M2.keys()


def test_monotone_in_subsets():
    Y = _gates(2000, 5, lambda S: S[0] * S[2])
    Is, p_t, leak = sn.specific_info(Y, 2, **FAST)
    for A in Is:
        for B in Is:
            if set(B) < set(A):
                assert np.all(Is[A] >= Is[B])
        assert np.all(Is[A] >= 0)
    assert p_t.sum() == pytest.approx(1.0)


def test_unknown_option_raises():
    with pytest.raises(TypeError):
        sn.specific_info(_gates(100, 0, lambda S: S[0]), 2, hiden=8)
