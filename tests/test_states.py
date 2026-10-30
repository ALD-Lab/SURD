"""``surd.states`` reproduces the SURD-states code and stays out of ``import surd``.

The reference values in ``tests/data/states_reference.npz`` were produced
with the original SURD-states package (commit 291bb76 of
https://github.com/ALD-Lab/SURD-states) on the inputs that
``states_cases.build_cases`` rebuilds here. The merged code must return the
same numbers, in the same dictionary order.
"""

import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stdout

import matplotlib.pyplot as plt
import numpy as np
import pytest

import surd
from surd import states

from conftest import REPO_ROOT
from states_cases import build_cases

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                       "states_reference.npz")
CASE_NAMES = [
    "xor", "and", "copy1", "duplicated",
    "random_2src", "random_3src", "counts_2src",
    "benchmark_source_target1", "benchmark_source_target2",
    "benchmark_source_target3",
]
# The results are bitwise equal to the fixture in the environment that
# produced it. The tolerance absorbs last-digit differences of log2 between
# platforms and NumPy builds.
TOL = 1e-15
TERMS = ("I_R", "I_S", "MI", "Rd", "Un", "Sy")


def _key(k):
    return ",".join(map(str, k))


def _same(value, reference):
    value = np.asarray(value, dtype=float)
    reference = np.asarray(reference, dtype=float)
    assert value.shape == reference.shape
    assert np.array_equal(value, reference) or np.all(np.abs(value - reference) <= TOL)


@pytest.fixture(scope="module")
def reference():
    with np.load(FIXTURE) as data:
        return {name: data[name] for name in data.files}


@pytest.fixture(scope="module")
def cases():
    return build_cases(states.analytic_eqs)


# --- Same numbers as the original SURD-states code -------------------------

def test_fixture_covers_every_case(reference):
    names = {name.split("/")[0] for name in reference}
    assert names == set(CASE_NAMES)


@pytest.mark.parametrize("name", CASE_NAMES)
def test_matches_original_code(name, cases, reference):
    out = states.surd_states(cases[name].copy())
    assert len(out) == 7
    results = dict(zip(TERMS, (out[0], out[1], out[2], out[4], out[5], out[6])))
    _same(out[3], reference[f"{name}/info_leak"])
    for term, values in results.items():
        prefix = f"{name}/{term}/"
        expected_keys = [k[len(prefix):] for k in reference if k.startswith(prefix)]
        # Same keys, in the same order.
        assert [_key(k) for k in values] == expected_keys
        for k, v in values.items():
            _same(v, reference[prefix + _key(k)])


def test_input_is_not_modified(cases):
    p = cases["random_2src"].copy()
    states.surd_states(p)
    assert np.array_equal(p, cases["random_2src"])


def test_no_regularization_on_exact_distributions(cases):
    # SURD-states adds nothing to the histogram, so an exact XOR gives
    # exactly one bit of synergy. surd.surd adds 1e-14 to every bin.
    assert states.surd_states(cases["xor"])[1][(1, 2)] == 1.0
    assert surd.surd(cases["xor"].copy())[1][(1, 2)] != 1.0


def test_state_arrays_have_one_entry_per_state(cases):
    p = cases["random_3src"]
    I_R, I_S, MI, leak, Rd, Un, Sy = states.surd_states(p)
    nt = p.shape[0]
    assert set(Un) == {(1,), (2,), (3,)}
    assert set(Sy) == set(Rd) == {(1, 2), (1, 3), (2, 3)}
    for (i,), arr in Un.items():
        assert arr.shape == (nt, p.shape[i])
    for (i, j), arr in Sy.items():
        assert arr.shape == Rd[(i, j)].shape == (nt, p.shape[i], p.shape[j])


@pytest.mark.parametrize("name", ["random_2src", "random_3src",
                                  "benchmark_source_target2"])
def test_state_arrays_sum_to_the_aggregate_terms(name, cases):
    I_R, I_S, MI, leak, Rd, Un, Sy = states.surd_states(cases[name])
    for k, arr in Un.items():
        assert arr.sum() == pytest.approx(I_R[k], abs=1e-12)
    for k, arr in Sy.items():
        assert arr.sum() == pytest.approx(I_S[k], abs=1e-12)
    for k, arr in Rd.items():
        assert arr.sum() == pytest.approx(I_R[k], abs=1e-12)


# --- Packaging: own submodules, lazy and optional ---------------------------

def test_public_api():
    for name in ("surd_states", "run", "plot", "plot_states", "plot_states_3d",
                 "nice_print", "it_tools", "analytic_eqs", "datasets"):
        assert hasattr(states, name)
    assert states is surd.states
    assert "states" in dir(surd)


def test_states_keeps_its_own_tools():
    assert states.it_tools is not surd.it_tools
    assert states.analytic_eqs is not surd.analytic_eqs
    assert hasattr(states.it_tools, "safe_div")
    assert not hasattr(surd.it_tools, "safe_div")
    assert hasattr(states.analytic_eqs, "source")
    assert hasattr(states.analytic_eqs, "target")
    assert hasattr(surd.analytic_eqs, "mediator")


PROBE = r"""
import json, sys
import surd
loaded = lambda: {m for m in ("surd.states", "surd.states.analytic_eqs",
                              "skimage", "sklearn") if m in sys.modules}
after_surd = sorted(loaded())
from surd import states
after_states = sorted(loaded())
print(json.dumps({"after_surd": after_surd, "after_states": after_states}))
"""


def test_import_surd_does_not_load_states_or_its_optional_deps():
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONPATH=os.path.join(REPO_ROOT, "src"))
    res = subprocess.run([sys.executable, "-c", PROBE], env=env, cwd=REPO_ROOT,
                         capture_output=True, text=True, check=True)
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert out["after_surd"] == []
    assert out["after_states"] == ["surd.states"]


# --- Printing, plotting and run ---------------------------------------------

def test_nice_print_reports_normalized_terms(cases, capsys):
    states.nice_print(*states.surd_states(cases["xor"])[:4])
    out = capsys.readouterr().out
    assert "Synergystic (S):" in out
    assert "(1, 2)      : 1.0000" in out
    assert "Information Leak:  0.00%" in out


def test_plot_returns_normalized_bars(cases):
    I_R, I_S, MI, leak, *_ = states.surd_states(cases["random_2src"])
    fig, axs = plt.subplots(1, 2)
    bars = states.plot(I_R, I_S, leak, axs, nvars=2, threshold=-0.01)
    plt.close(fig)
    assert set(bars) == {"R12", "U1", "U2", "S12"}
    assert sum(bars.values()) == pytest.approx(1.0)
    assert len(axs[0].patches) == 4
    assert axs[1].patches[0].get_height() == pytest.approx(leak)


def test_run_matches_surd_states_on_the_last_target():
    rng = np.random.default_rng(8)
    X = rng.standard_normal((2, 10_000))
    X[1, 1:] += 0.9 * X[0, :-1]
    nbins = 8
    fig, axs = plt.subplots(2, 2)
    with redirect_stdout(io.StringIO()):
        out = states.run(X, 2, 1, nbins, axs)
    plt.close(fig)
    assert len(out) == 7
    Y = np.vstack([X[1, 1:], X[:, :-1]])
    hist, _ = np.histogramdd(Y.T, nbins)
    ref = states.surd_states(hist)
    assert out[0] == pytest.approx(ref[0])
    assert out[1] == pytest.approx(ref[1])
    assert out[3] == pytest.approx(ref[3])
    for k in ref[5]:
        assert np.allclose(out[5][k], ref[5][k])


def test_plot_states_saves_the_figure(cases, tmp_path):
    p = cases["benchmark_source_target2"]
    I_R, I_S, MI, leak, Rd, Un, Sy = states.surd_states(p)
    data = np.maximum(Un[(1,)] / I_R[(1,)], 0)
    edges = [np.linspace(-3, 3, n + 1) for n in p.shape]
    path = tmp_path / "figures" / "U1_states.pdf"
    fig, ax_main, ax_top, ax_right = states.plot_states(
        data=data, bins=edges, target=2, source=1, save_path=str(path),
        title="Unique", vmax=data.max(), vmin=-data.max(), xlabel="$q_1$",
        ylabel="$q_2^+$", cmap="RdBu_r", norm=False)
    plt.close(fig)
    assert path.is_file() and path.stat().st_size > 0
    assert len(ax_main.images) == 1


def test_plot_states_3d_needs_scikit_image(cases):
    pytest.importorskip("skimage")
    p = cases["benchmark_source_target2"]
    I_R, I_S, MI, leak, Rd, Un, Sy = states.surd_states(p)
    data = np.maximum(Sy[(1, 2)] / I_S[(1, 2)], 0)
    edges = [np.linspace(-3, 3, n + 1) for n in data.shape]
    states.plot_states_3d(data, edges, "S12", level=0.5 * data.max())
    plt.close("all")


# --- Dataset download helper ------------------------------------------------

@pytest.fixture
def fake_remote(tmp_path, monkeypatch):
    """Two small .mat files served from file:// URLs in place of GitHub."""
    import scipy.io as sio

    from surd.states import datasets

    remote = tmp_path / "remote"
    remote.mkdir()
    rng = np.random.default_rng(0)
    per_part = 10
    t = np.arange(per_part, dtype=np.float32)[:, None]
    files = {}
    for k in (1, 2):
        name = f"tbl_inner_outer_part{k}.mat"
        u = rng.standard_normal((2, per_part)).astype(np.float32) + 10 * k
        sio.savemat(remote / name, {"u": u, "t": t})
        files[name] = ((remote / name).as_uri(), datasets._sha256(remote / name))
    monkeypatch.setattr(datasets, "FILES", files)
    monkeypatch.setattr(datasets, "_SAMPLES_PER_PART", per_part)
    return remote, tmp_path / "cache", per_part


def test_load_inner_outer_downloads_only_what_it_needs(fake_remote, capsys):
    remote, cache, per_part = fake_remote
    X, t = states.datasets.load_inner_outer(5, cache_dir=cache)
    assert X.shape == (2, 5) and X.dtype == np.float32
    assert t.shape == (per_part, 1)
    assert (cache / "tbl_inner_outer_part1.mat").is_file()
    assert not (cache / "tbl_inner_outer_part2.mat").exists()
    assert "Downloading tbl_inner_outer_part1.mat" in capsys.readouterr().out

    X, t = states.datasets.load_inner_outer(None, cache_dir=cache)
    assert X.shape == (2, 2 * per_part)
    assert np.all(X[:, per_part:] > X[:, :per_part])  # part 2 after part 1
    assert (cache / "tbl_inner_outer_part2.mat").is_file()
    out = capsys.readouterr().out
    assert "part1" not in out and "part2" in out  # part 1 came from the cache

    # The cache is enough from now on.
    for f in remote.iterdir():
        f.unlink()
    X2, _ = states.datasets.load_inner_outer(None, cache_dir=cache)
    assert np.array_equal(X, X2)
    assert capsys.readouterr().out == ""


def test_load_inner_outer_rejects_a_wrong_checksum(fake_remote, monkeypatch):
    remote, cache, per_part = fake_remote
    from surd.states import datasets

    url, _ = datasets.FILES["tbl_inner_outer_part1.mat"]
    monkeypatch.setitem(datasets.FILES, "tbl_inner_outer_part1.mat", (url, "0" * 64))
    with pytest.raises(OSError, match="Checksum mismatch"):
        states.datasets.load_inner_outer(1, cache_dir=cache)
    assert not (cache / "tbl_inner_outer_part1.mat").exists()


def test_cache_dir_follows_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("SURD_DATA", str(tmp_path / "from_env"))
    assert states.datasets.cache_dir() == str(tmp_path / "from_env")
    assert (tmp_path / "from_env").is_dir()
    assert states.datasets.cache_dir(tmp_path / "explicit") == str(tmp_path / "explicit")
    monkeypatch.delenv("SURD_DATA")
    assert states.datasets.cache_dir().endswith(os.path.join(".cache", "surd"))


# --- Example E03 with the full record (slow, deselected by default) ----------

@pytest.mark.slow
def test_inner_outer_example_full_record():
    """Example E03 on the full record: outer-to-inner unique causality peaks
    at lag index 39 (Delta t+ = 58.28) and the decomposition at that lag
    matches the notebook. Downloads 148 MB on first use (cache in SURD_DATA
    or ~/.cache/surd) and takes several minutes. Run with
    ``pytest -m slow tests/test_states.py``.
    """
    it = states.it_tools
    X, time_plus = states.datasets.load_inner_outer(None)
    assert X.shape == (2, 19_990_000)
    nvars, nbins, max_lag = X.shape[0], 25, 100

    # Bins of the notebook.
    bins_list = []
    for i in range(nvars):
        if i == 0:
            lo, hi = np.percentile(X[i], 0.01), np.percentile(X[i], 99.99)
        else:
            lo, hi = np.percentile(X[i], 0.001), np.percentile(X[i], 98.5)
        width = (hi - lo) / (nbins - 1)
        bins_list.append(np.linspace(lo, hi + width, nbins + 1))

    # Unique causality from the outer layer to the inner layer (target 2)
    # as a function of the lag, normalized by the entropy of the target.
    i = 1
    lags = range(1, max_lag)
    unique_lag = np.zeros(len(lags))
    for n_idx, nlag in enumerate(lags):
        Y = np.vstack([X[i, nlag:], X[:, :-nlag]])
        hist, _ = np.histogramdd(Y.T, bins=[bins_list[i], bins_list[0], bins_list[1]])
        I_R, I_S, MI, leak, *_ = states.surd_states(hist)
        H = it.entropy_nvars(hist / hist.sum(), (0,))
        unique_lag[n_idx] = I_R[(1,)] / H
    nlag = int(unique_lag.argmax())
    assert nlag == 39
    assert time_plus[nlag][0] == pytest.approx(58.28, abs=0.005)

    expected = {
        1: {"R12": 0.0548, "U1": 0.7390, "U2": 0.0000, "S12": 0.2063},
        2: {"R12": 0.2514, "U1": 0.4576, "U2": 0.0403, "S12": 0.2507},
    }
    for i in range(nvars):
        Y = np.vstack([X[i, nlag:], X[:, :-nlag]])
        hist, _ = np.histogramdd(Y.T, bins=[bins_list[i], bins_list[0], bins_list[1]])
        I_R, I_S, MI, leak, *_ = states.surd_states(hist)
        scale = max(MI.values())
        got = {"R12": I_R[(1, 2)] / scale, "U1": I_R[(1,)] / scale,
               "U2": I_R[(2,)] / scale, "S12": I_S[(1, 2)] / scale}
        for key, value in expected[i + 1].items():
            assert round(got[key], 4) == value, (i + 1, key, got[key])
