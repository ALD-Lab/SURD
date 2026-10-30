"""The utils/ shims and the package expose the same API and numbers.

Each check runs in a fresh interpreter so that nothing imported by the test
session can mask the import path under test.
"""

import json
import os
import subprocess
import sys

import pytest

from conftest import REPO_ROOT

PROBE = r"""
import json, sys
mode = sys.argv[1]
if mode == "append":
    sys.path.append("utils")
elif mode == "insert":
    sys.path.insert(0, "utils")
elif mode == "package":
    pass
import numpy as np
import surd
if mode == "package":
    from surd import it_tools, analytic_eqs
else:
    import it_tools, analytic_eqs

p = np.zeros((2, 2, 2))
for a in (0, 1):
    for b in (0, 1):
        p[a ^ b, a, b] = 0.25
I_R, I_S, MI, leak = surd.surd(p)

np.random.seed(0)
q = analytic_eqs.mediator(200)

out = {
    "file": surd.__file__,
    "version": surd.__version__,
    "public": sorted(n for n in dir(surd) if not n.startswith("_")),
    "I_R": {str(k): v for k, v in I_R.items()},
    "I_S": {str(k): v for k, v in I_S.items()},
    "MI": {str(k): v for k, v in MI.items()},
    "leak": leak,
    "entropy": it_tools.entropy_nvars(p, (0,)),
    "mediator": [float(x.sum()) for x in q],
    "it_tools_is_submodule": it_tools is surd.it_tools,
    "analytic_eqs_is_submodule": analytic_eqs is surd.analytic_eqs,
}
print(json.dumps(out))
"""

EXPECTED_PUBLIC = {
    "surd", "surd_hd", "run", "run_parallel", "run_multiple_lags",
    "plot", "plot_nlabels", "plot_multiple_lags", "nice_print",
    "it_tools", "analytic_eqs", "transport_map",
}


def _probe(mode):
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONPATH="")
    if mode == "package" and not _package_installed():
        env["PYTHONPATH"] = os.path.join(REPO_ROOT, "src")
    res = subprocess.run(
        [sys.executable, "-c", PROBE, mode],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=True,
    )
    return json.loads(res.stdout.strip().splitlines()[-1])


def _package_installed():
    res = subprocess.run(
        [sys.executable, "-c", "import surd"],
        cwd=REPO_ROOT, env=dict(os.environ, PYTHONPATH=""),
        capture_output=True, text=True,
    )
    return res.returncode == 0


@pytest.fixture(scope="module")
def package():
    return _probe("package")


@pytest.mark.parametrize("mode", ["append", "insert"])
def test_shim_gives_identical_numbers(mode, package):
    shim = _probe(mode)
    assert shim["I_R"] == package["I_R"]
    assert shim["I_S"] == package["I_S"]
    assert shim["MI"] == package["MI"]
    assert shim["leak"] == package["leak"]
    assert shim["entropy"] == package["entropy"]
    assert shim["mediator"] == package["mediator"]


@pytest.mark.parametrize("mode", ["append", "insert"])
def test_shim_loads_the_package_not_a_copy(mode):
    shim = _probe(mode)
    assert os.path.basename(shim["file"]) == "__init__.py"
    assert os.sep + "utils" + os.sep not in shim["file"]
    assert shim["it_tools_is_submodule"]
    assert shim["analytic_eqs_is_submodule"]
    with open(os.path.join(REPO_ROOT, "src", "surd", "__init__.py")) as fh:
        version = next(l for l in fh if l.startswith("__version__")).split('"')[1]
    assert shim["version"] == version


def test_public_api_is_complete(package):
    assert EXPECTED_PUBLIC <= set(package["public"])


def test_xor_numbers_from_the_package(package):
    scale = max(package["MI"].values())
    assert package["I_S"]["(1, 2)"] / scale == pytest.approx(1.0, abs=1e-6)
    assert package["leak"] == pytest.approx(0.0, abs=1e-6)
