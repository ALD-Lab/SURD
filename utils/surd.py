"""Compatibility shim: ``import surd`` after ``sys.path.append('utils')``.

The code lives in the ``surd`` package under ``src/``. Scripts and notebooks
written for the original layout add ``utils/`` to ``sys.path`` and import
``surd``, ``it_tools``, ``analytic_eqs`` and ``transport_map`` as top-level
modules. This file keeps that working: it loads the package and hands it over
under the name ``surd`` so that every function keeps its old signature.

New code should install the package (``pip install .``) and ``import surd``.
"""

import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _package_candidates():
    # The source tree next to utils/ first, then any installed copy.
    yield os.path.abspath(os.path.join(_HERE, os.pardir, "src", "surd"))
    for entry in sys.path:
        if os.path.abspath(entry or os.curdir) == _HERE:
            continue  # the directory of this shim
        yield os.path.join(os.path.abspath(entry or os.curdir), "surd")


def _load_package():
    for pkg_dir in _package_candidates():
        init = os.path.join(pkg_dir, "__init__.py")
        if not os.path.isfile(init):
            continue
        spec = importlib.util.spec_from_file_location(
            "surd", init, submodule_search_locations=[pkg_dir]
        )
        module = importlib.util.module_from_spec(spec)
        # Replace this shim in sys.modules before executing the package so
        # that its relative imports resolve and ``import surd`` returns it.
        sys.modules["surd"] = module
        spec.loader.exec_module(module)
        return module
    raise ImportError(
        "The surd package was not found next to utils/ (../src/surd) nor on "
        "sys.path. Install it with 'pip install .' from the repository root."
    )


_load_package()
