"""Causality by states: SURD resolved over the states of the variables.

``surd.states`` splits the mutual information between the future of a
target and the present of the sources into redundant, unique and
synergistic contributions, as ``surd.surd`` does, and also returns how
much each state of the sources and of the target contributes to each
term. The result tells not only which variables are causal but in which
states they are.

Reference: Martínez-Sánchez & Lozano-Durán, "Observational causality by
states and interaction type for scientific discovery", Communications
Physics 9, 15 (2025). https://doi.org/10.1038/s42005-025-02447-w

Typical use::

    import numpy as np
    from surd import states

    hist, bins = np.histogramdd(Y.T, nbins)   # Y[0] target, Y[1:] sources
    I_R, I_S, MI, leak, Rd, Un, Sy = states.surd_states(hist)
    states.nice_print(I_R, I_S, MI, leak)

The code is the one published with the paper. Its information-theoretic
tools (``surd.states.it_tools``) and benchmark systems
(``surd.states.analytic_eqs``) are kept apart from ``surd.it_tools`` and
``surd.analytic_eqs``: the names overlap but the numerics differ, since
this version adds no regularization to the histogram.

Submodules:

- ``surd.states.it_tools``: entropies, mutual information, transfer
  entropy and a k-nearest-neighbours density estimate (the last one needs
  ``scikit-learn``).
- ``surd.states.analytic_eqs``: the two benchmark systems of the paper,
  ``source`` and ``target``. Imported on first access, since it seeds the
  NumPy random generator.
- ``surd.states.datasets``: downloads the turbulent boundary layer record
  of example E03 on first use.

``plot_states_3d`` needs ``scikit-image``. Both optional packages come with
``pip install "surd[states]"``.
"""

import importlib

from . import datasets, it_tools
from .core import (
    nice_print,
    plot,
    plot_states,
    plot_states_3d,
    run,
    surd_states,
)

__all__ = [
    "surd_states",
    "run",
    "plot",
    "plot_states",
    "plot_states_3d",
    "nice_print",
    "it_tools",
    "analytic_eqs",
    "datasets",
]

_LAZY_SUBMODULES = ("analytic_eqs",)


def __getattr__(name):
    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_SUBMODULES))
