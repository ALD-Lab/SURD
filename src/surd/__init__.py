"""SURD: Synergistic-Unique-Redundant Decomposition of causality.

SURD measures the information that past observations of a set of source
variables carry about the future of a target variable and splits it into
redundant, unique and synergistic contributions. The remaining uncertainty
is the causality leak from unobserved variables.

Reference: Martínez-Sánchez, Arranz & Lozano-Durán, "Decomposing causality
into its synergistic, unique, and redundant components", Nature
Communications 15, 9296 (2024). https://doi.org/10.1038/s41467-024-53373-4

Typical use::

    import numpy as np
    import surd

    hist, _ = np.histogramdd(Y.T, bins)      # Y[0] target, Y[1:] sources
    I_R, I_S, MI, info_leak = surd.surd(hist)
    surd.nice_print(I_R, I_S, MI, info_leak)

Submodules:

- ``surd.it_tools``: entropies, mutual information and transfer entropy
  on discrete distributions.
- ``surd.analytic_eqs``: the stochastic benchmark systems of the paper
  (mediator, confounder, synergistic and redundant colliders).
- ``surd.transport_map``: density estimation with triangular transport
  maps. Imported on first access and needs the optional ``mpart`` package.
- ``surd.states``: causality by states (Martínez-Sánchez & Lozano-Durán,
  Communications Physics 9, 15, 2025). The decomposition resolved over
  the states of the variables, with its own ``it_tools`` and
  ``analytic_eqs``. Imported on first access.
- ``surd.neural``: the specific mutual information estimated with a
  classifier instead of the histogram, for short records or many sources.
  Imported on first access and needs PyTorch.
"""

import importlib

from . import it_tools
from .core import (
    nice_print,
    plot,
    plot_multiple_lags,
    plot_nlabels,
    run,
    run_multiple_lags,
    run_parallel,
    surd,
    assign,
    surd_hd,
)

__version__ = "1.3.0"

__all__ = [
    "surd",
    "assign",
    "surd_hd",
    "run",
    "run_parallel",
    "run_multiple_lags",
    "plot",
    "plot_nlabels",
    "plot_multiple_lags",
    "nice_print",
    "it_tools",
    "analytic_eqs",
    "transport_map",
    "states",
    "neural",
    "__version__",
]

# Submodules loaded on first access. ``analytic_eqs`` seeds NumPy's global
# random generator when imported, ``transport_map`` needs ``mpart``,
# ``states`` is a package of its own and ``neural`` needs PyTorch, so none
# is imported by a plain ``import surd``.
_LAZY_SUBMODULES = ("analytic_eqs", "transport_map", "states", "neural")


def __getattr__(name):
    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_SUBMODULES))
