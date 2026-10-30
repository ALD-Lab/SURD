"""Shared configuration for the test suite.

The tests run against the installed package when there is one. From a
plain clone without installation, the source tree under ``src/`` is used.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    import surd  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
