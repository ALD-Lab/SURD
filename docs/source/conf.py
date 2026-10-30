# Sphinx configuration. Build with ``make html`` from docs/ after
# ``pip install "surd[docs]"``. The package is imported from ../src when it
# is not installed.

import sys
from pathlib import Path

sys.path.insert(0, str((Path(__file__).parent / ".." / ".." / "src").resolve()))

import surd  # noqa: E402

project = "surd"
copyright = "2024-2026, Álvaro Martínez-Sánchez, Gonzalo Arranz, Adrián Lozano-Durán"
author = "Álvaro Martínez-Sánchez, Gonzalo Arranz, Adrián Lozano-Durán"
release = surd.__version__
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

# Optional packages that are not needed to document the API.
autodoc_mock_imports = ["mpart", "pymp", "skimage", "sklearn"]
autodoc_member_order = "bysource"

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
html_static_path = []
