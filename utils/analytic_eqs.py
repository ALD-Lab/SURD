"""Compatibility shim: ``import analytic_eqs`` is ``surd.analytic_eqs``."""

import sys

import surd.analytic_eqs

sys.modules[__name__] = surd.analytic_eqs
