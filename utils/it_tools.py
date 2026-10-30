"""Compatibility shim: ``import it_tools`` is ``surd.it_tools``."""

import sys

import surd.it_tools

sys.modules[__name__] = surd.it_tools
