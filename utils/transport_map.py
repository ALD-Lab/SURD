"""Compatibility shim: ``import transport_map`` is ``surd.transport_map``.

Needs the optional ``mpart`` package.
"""

import sys

import surd.transport_map

sys.modules[__name__] = surd.transport_map
