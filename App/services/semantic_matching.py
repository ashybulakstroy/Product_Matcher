"""Compatibility module alias preserving the shared model/cache state."""
import sys
from App.matching.algorithms.semantic import e5 as _implementation
sys.modules[__name__] = _implementation
