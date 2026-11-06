"""Compatibility imports; implementation moved to App.matching."""
from App.matching.algorithms.hybrid.opencode_4 import _score, match_products_opencode_4
from App.matching.attributes import (
    BRAND_ALIASES, COLOR_ROOTS, COLOR_ENDINGS, TYPE_FAMILIES, TYPE_LOOKUP,
    _features, apply_attribute_penalties,
)
