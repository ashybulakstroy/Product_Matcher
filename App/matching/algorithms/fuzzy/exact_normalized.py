"""Exact / Normalized Match; adapted from CPU Matching-6, preserving its score formula."""
from App.matching.algorithms.fuzzy.cpu6_normalization import PreparedName
from App.matching.algorithms.fuzzy.cpu6_common import match_prepared


def score(query: PreparedName, product: PreparedName) -> float:
    return float(bool(query.normalized) and query.normalized == product.normalized)


def match_products_exact_normalized(checked_products, stock_products, top_n=5):
    return match_prepared(score, checked_products, stock_products, top_n, 'Exact / Normalized Match')
