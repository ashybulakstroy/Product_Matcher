"""Levenshtein; adapted from CPU Matching-6, preserving its score formula."""
from rapidfuzz.distance import Levenshtein
from App.matching.algorithms.fuzzy.cpu6_normalization import PreparedName
from App.matching.algorithms.fuzzy.cpu6_common import match_prepared


def score(query: PreparedName, product: PreparedName) -> float:
    if not query.normalized or not product.normalized:
        return 0.0
    return Levenshtein.normalized_similarity(query.normalized, product.normalized)


def match_products_levenshtein(checked_products, stock_products, top_n=5):
    return match_prepared(score, checked_products, stock_products, top_n, 'Levenshtein')
