"""Jaccard; adapted from CPU Matching-6, preserving its score formula."""
from App.matching.algorithms.fuzzy.cpu6_normalization import PreparedName
from App.matching.algorithms.fuzzy.cpu6_common import match_prepared


def score(query: PreparedName, product: PreparedName) -> float:
    a, b = (query.word_set, product.word_set)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def match_products_jaccard(checked_products, stock_products, top_n=5):
    return match_prepared(score, checked_products, stock_products, top_n, 'Jaccard')
