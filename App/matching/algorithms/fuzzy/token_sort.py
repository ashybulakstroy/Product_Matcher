"""Token Sort Ratio; adapted from CPU Matching-6, preserving its score formula."""
from rapidfuzz import fuzz
from App.matching.algorithms.fuzzy.cpu6_normalization import PreparedName
from App.matching.algorithms.fuzzy.cpu6_common import match_prepared


def score(query: PreparedName, product: PreparedName) -> float:
    if not query.sorted_words or not product.sorted_words:
        return 0.0
    return fuzz.ratio(query.sorted_words, product.sorted_words) / 100.0


def match_products_token_sort(checked_products, stock_products, top_n=5):
    return match_prepared(score, checked_products, stock_products, top_n, 'Token Sort Ratio')
