"""semantic_minilm_13 — paraphrase-multilingual-MiniLM-L12-v2, mean pooling."""
from App.matching.algorithms.semantic.extra_runtime import match_dense


def match_products_semantic_minilm_13(checked_products, stock_products, top_n=5):
    return match_dense('semantic_minilm_13', checked_products, stock_products, top_n)
