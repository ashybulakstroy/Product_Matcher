"""semantic_rubert_12 — cointegrated/rubert-tiny2, CLS pooling, offline ONNX."""
from App.matching.algorithms.semantic.extra_runtime import match_dense


def match_products_semantic_rubert_12(checked_products, stock_products, top_n=5):
    return match_dense('semantic_rubert_12', checked_products, stock_products, top_n)
