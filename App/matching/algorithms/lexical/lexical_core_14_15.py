"""Shared compatibility helpers for BM25-14 and TF-IDF-15.

Pure lexical retrieval; never mutates product data or converts identifiers to numbers.
The project's existing attribute penalties remain the sole source of attribute rules.
"""
from __future__ import annotations

import math
from App.matching.algorithms.fuzzy.cpu6_normalization import normalize_name
from App.matching.attributes import _features, apply_attribute_penalties

_STOP = frozenset({'и', 'или', 'для', 'с', 'со', 'по', 'из', 'на', 'в', 'во', 'шт', 'товар', 'изделие'})


def name_of(item: dict) -> str:
    return str(item.get('name') or item.get('product_name') or '')


def identifier_of(item: dict) -> str:
    value = item.get('id')
    if value is None or value == '':
        value = item.get('sku')
    return '' if value is None else str(value)


def word_tokens(name: str) -> tuple[str, ...]:
    # Use precisely the project's existing CPU6 normalization, without editing it.
    return tuple(t for t in normalize_name(name).split() if t and t not in _STOP)


def stock_names(stock: list[dict]) -> tuple[str, ...]:
    return tuple(name_of(product) for product in stock)


def checked_row(position: int, item: dict) -> dict:
    return dict(position_no=position, checked_id=identifier_of(item),
                checked_name=name_of(item), candidates=[])


def validated_top_n(top_n: int) -> int:
    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 0:
        raise ValueError('top_n должен быть неотрицательным целым числом')
    return top_n


def make_candidate(stock_item: dict, lexical_score: float, query_attrs: dict,
                   stock_attrs: dict, label: str, details: str) -> dict | None:
    if not math.isfinite(lexical_score) or lexical_score <= 0:
        return None
    lexical_score = max(0., min(1., lexical_score))
    reasons = [f'{label}: {lexical_score * 100:.2f}% ({details})']
    adjusted, reasons = apply_attribute_penalties(query_attrs, stock_attrs,
                                                  lexical_score * 100, reasons)
    adjusted = max(0., min(100., float(adjusted)))
    if adjusted <= 0:
        return None
    return dict(stock_id=identifier_of(stock_item), stock_name=name_of(stock_item),
                similarity_pct=round(adjusted, 2), raw_score=round(lexical_score, 6),
                match_reasons=reasons)


def sort_candidates(candidates: list[dict], top_n: int) -> list[dict]:
    candidates.sort(key=lambda candidate: (-candidate['similarity_pct'], candidate['stock_id']))
    return candidates[:top_n]
