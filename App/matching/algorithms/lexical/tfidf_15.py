"""TF-IDF-15: sparse word + character 3..5-gram cosine, pure Python on CPU.

Particularly useful for near-matching product names and OCR spelling mistakes.
Not semantic synonym matching; it complements, rather than replaces, E5.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from math import log, sqrt

from App.matching.attributes import _features
from App.matching.algorithms.lexical.lexical_core_14_15 import (
    checked_row, make_candidate, sort_candidates, stock_names,
    validated_top_n, word_tokens,
)


@dataclass(frozen=True)
class _Index:
    idf: dict[str, float]
    postings: dict[str, tuple[tuple[int, float], ...]]
    attributes: tuple[dict, ...]
    count: int


def _terms(name: str) -> Counter:
    tokens = word_tokens(name)
    terms = Counter()
    for token in tokens:
        terms['w:' + token] += 1
        padded = '^' + token + '$'
        for width in (3, 4, 5):
            for start in range(len(padded) - width + 1):
                terms[f'c{width}:' + padded[start:start + width]] += 1
    return terms


def _weights(counts: Counter, idf: dict[str, float], count: int) -> dict[str, float]:
    output = {}
    for term, tf in counts.items():
        # Include absent query features in its norm; they should lower cosine.
        weight_idf = idf.get(term, log(1 + count) + 1)
        emphasis = 2.25 if term.startswith('w:') else .65
        output[term] = (1 + log(tf)) * weight_idf * emphasis
    norm = sqrt(sum(value * value for value in output.values()))
    return {term: value / norm for term, value in output.items()} if norm else {}


@lru_cache(maxsize=2)
def _prepare(names: tuple[str, ...]) -> _Index:
    n = len(names)
    documents = [_terms(name) for name in names]
    frequency = Counter(term for doc in documents for term in doc)
    idf = {term: log((n + 1) / (df + 1)) + 1 for term, df in frequency.items()}
    postings = defaultdict(list)
    for row_index, terms in enumerate(documents):
        for term, normalized_weight in _weights(terms, idf, n).items():
            postings[term].append((row_index, normalized_weight))
    return _Index(idf=idf, postings={key: tuple(value) for key, value in postings.items()},
                  attributes=tuple(_features(name) for name in names), count=n)


def _scores(query: str, index: _Index) -> dict[int, float]:
    if not index.count:
        return {}
    vector = _weights(_terms(query), index.idf, index.count)
    scores = defaultdict(float)
    for term, qvalue in vector.items():
        for row_index, dvalue in index.postings.get(term, ()):
            scores[row_index] += qvalue * dvalue
    return {row_index: min(1., max(0., value)) for row_index, value in scores.items()
            if value > 0}


def match_products_tfidf_15(checked_products, stock_products, top_n=5):
    top_n = validated_top_n(top_n)
    stock = list(stock_products)
    index = _prepare(stock_names(stock))
    rows = []
    for position, checked in enumerate(checked_products, 1):
        row = checked_row(position, checked)
        query = row['checked_name']
        if not query.strip() or not top_n:
            rows.append(row)
            continue
        qattrs = _features(query)
        candidates = []
        for stock_idx, lexical_score in _scores(query, index).items():
            candidate = make_candidate(stock[stock_idx], lexical_score, qattrs,
                index.attributes[stock_idx], 'TF-IDF-15', 'word + char 3-5gram cosine')
            if candidate is not None:
                candidates.append(candidate)
        row['candidates'] = sort_candidates(candidates, top_n)
        rows.append(row)
    return rows
