"""BM25-14: Okapi BM25 lexical retrieval with a prebuilt in-memory inverted index.

BM25 raw scores are query-dependent. The API's percentage is a *ranking indicator*,
obtained by normalizing to a virtual exact query document, NOT a probability.
No database, cloud API, PyTorch or downloaded model is required.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from math import log

from App.matching.attributes import _features
from App.matching.algorithms.lexical.lexical_core_14_15 import (
    checked_row, make_candidate, sort_candidates, stock_names,
    validated_top_n, word_tokens,
)

K1 = 1.2
B = .75


@dataclass(frozen=True)
class _Index:
    lengths: tuple[int, ...]
    average_length: float
    idf: dict[str, float]
    postings: dict[str, tuple[tuple[int, int], ...]]
    attributes: tuple[dict, ...]
    count: int


@lru_cache(maxsize=2)
def _prepare(names: tuple[str, ...]) -> _Index:
    n = len(names)
    postings = defaultdict(list)
    lengths = []
    for row_index, name in enumerate(names):
        counts = Counter(word_tokens(name))
        lengths.append(sum(counts.values()))
        for term, frequency in counts.items():
            postings[term].append((row_index, frequency))
    avgdl = (sum(lengths) / n) if n else 1.
    return _Index(
        lengths=tuple(lengths), average_length=max(avgdl, 1e-12),
        idf={term: log(1 + (n - len(rows) + .5) / (len(rows) + .5))
             for term, rows in postings.items()},
        postings={term: tuple(rows) for term, rows in postings.items()},
        attributes=tuple(_features(name) for name in names), count=n,
    )


def _tf_component(tf: int, doc_length: int, average_length: float) -> float:
    return tf * (K1 + 1) / (tf + K1 * (1 - B + B * doc_length / average_length))


def _scores(query: str, index: _Index) -> dict[int, float]:
    counts = Counter(word_tokens(query))
    if not counts or not index.count:
        return {}
    n = index.count
    query_length = sum(counts.values())
    # Normalize against the score of an exact document containing the query terms.
    # Unseen words contribute to the ideal, preventing high scores for partial queries.
    ideal = sum(log(1 + (n - len(index.postings.get(term, ())) + .5) /
                        (len(index.postings.get(term, ())) + .5)) *
                _tf_component(qtf, query_length, index.average_length)
                for term, qtf in counts.items())
    if ideal <= 0:
        return {}
    scores = defaultdict(float)
    for term in counts:
        for row_index, tf in index.postings.get(term, ()):
            scores[row_index] += index.idf[term] * _tf_component(
                tf, index.lengths[row_index], index.average_length)
    return {row_index: min(1.0, max(0., score / ideal)) for row_index, score in scores.items()
            if score > 0}


def match_products_bm25_14(checked_products, stock_products, top_n=5):
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
                index.attributes[stock_idx], 'BM25-14', 'Okapi BM25, query-normalized')
            if candidate is not None:
                candidates.append(candidate)
        row['candidates'] = sort_candidates(candidates, top_n)
        rows.append(row)
    return rows
