"""Adapt CPU Matching-6 pair scores to our multi-row, full-candidate contract."""
import math
from functools import lru_cache

from App.matching.algorithms.fuzzy.cpu6_normalization import prepare_name


@lru_cache(maxsize=2)
def prepare_names(names):
    """Immutable representations reused between algorithms in each CPU worker."""
    return tuple(prepare_name(name) for name in names)


def match_prepared(score_pair, checked_products, stock_products, top_n, title):
    stock = list(stock_products)
    names = tuple(str(item.get('name') or item.get('product_name') or '') for item in stock)
    prepared = prepare_names(names)
    results = []
    for position, item in enumerate(checked_products, 1):
        name = str(item.get('name') or item.get('product_name') or '')
        query = prepare_name(name)
        candidates = []
        for target, target_name, target_prepared in zip(stock, names, prepared):
            score = float(score_pair(query, target_prepared))
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f'{title}: invalid score {score}')
            if score <= 0:
                continue
            candidates.append(dict(stock_id=target.get('id') or target.get('sku') or '',
                stock_name=target_name, similarity_pct=round(score * 100, 2), raw_score=score,
                match_reasons=[f'{title}: {score * 100:.2f}%']))
        candidates.sort(key=lambda candidate: (-candidate['similarity_pct'], str(candidate['stock_id'])))
        results.append(dict(position_no=position, checked_id=item.get('id') or item.get('sku') or '',
                            checked_name=name, candidates=candidates[:max(top_n, 0)]))
    return results
