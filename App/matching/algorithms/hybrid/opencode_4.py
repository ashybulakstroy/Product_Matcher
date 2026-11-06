"""Attribute reranking inspired by OpenCode failure analysis."""
from collections import Counter
from math import log
from App.matching.attributes import _features, apply_attribute_penalties
from App.matching.token_scoring import _sql3_compare


def _score(query, target, weights):
    # Numbers are compared separately rather than conflating articles with measurements.
    score, reasons = _sql3_compare((query['words'], {}, set()), (target['words'], {}, set()), weights)
    if not query['words']:
        shared = any(query[key] & target[key] for key in ('articles', 'models'))
        score = 100.0 if shared else 0.0
        reasons = ['Поиск по артикулу/модели']
    if not score:
        return 0.0, reasons
    return apply_attribute_penalties(query, target, score, reasons)


def match_products_opencode_4(checked_products, stock_products, top_n=5):
    stock = list(stock_products)
    prepared = [(item, _features(item.get('name') or item.get('product_name'))) for item in stock]
    counts = Counter(word for _, features in prepared for word in set(features['words']))
    weights = {word: 1 + log((1 + len(stock)) / (1 + count)) for word, count in counts.items()}
    result = []
    for position, item in enumerate(checked_products, 1):
        name = str(item.get('name') or item.get('product_name') or '')
        query = _features(name)
        candidates = []
        for target, features in prepared:
            score, reasons = _score(query, features, weights)
            if score > 0:
                candidates.append(dict(stock_id=target.get('id') or target.get('sku') or '',
                    stock_name=str(target.get('name') or target.get('product_name') or ''),
                    similarity_pct=score, match_reasons=reasons))
        candidates.sort(key=lambda c: (-c['similarity_pct'], str(c['stock_id'])))
        result.append(dict(position_no=position, checked_id=item.get('id') or item.get('sku') or '',
                           checked_name=name, candidates=candidates[:top_n]))
    return result
