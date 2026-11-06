from App.matching.attributes import _sql3_features
from App.matching.token_scoring import _sql3_compare


def match_products_sql_str_3(checked_products, stock_products, top_n=5):
    from collections import Counter
    from math import log
    stock = list(stock_products)
    prepared = [(item, _sql3_features(item.get("name") or item.get("product_name") or "")) for item in stock]
    counts = Counter(word for _, features in prepared for word in set(features[0]))
    weights = {word: 1 + log((1 + len(stock)) / (1 + count)) for word, count in counts.items()}
    result = []
    for position, item in enumerate(checked_products, 1):
        name = str(item.get("name") or item.get("product_name") or "")
        features = _sql3_features(name)
        candidates = []
        for target, target_features in prepared:
            score, reasons = _sql3_compare(features, target_features, weights)
            if score > 0:
                candidates.append(dict(stock_id=target.get("id") or target.get("sku") or "",
                    stock_name=str(target.get("name") or target.get("product_name") or ""),
                    similarity_pct=score, match_reasons=reasons))
        candidates.sort(key=lambda c: (-c["similarity_pct"], str(c["stock_id"])))
        result.append(dict(position_no=position, checked_id=item.get("id") or item.get("sku") or "",
                           checked_name=name, candidates=candidates[:top_n]))
    return result
