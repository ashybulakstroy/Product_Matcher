from typing import Any, Dict, Iterable, List
from rapidfuzz import fuzz
from App.matching.normalization import normalize_name


def _score_name(left: str, right: str) -> float:
    left_norm = normalize_name(left)
    right_norm = normalize_name(right)
    if not left_norm or not right_norm:
        return 0.0

    raw_score = fuzz.ratio(left_norm, right_norm) / 100.0
    token_score = fuzz.token_sort_ratio(left_norm, right_norm) / 100.0
    return max(raw_score, token_score)


def match_products(
    checked_products: Iterable[Dict[str, Any]],
    stock_products: Iterable[Dict[str, Any]],
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    checked_list = list(checked_products)
    stock_list = list(stock_products)
    result: List[Dict[str, Any]] = []

    for position_no, item in enumerate(checked_list, start=1):
        checked_name = str(item.get("name") or item.get("product_name") or "")
        checked_id = item.get("id") or item.get("sku") or ""

        candidates = []
        for stock_item in stock_list:
            stock_name = str(stock_item.get("name") or stock_item.get("product_name") or "")
            stock_id = stock_item.get("id") or stock_item.get("sku") or ""
            similarity = _score_name(checked_name, stock_name)

            if similarity > 0:
                candidates.append(
                    {
                        "stock_id": stock_id,
                        "stock_name": stock_name,
                        "similarity_pct": round(similarity * 100, 2),
                    }
                )

        candidates.sort(key=lambda entry: (-entry["similarity_pct"], entry["stock_id"]))

        result.append(
            {
                "position_no": position_no,
                "checked_id": checked_id,
                "checked_name": checked_name,
                "candidates": candidates[:top_n],
            }
        )

    return result
