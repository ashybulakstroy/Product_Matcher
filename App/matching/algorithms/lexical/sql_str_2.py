from typing import Any, Dict, Iterable, List
from rapidfuzz import fuzz
from App.matching.normalization import normalize_name


def _meaningful_tokens(text: str) -> List[str]:
    norm = normalize_name(text)
    if not norm:
        return []

    stop_words = {
        "и", "или", "для", "с", "со", "по", "из", "на", "в", "во", "изделие", "товар",
        "шт", "см", "мм", "г", "кг", "л", "мл", "pcs", "pc", "set", "unit", "комплект"
    }

    tokens = []
    for token in norm.split():
        if token.isdigit():
            continue
        if len(token) <= 2:
            continue
        if token in stop_words:
            continue
        tokens.append(token)
    return tokens


def _sql_like_score(left: str, right: str) -> float:
    left_norm = normalize_name(left)
    right_norm = normalize_name(right)
    if not left_norm or not right_norm:
        return 0.0

    if left_norm in right_norm:
        return round((len(left_norm) / max(len(right_norm), 1)) * 100, 2)
    if right_norm in left_norm:
        return round((len(right_norm) / max(len(left_norm), 1)) * 100, 2)

    left_tokens = _meaningful_tokens(left_norm)
    right_tokens = _meaningful_tokens(right_norm)
    if not left_tokens or not right_tokens:
        return 0.0

    right_set = set(right_tokens)
    matched_positions = []

    for index, token in enumerate(left_tokens, start=1):
        weight = 1.0 + ((len(left_tokens) - index) / max(len(left_tokens), 1))
        if token in right_set:
            matched_positions.append(weight)
            continue
        if any(token in other or other in token for other in right_tokens):
            matched_positions.append(weight)
            continue
        if len(left_tokens) > 1:
            fuzzy_scores = [fuzz.partial_ratio(token, other) for other in right_tokens]
            if fuzzy_scores and max(fuzzy_scores) >= 88:
                matched_positions.append(weight * 0.75)

    if not matched_positions:
        return 0.0

    score = (sum(matched_positions) / sum((1.0 + ((len(left_tokens) - i) / max(len(left_tokens), 1))) for i in range(1, len(left_tokens) + 1))) * 100.0
    return round(min(score, 100.0), 2)


def match_products_sql_str_2(
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
            similarity = _sql_like_score(checked_name, stock_name)

            if similarity > 0:
                candidates.append(
                    {
                        "stock_id": stock_id,
                        "stock_name": stock_name,
                        "similarity_pct": round(similarity, 2),
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
