import re
import unicodedata
from typing import Any, Dict, Iterable, List

from rapidfuzz import fuzz


def normalize_name(value: str) -> str:
    if value is None:
        return ""

    text = unicodedata.normalize("NFKC", str(value)).lower()
    text = text.replace("%", " ")
    text = text.replace("/", " ")
    text = text.replace("-", " ")
    text = text.replace("_", " ")
    text = re.sub(r"[^0-9а-яa-zё\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _score_name(left: str, right: str) -> float:
    left_norm = normalize_name(left)
    right_norm = normalize_name(right)
    if not left_norm or not right_norm:
        return 0.0

    raw_score = fuzz.ratio(left_norm, right_norm) / 100.0
    token_score = fuzz.token_sort_ratio(left_norm, right_norm) / 100.0
    return max(raw_score, token_score)


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


# Version 3 deliberately keeps its normalization separate from legacy algorithms.
_SQL3_UNITS = {
    "мм": ("length", 1), "см": ("length", 10), "м": ("length", 1000),
    "г": ("mass", 1), "кг": ("mass", 1000),
    "мл": ("volume", 1), "л": ("volume", 1000),
}


def _sql3_features(name):
    from decimal import Decimal
    text = unicodedata.normalize("NFKC", str(name or "")).lower().replace("ё", "е")
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)
    attributes = {}

    def measure(match):
        dimension, factor = _SQL3_UNITS[match[2]]
        attributes.setdefault(dimension, set()).add(Decimal(match[1]) * factor)
        return " "

    text = re.sub(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(мм|см|мл|кг|г|л|м)(?![а-яa-z])", measure, text)
    # Preserve model identifiers such as МР-3, МР 3, МР3 as one token.
    text = re.sub(r"([а-яa-z]+)[-\s]+(\d)", r"\1\2", text)
    tokens = re.findall(r"[а-яa-z0-9]+(?:\.[0-9]+)?", text)
    stop = {"и", "или", "для", "с", "со", "по", "из", "на", "в", "во", "шт", "товар", "изделие"}
    words = tuple(dict.fromkeys(t for t in tokens if t not in stop and not any(c.isdigit() for c in t) and len(t) > 2))
    codes = set(t for t in tokens if any(c.isdigit() for c in t))
    return words, attributes, codes


def _sql3_compare(left, right, weights):
    words, attributes, codes = left
    other_words, other_attributes, other_codes = right
    if not words and not codes:
        return 0.0, ["Недостаточно признаков"]
    available = set(other_words)
    matched = 0.0
    reasons = []
    # Longest/rarest words first; one candidate word cannot match multiple query words.
    for word in sorted(words, key=lambda w: (-weights.get(w, 1), -len(w), w)):
        best, quality = None, 0.0
        for other in sorted(available):
            if word == other:
                value = 1.0
            elif min(len(word), len(other)) >= 5 and fuzz.ratio(word, other) >= 88:
                value = 0.8
            else:
                value = 0.0
            if value > quality:
                best, quality = other, value
        if best is not None:
            available.remove(best)
            matched += weights.get(word, 1) * quality
    total = sum(weights.get(w, 1) for w in words)
    coverage = matched / total if total else 1.0
    if words and not matched:
        return 0.0, ["Название не совпало"]
    # Extra descriptive words reduce a complete match only slightly.
    precision = (len(other_words) - len(available)) / max(len(other_words), 1)
    score = 100 * coverage * (0.9 + 0.1 * precision)
    reasons.append(f"Совпадение слов: {coverage:.0%}")
    for dimension, values in attributes.items():
        actual = other_attributes.get(dimension)
        label = {"length": "длина/размер", "mass": "масса", "volume": "объём"}[dimension]
        if actual is None:
            score *= 0.9
            reasons.append(f"Характеристика {label}: отсутствует")
        elif values == actual:
            reasons.append(f"Характеристика {label}: совпала")
        else:
            score *= 0.35
            reasons.append(f"Характеристика {label}: отличается")
    if codes:
        if codes == other_codes:
            reasons.append("Модель/числа совпали")
        elif not other_codes:
            score *= 0.8
            reasons.append("Модель/числа отсутствуют")
        else:
            score *= 0.3
            reasons.append("Модель/числа отличаются")
    return round(min(score, 100), 2), reasons


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
