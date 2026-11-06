from rapidfuzz import fuzz


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
