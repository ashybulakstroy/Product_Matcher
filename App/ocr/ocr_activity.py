"""Понятные пользовательские статусы OCR. Не зависит от веб-фреймворка.

Счетчик строк не выводится из количества фрагментов OCR / 4.
Границы строк берутся из предварительно найденной сетки.
"""
from __future__ import annotations

from bisect import bisect_right
from math import isfinite


def preliminary_row_count(boundaries) -> int:
    """Сколько диапазонов строк реально установлено детектором сетки."""
    if not boundaries:
        return 0
    try:
        values = [float(v) for v in boundaries]
    except (ValueError, TypeError):
        return 0
    if any(not isfinite(v) for v in values):
        return 0
    return sum(b - a >= 8 for a, b in zip(values, values[1:]) if b > a)


def completed_text_rows(boxes, processed: int, boundaries) -> tuple[int, int]:
    """Количество обработанных строк из строк, где детектор нашёл текст.

    Все обнаруженные текстовые фрагменты полосы должны пройти OCR, прежде
    чем полоса считается обработанной. Не утверждает, что текст распознан верно.
    Пустые полосы не учитываются в знаменателе.
    """
    try:
        limits = [float(value) for value in boundaries]
    except (TypeError, ValueError):
        return 0, 0
    if len(limits) < 2 or any(not isfinite(v) for v in limits):
        return 0, 0
    if any(b <= a for a, b in zip(limits, limits[1:])):
        return 0, 0
    last_seen = {}
    for index, box in enumerate(boxes):
        try:
            middle_y = sum(float(point[1]) for point in box) / len(box)
        except (TypeError, ValueError, IndexError, ZeroDivisionError):
            continue
        if not isfinite(middle_y) or not (limits[0] <= middle_y < limits[-1]):
            continue
        row = bisect_right(limits, middle_y) - 1
        if 0 <= row < len(limits) - 1:
            last_seen[row] = max(last_seen.get(row, -1), index)
    done = sum(index < processed for index in last_seen.values())
    return done, len(last_seen)


def row_totals(rows) -> dict:
    """Счёт только по реально разобранным строкам, с учетом исключений."""
    kinds = [row.get('kind') for row in rows]
    return {
        'products': sum(kind == 'product' for kind in kinds),
        'tax_rows': sum(kind == 'tax' for kind in kinds),
        'other_rows': sum(kind not in ('product', 'tax') for kind in kinds),
    }


def emit_status(job, stage: str, title: str, **payload) -> None:
    """Отдельный SSE-тип, не меняющий бизнес-результат и прогресс сам по себе."""
    job.publish('ui_status', stage=stage, title=title, **payload)
