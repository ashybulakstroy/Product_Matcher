"""Определяет, НУЖЕН ли нелинейный Dewarping ПОСЛЕ гомографии.

Не изменяет изображение. Использует существующее App.wave_dewarp.find_rules,
но принимает решение по согласованной кривизне НЕСКОЛЬКИХ линий,
а не по одному наихудшему значению.

Использование из Python:
    decision = assess_dewarp_necessity(perspective_bgr)
    if decision['needs_dewarp']:
        ...

Запуск из корня проекта:
    .\\.venv\\Scripts\\python.exe -m App.dewarp_necessity_detector \
        "experiment_outputs\\...\\perspective.png" \
        --output "experiment_outputs\\...\\dewarp_need.json"
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from App.wave_dewarp import find_rules

VERSION = "dewarp-necessity-detector-1.1"


def _rule_profile(rule: Any) -> np.ndarray:
    """64 точки нелинейной составляющей, линейный наклон НЕ считается изгибом."""
    values = np.asarray(rule.values, dtype=np.float64)
    x = np.linspace(-1., 1., values.size)
    target_x = np.linspace(-1., 1., 64)
    sampled = np.interp(target_x, x, values)
    # Убираем наклон и сдвиг — тестируем только непрямолинейность.
    coefficients = np.polyfit(target_x, sampled, deg=1)
    residual = sampled - np.polyval(coefficients, target_x)
    return residual


def _axis_report(rules: list, *, span: int, axis: str) -> dict:
    """Ищет группу линий с похожим искривлением, без единичных выбросов."""
    # Должна быть заметная кривизна на размере фактического документа;
    # пороги — стартовые эмпирические, требуют регрессионной настройки.
    threshold = max(1.8 if axis == "horizontal" else 2.2,
                    span * (0.0033 if axis == "horizontal" else 0.0028))
    qualified = []
    candidates = []
    for rule in rules:
        coverage = float(rule.coverage)
        fit_error = float(rule.fit_error)
        if not (np.isfinite(rule.values).all() and coverage >= .72
                and fit_error <= 1.8):
            continue
        profile = _rule_profile(rule)
        amplitude = float(np.percentile(np.abs(profile), 95))
        info = {
            "level_px": round(float(np.median(rule.values)), 2),
            "curve_p95_px": round(amplitude, 3),
            "coverage": round(coverage, 3),
            "fit_error_px": round(fit_error, 3),
        }
        qualified.append(info)
        if amplitude >= threshold:
            candidates.append((info, profile))

    # Для горизонтали достаточно 3 согласованных линий, для вертикали 2.
    # При одном сильно изогнутом разделителе нельзя исключать артефакт
    # распознавания (например, нижняя граница строки №9).
    required = 3 if axis == "horizontal" else 2
    best_group: list[int] = []
    for center, (_, base) in enumerate(candidates):
        group = [center]
        for i, (_, profile) in enumerate(candidates):
            if i == center:
                continue
            a = base - np.mean(base)
            b = profile - np.mean(profile)
            denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
            similarity = float(a @ b / denominator) if denominator > 1e-5 else 0.
            # Подобное направление и форма кривой на независимых линиях.
            if similarity >= .78:
                group.append(i)
        if len(group) > len(best_group):
            best_group = group

    coherent = len(best_group) >= required
    return {
        "axis": axis,
        "detected_rules": len(rules),
        "qualified_rules": len(qualified),
        "curved_rules": len(candidates),
        "coherent_curved_rules": len(best_group),
        "required_coherent_rules": required,
        "threshold_px": round(float(threshold), 3),
        "coherent": coherent,
        "coherent_levels_px": [candidates[i][0]["level_px"] for i in best_group],
        "rules": qualified,
    }


def assess_dewarp_necessity(perspective: np.ndarray) -> dict:
    """Результат: SKIP / DEWARP / UNCERTAIN, без изменения пикселей.

    SKIP: после гомографии изгиб не доказан, возвращаем perspective.
    DEWARP: есть согласованная нелинейная деформация нескольких линий.
    UNCERTAIN: сетка не обнаружена надёжно, безопаснее SKIP.
    """
    if perspective is None or perspective.ndim not in (2, 3):
        raise ValueError("Ожидается изображение OpenCV (gray/BGR)")
    height, width = perspective.shape[:2]
    if min(width, height) < 90:
        return {
            "version": VERSION, "decision": "UNCERTAIN", "needs_dewarp": False,
            "reason": "Слишком маленькое изображение", "horizontal": None,
            "vertical": None,
        }

    horizontal, _ = find_rules(perspective, vertical=False)
    vertical, _ = find_rules(perspective, vertical=True)
    h = _axis_report(horizontal, span=width, axis="horizontal")
    v = _axis_report(vertical, span=height, axis="vertical")
    if h["qualified_rules"] < 3 and v["qualified_rules"] < 3:
        decision, reason = "UNCERTAIN", "Слишком мало достоверных печатных линий"
    elif h["coherent"] or v["coherent"]:
        decision, reason = "DEWARP", "Обнаружен согласованный изгиб нескольких печатных линий"
    else:
        decision, reason = "SKIP", "После Homography нет доказанной систематической кривизны"
    return {
        "version": VERSION,
        "decision": decision,
        "needs_dewarp": decision == "DEWARP",
        "reason": reason,
        "horizontal": h,
        "vertical": v,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка необходимости Dewarping после перспективы")
    parser.add_argument("photo", type=Path)
    parser.add_argument("--output", type=Path, default=Path("experiment_outputs/dewarp_need.json"))
    args = parser.parse_args()
    image = cv2.imdecode(np.fromfile(str(args.photo), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        parser.error(f"Не удалось прочитать {args.photo}")
    result = assess_dewarp_necessity(image)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("decision", "needs_dewarp", "reason")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
