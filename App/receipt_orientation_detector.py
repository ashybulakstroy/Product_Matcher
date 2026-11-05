# -*- coding: utf-8 -*-
"""Определение ориентации чеков Торгсофт, используя RapidOCR из основного App.

Структура проекта:
    App/
    App/receipt_orientation_detector.py
    experiment_outputs/
    models/cyrillic_PP-OCRv5_rec_mobile.onnx

ДВА режима: normal (портрет пропускается без OCR), recovery (поиск по 4 углам).
1. Существующий веб-конвейер передаёт уже созданные engine и ocr_lock:
     image, info = orient_receipt(image, engine=self.engine, ocr_lock=self.ocr_lock)
2. Экспериментальный запуск создаёт RapidOCR с ТОЙ ЖЕ моделью из App.ocr_run.MODEL.

В модуле НЕ используются Tesseract, shell и дополнительные модели.
Коррекция EXIF должна выполняться перед вызовом (CLI делает это автоматически).
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np

VERSION = "receipt-orientation-rapidocr-1.4"

# Слова и выражения, специфичные для печатных предзаказов Торгсофт.
# Баллы помогают выбрать читаемую ориентацию, но сами по себе поворот не разрешают.
ANCHORS = {
    "ТАПСЫРЫС": 9, "ЗАКАЗ": 6, "ПРЕДЗАКАЗ": 7,
    "ПРОДАВЕЦ": 5, "САТУШЫ": 4,
    "ТАУАРДЫҢ": 5, "ТАУАРДЫН": 5, "АТАУЫ": 4,
    "САНЫ": 4, "БАҒАСЫ": 5, "БАГАСЫ": 5,
    "ҚОРЫТЫНДЫ": 6, "КОРЫТЫНДЫ": 6, "ИТОГО": 6,
    "НДС": 3, "ВСЕГО": 3, "ЦЕНА": 3, "КОЛИЧЕСТВО": 3,
}
HEADER_TERMS = ("ТАПСЫРЫС", "ПРЕДЗАКАЗ", "ПРОДАВЕЦ", "ЗАКАЗ", "САТУШЫ")
TABLE_TERMS = ("ТАУАРДЫҢ", "ТАУАРДЫН", "АТАУЫ", "САНЫ", "БАҒАСЫ", "БАГАСЫ", "ЦЕНА")
BOTTOM_TERMS = ("ҚОРЫТЫНДЫ", "КОРЫТЫНДЫ", "ИТОГО", "ВСЕГО")


def rotate_clockwise(image: np.ndarray, degrees: int) -> np.ndarray:
    """Только четверть оборота; cv2.rotate не интерполирует пиксели."""
    a = int(degrees) % 360
    codes = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
             270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    if a == 0:
        return image.copy()
    if a not in codes:
        raise ValueError("Допустимые повороты: 0, 90, 180, 270 градусов")
    return cv2.rotate(image, codes[a])


def _preview(image: np.ndarray, max_side: int) -> np.ndarray:
    """Уменьшенный кадр для дешёвого поиска ориентации, не для окончательного OCR."""
    h, w = image.shape[:2]
    factor = min(1.0, max_side / max(h, w))
    if factor == 1:
        return image
    return cv2.resize(image, (max(1, round(w * factor)), max(1, round(h * factor))),
                      interpolation=cv2.INTER_AREA)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.upper().replace("Ё", "Е")).strip()


def _ocr_features(lines: Any, width: int, height: int) -> dict[str, Any]:
    """RapidOCR result: [[quad(4x2), text, confidence], ...].

    В отличие от Tesseract TSV, Y-координаты берутся непосредственно из polygon
    каждого распознанного блока. Не ломаемся на пустом результате или отдельной
    некорректной записи OCR.
    """
    entries: list[dict[str, Any]] = []
    for item in lines or []:
        try:
            box, text, confidence = item[:3]
            arr = np.asarray(box, dtype=np.float32).reshape(-1, 2)
            if arr.shape[0] < 4 or not np.isfinite(arr).all():
                continue
            value = _norm(str(text))
            conf = float(confidence)
            if not value or not np.isfinite(conf) or conf < 0:
                continue
            entries.append({"text": value, "confidence": min(conf, 1.0),
                            "y": float(arr[:, 1].mean() / max(1, height))})
        except (ValueError, TypeError, IndexError):
            continue

    joined = " ".join(item["text"] for item in entries)
    # Слова могут быть разбиты на соседние OCR-блоки; объединённая строка
    # позволяет найти специфичные фразы и часть многострочных заголовков.
    anchors = sorted(term for term in ANCHORS if term in joined)
    anchor_score = sum(ANCHORS[a] for a in anchors)
    # Учитываем только реально читаемые буквенные токены (не цифры таблицы).
    word_count = len(re.findall(r"[А-ЯӘҒҚҢӨҰҮҺІA-Z]{3,}", joined))

    def first_y(terms: tuple[str, ...]) -> float | None:
        found = [e["y"] for e in entries if any(term in e["text"] for term in terms)]
        return round(float(min(found)), 4) if found else None

    return {
        "anchors": anchors,
        "anchor_count": len(anchors),
        "anchor_score": int(anchor_score),
        "word_count": int(word_count),
        "recognized_blocks": len(entries),
        "mean_confidence": round(float(np.mean([e["confidence"] for e in entries])), 4)
            if entries else 0.0,
        "header_y_normalized": first_y(HEADER_TERMS),
        "table_y_normalized": first_y(TABLE_TERMS),
        "total_y_normalized": first_y(BOTTOM_TERMS),
    }


def _barcode_location(image: np.ndarray, max_side: int = 1600) -> dict[str, Any]:
    """Геометрический поиск области штрихкода (не OCR и не чтение цифр).

    Если OpenCV contrib имеет barcode_BarcodeDetector, используем его.
    Иначе ищем пакет узких вертикальных штрихов морфологией. Возможны ложные
    срабатывания: поэтому heuristic лишь ВСПОМОГАТЕЛЬНЫЙ фактор.
    """
    im = _preview(image, max_side)
    h, w = im.shape[:2]
    if hasattr(cv2, "barcode_BarcodeDetector"):
        try:
            found, points = cv2.barcode_BarcodeDetector().detect(im)
            if found and points is not None:
                centers = np.asarray(points, dtype=np.float32).reshape((-1, 4, 2)).mean(axis=1)
                if len(centers):
                    return {"found": True, "source": "opencv_barcode",
                            "center_y_normalized": round(float(max(p[1] for p in centers)) / h, 4)}
        except (cv2.error, ValueError):
            pass

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    length = max(15, round(h * 0.018))
    bars = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (1, length)))
    merged = cv2.dilate(bars, cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(12, round(w * 0.035)), 5)))
    count, _, stats, _ = cv2.connectedComponentsWithStats(merged)
    candidates = []
    for x, y, box_w, box_h, area in stats[1:]:
        if (0.14*w <= box_w <= 0.75*w and
                0.035*h <= box_h <= 0.19*h and
                box_w / max(box_h, 1) >= 1.7 and
                0.01*h < y < 0.95*h):
            candidates.append((area, (y + box_h/2) / h))
    if candidates:
        _, cy = max(candidates)
        return {"found": True, "source": "vertical_bars_heuristic",
                "center_y_normalized": round(float(cy), 4)}
    return {"found": False, "source": "none", "center_y_normalized": None}


def _layout(features: dict[str, Any], barcode: dict[str, Any]) -> dict[str, Any]:
    header, table, total = (features[f"{key}_y_normalized"]
                            for key in ("header", "table", "total"))
    barcode_y = barcode["center_y_normalized"]
    score = 0.0
    evidence = []
    conflicts = []
    if header is not None and table is not None:
        if header + 0.02 < table:
            score += 5.0
            evidence.append("Шапка выше таблицы")
        elif table + 0.02 < header:
            score -= 10.0
            conflicts.append("Таблица выше шапки")
    if header is not None and total is not None:
        if header + 0.1 < total:
            score += 5.0
            evidence.append("Итог ниже шапки")
        elif total + 0.1 < header:
            score -= 12.0
            conflicts.append("Итог выше шапки")
    if header is not None and barcode_y is not None:
        dist = float(barcode_y - header)
        # Контурный barcode heuristic может перепутать линии таблицы,
        # поэтому его влияние меньше, чем подтверждённого BarcodeDetector.
        weight = 9.0 if barcode["source"] == "opencv_barcode" else 5.0
        if dist >= 0.16:
            score += weight
            evidence.append("Штрихкод ниже шапки")
        elif dist <= -0.12:
            score -= weight + 8.0
            conflicts.append("Штрихкод выше шапки")
    return {"score": round(score, 3), "evidence": evidence,
            "conflicts": conflicts, "header_found": header is not None,
            "layout_confirmed": bool(evidence) and not conflicts}


def _evaluate(image: np.ndarray, engine: Any, lock: Any, max_side: int) -> dict[str, Any]:
    preview = _preview(image, max_side)
    h, w = preview.shape[:2]
    with (lock if lock is not None else nullcontext()):
        result, _ = engine(preview)  # Тот же RapidOCR, что у App.
    feat = _ocr_features(result, w, h)
    # Штрихкод проверяем по полной повернутой фотографии независимо от OCR.
    barcode = _barcode_location(image)
    layout = _layout(feat, barcode)
    score = (feat["anchor_score"] +
             min(feat["word_count"], 30) * 0.12 +
             min(feat["recognized_blocks"], 30) * 0.05 +
             feat["mean_confidence"] * min(feat["recognized_blocks"], 10) * 0.2 +
             layout["score"])
    return {**feat, "barcode": barcode, "layout": layout, "score": round(score, 3)}


def _skipped_portrait_report(image: np.ndarray) -> dict[str, Any]:
    """Быстрый путь: прямоугольный книжный кадр не проверяем через OCR.

    Это НЕ доказательство правильной ориентации. Перевёрнутые на 180 градусов
    и вертикальные снимки с неверным расположением будут обработаны при
    recovery, если основной OCR не прочёл текст.
    """
    h, w = image.shape[:2]
    return {
        "version": VERSION,
        "mode": "normal",
        "status": "SKIPPED_PORTRAIT",
        "rotation_clockwise": 0,
        "proposed_rotation_clockwise": 0,
        "aspect_ratio_height_to_width": round(h / w, 4),
        "landscape_hint": False,
        "priority_candidates": [],
        "ocr_engine": "rapidocr_onnxruntime",
        "ocr_invocations": 0,
        "tesseract_used": False,
        "scores": {},
        "score_margin": None,
        "reason": "Книжный кадр: в обычном режиме OCR для ориентации не вызывается; при неудаче основного OCR используйте recovery",
        "original_size": [int(w), int(h)],
        "output_size": [int(w), int(h)],
    }


def detect_orientation(image: np.ndarray, *, engine: Any = None,
                       ocr_lock: Any = None, preview_max_side: int = 1600,
                       min_margin: float = 4.0, mode: str = "normal") -> dict[str, Any]:
    """Два режима определения ориентации.

    normal: книжное фото -> немедленный SKIP, никаких OCR-вызовов.
            альбомное -> проверка 90/270, при неуверенности также 0/180.
    recovery: после неудачного основного OCR проверить ВСЕ 4 положения
              независимо от отношения сторон (включая 180° для книжных).

    engine нужен только когда предстоит распознавание; для portrait/normal
    допускается None. Если подтверждения недостаточно, оставляем 0°.
    """
    if image is None or image.ndim not in (2, 3) or min(image.shape[:2]) < 80:
        raise ValueError("Некорректная фотография чека")
    if mode not in ("normal", "recovery"):
        raise ValueError("mode должен быть normal или recovery")

    h, w = image.shape[:2]
    landscape = h < w
    if mode == "normal" and not landscape:
        return _skipped_portrait_report(image)
    if engine is None or not callable(engine):
        raise ValueError("Для поиска ориентации нужен существующий экземпляр RapidOCR: engine")

    first, second = ([90, 270], [0, 180]) if landscape else ([0, 180], [90, 270])
    tested: dict[int, dict[str, Any]] = {}

    def evaluate_angle(degrees: int) -> None:
        if degrees not in tested:
            rotated = rotate_clockwise(image, degrees)
            tested[degrees] = _evaluate(rotated, engine, ocr_lock, preview_max_side)

    def best_two() -> tuple[int, int, float]:
        order = sorted(tested, key=lambda k: tested[k]["score"], reverse=True)
        best, runner_up = order[0], order[1]
        return best, runner_up, tested[best]["score"] - tested[runner_up]["score"]

    for angle in first:
        evaluate_angle(angle)

    best, _runner, margin = best_two()
    features = tested[best]
    reliable_early = (
        features["anchor_count"] >= 2
        and features["anchor_score"] >= 10
        and features["layout"]["layout_confirmed"]
        and len(features["layout"]["evidence"]) >= 2
        and margin >= max(6.0, min_margin)
        and not features["layout"]["conflicts"]
    )
    # В recovery обязательно исследуем все 4 угла: исходный OCR уже не смог
    # прочесть документ, поэтому экономить на проверке больше нельзя.
    if mode == "recovery" or not reliable_early:
        for angle in second:
            evaluate_angle(angle)
        best, _runner, margin = best_two()
        features = tested[best]

    # В отличие от OCR confidence учитываем физическую последовательность
    # частей чека: шапка выше товарной таблицы/итога и штрихкода.
    confirmed = (
        features["anchor_count"] >= 2
        and features["anchor_score"] >= 10
        and features["layout"]["header_found"]
        and features["layout"]["layout_confirmed"]
        and not features["layout"]["conflicts"]
        and margin >= min_margin
    )
    chosen = int(best) if confirmed else 0
    status = "OK" if confirmed else "UNCERTAIN"
    out_h, out_w = ((w, h) if chosen in (90, 270) else (h, w))
    return {
        "version": VERSION, "mode": mode, "status": status,
        "rotation_clockwise": chosen,
        "proposed_rotation_clockwise": int(best),
        "aspect_ratio_height_to_width": round(h / w, 4),
        "landscape_hint": bool(landscape),
        "priority_candidates": first,
        "ocr_engine": "rapidocr_onnxruntime", "tesseract_used": False,
        "ocr_invocations": len(tested),
        "scores": {str(k): tested[k] for k in sorted(tested)},
        "score_margin": round(margin, 3),
        "reason": (
            "Правильное расположение шапки, таблицы/итога и преимущество лучшего угла подтверждены"
            if confirmed else
            "Ориентация не подтверждена; возвращаем исходное изображение без поворота"
        ),
        "original_size": [int(w), int(h)],
        "output_size": [int(out_w), int(out_h)],
    }


def orient_receipt(image: np.ndarray, *, engine: Any = None,
                   ocr_lock: Any = None, preview_max_side: int = 1600,
                   min_margin: float = 4.0, mode: str = "normal") -> tuple[np.ndarray, dict[str, Any]]:
    """Возвращает повернутое изображение BGR и JSON-совместимый отчёт.

    При mode=normal и книжной ориентации engine=None допустим:
    RapidOCR в этом случае не загружается и не вызывается.
    """
    info = detect_orientation(
        image, engine=engine, ocr_lock=ocr_lock,
        preview_max_side=preview_max_side, min_margin=min_margin, mode=mode,
    )
    if info["rotation_clockwise"] == 0:
        return image, info  # не копируем изображение, если поворот не нужен
    return rotate_clockwise(image, info["rotation_clockwise"]), info


def _read_photo_with_exif(path: Path) -> np.ndarray:
    """Считать фото в BGR, учесть EXIF, не изменяя файл пользователя."""
    from PIL import Image, ImageOps
    with Image.open(path) as source:
        oriented = ImageOps.exif_transpose(source)
        if oriented.mode in {"RGBA", "LA"} or "transparency" in oriented.info:
            rgba = oriented.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            oriented = Image.alpha_composite(bg, rgba)
        rgb = np.asarray(oriented.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def create_project_rapidocr() -> Any:
    """CLI: тот же конструктор, те же пути и аргументы, что в App.web_pipeline."""
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise RuntimeError("rapidocr_onnxruntime отсутствует в текущем .venv") from exc
    from App.ocr_run import MODEL
    if not MODEL.is_file():
        raise FileNotFoundError(f"Не найдена модель из App.ocr_run.MODEL: {MODEL}")
    return RapidOCR(rec_model_path=str(MODEL), rec_img_shape=[3, 48, 320], text_score=0.3)


def main() -> int:
    parser = argparse.ArgumentParser(description="Быстрый normal и принудительный recovery для ориентации чека")
    parser.add_argument("photo", type=Path)
    parser.add_argument("--output", type=Path, default=Path("experiment_outputs/oriented.png"))
    parser.add_argument("--preview-max-side", type=int, default=1600)
    parser.add_argument("--mode", choices=("normal", "recovery"), default="normal",
                        help="normal: книжное пропустить; recovery: проверить все 4 поворота")
    args = parser.parse_args()
    if not args.photo.is_file():
        parser.error(f"Не найден файл: {args.photo}")
    if args.preview_max_side < 256:
        parser.error("--preview-max-side должен быть не меньше 256")

    image = _read_photo_with_exif(args.photo)
    # Не импортируем и не загружаем ONNX-модель на книжных кадрах в normal.
    need_ocr = args.mode == "recovery" or image.shape[0] < image.shape[1]
    engine = create_project_rapidocr() if need_ocr else None
    oriented, info = orient_receipt(image, engine=engine,
                                    preview_max_side=args.preview_max_side, mode=args.mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded_ok, encoded = cv2.imencode(".png", oriented)
    if not encoded_ok:
        raise RuntimeError("Не удалось закодировать PNG")
    encoded.tofile(str(args.output))
    info["input"] = str(args.photo)
    info["output"] = str(args.output)
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": info["status"], "mode": info["mode"], "ocr_invocations": info["ocr_invocations"], "rotation_clockwise": info["rotation_clockwise"],
                      "reason": info["reason"], "output": str(args.output),
                      "report": str(report_path)}, indent=2, ensure_ascii=False))
    return 0 if info["status"] in ("OK", "SKIPPED_PORTRAIT") else 2


if __name__ == "__main__":
    raise SystemExit(main())
