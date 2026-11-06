"""Targeted OCR for unreadable receipt row numbers. Zero full-page OCR calls.

Input: the already rectified BGR table, App's parsed rows and layout.
Recognize only suspect number-column cells, never replace raw OCR evidence.
Uses an existing RapidOCR text_recognizer (not text_detector).
"""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from typing import Callable
import re
from time import perf_counter

import cv2
import numpy as np

VERSION = 'selective-number-ocr-1.0'
EXCLUDED_KINDS = {'header', 'total', 'subtotal'}
DOMINANT_CONFIDENCE = 0.40
WEAK_ALTERNATIVE_CONFIDENCE = 0.20


def parse_index(text: object) -> int | None:
    """Only literal Arabic digit strings count as an OCR reading.

    Never infer numbers from missing gaps or from text such as 'I'/'l'.
    """
    s = re.sub(r'\s+', '', str(text or ''))
    if not re.fullmatch(r'\d{1,3}', s, flags=re.ASCII):
        return None
    n = int(s)
    return n if 1 <= n <= 999 else None


def suspect(row: dict) -> tuple[bool, str]:
    if row.get('kind') in EXCLUDED_KINDS:
        return False, 'non_item_row'
    raw = parse_index(row.get('item_number_raw'))
    expected = row.get('row')
    try:
        expected = int(expected)
    except (TypeError, ValueError):
        expected = None
    if raw is None:
        return True, 'unreadable_or_missing_number'
    if expected is not None and raw != expected:
        return True, 'sequence_conflict'
    return False, 'already_readable'


def _safe_bounds(image: np.ndarray, row: dict, layout: dict) -> tuple[int, int, int, int] | None:
    """Use the existing physical cell bounds, never the name/quantity columns."""
    h, w = image.shape[:2]
    y_range = row.get('y_range')
    if not isinstance(y_range, (list, tuple)) or len(y_range) != 2:
        return None
    try:
        x0 = int(round(float(layout.get('body_left', 0)))) + 3
        x1 = int(round(float(layout['name_left']))) - 3
        y0 = int(round(float(y_range[0]))) + 3
        y1 = int(round(float(y_range[1]))) - 3
    except (TypeError, ValueError, KeyError):
        return None
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)
    if x1 - x0 < 8 or y1 - y0 < 9:
        return None
    return x0, y0, x1, y1


def _text_bbox(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    """Find a compact ink ROI without chopping thin '1' glyphs.

    The printed table rules are excluded by the cell inset; additional long
    straight borders are ignored as connected components.
    """
    hh, ww = gray.shape
    enhanced = cv2.GaussianBlur(gray, (3, 3), 0)
    _, mask = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    components = []
    for idx in range(1, count):
        x, y, w, h, area = [int(v) for v in stats[idx]]
        if area < 2 or h < max(3, int(hh * 0.15)):
            continue
        if (w >= ww * 0.78 and h <= max(3, hh * .15)) or (h >= hh * .89 and w <= 2):
            continue
        if (x < 2 or x + w > ww - 2) and h >= hh * .85:
            continue
        components.append((x, y, x + w, y + h))
    if not components:
        return None
    x0 = max(0, min(p[0] for p in components) - 4)
    y0 = max(0, min(p[1] for p in components) - 4)
    x1 = min(ww, max(p[2] for p in components) + 4)
    y1 = min(hh, max(p[3] for p in components) + 4)
    if x1 - x0 < 3 or y1 - y0 < 4:
        return None
    return x0, y0, x1, y1


def make_views(image: np.ndarray, row: dict, layout: dict) -> dict | None:
    """Produce two BGR variants, each with 3 channels as required by RapidOCR.

    One is an unaltered-color tight crop; one is CLAHE-enhanced. This uses a
    small cell only. Avoid aggressive binary morphology that may erase '1'.
    """
    bounds = _safe_bounds(image, row, layout)
    if bounds is None:
        return None
    x0, y0, x1, y1 = bounds
    cell = image[y0:y1, x0:x1]
    if cell.ndim == 2:
        cell = cv2.cvtColor(cell, cv2.COLOR_GRAY2BGR)
    elif cell.ndim == 3 and cell.shape[2] == 4:
        cell = cv2.cvtColor(cell, cv2.COLOR_BGRA2BGR)
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    bbox = _text_bbox(gray)
    if bbox is None:
        return None
    bx0, by0, bx1, by1 = bbox
    tight = cell[by0:by1, bx0:bx1]
    if tight.size == 0:
        return None

    def prepare(src: np.ndarray, *, enhance: bool) -> np.ndarray:
        if enhance:
            g = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(g)
            src = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        # Enlarging a tight crop makes a thin solitary glyph visible to the
        # recognizer without forcing a wide empty cell into its 320px input.
        resized = cv2.resize(src, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        pad = max(7, int(resized.shape[0] * .20))
        return cv2.copyMakeBorder(resized, pad, pad, pad, pad,
                                  cv2.BORDER_CONSTANT, value=(255, 255, 255))

    return {
        'row': row.get('row'),
        'cell_box': [x0, y0, x1, y1],
        'digit_box': [x0 + bx0, y0 + by0, x0 + bx1, y0 + by1],
        'views': [prepare(tight, enhance=False), prepare(tight, enhance=True)],
    }


def _recognizer_results(recognize: Callable, images: list[np.ndarray]) -> list[tuple[str, float]]:
    """Accept text_recognizer(crops)[0] or a test callback returning pairs."""
    outcome = recognize(images)
    # Typical existing App wrapper is: lambda crops: engine.text_recognizer(crops)[0]
    if (isinstance(outcome, tuple) and len(outcome) == 2 and
            isinstance(outcome[0], (list, tuple)) and not isinstance(outcome[0], str)):
        outcome = outcome[0]
    if len(outcome) != len(images):
        raise RuntimeError('Recognizer returned a different count of results')
    result = []
    for item in outcome:
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            result.append(('', 0.0))
            continue
        try:
            confidence = float(item[1])
        except (TypeError, ValueError):
            confidence = 0.0
        result.append((str(item[0]).strip(), confidence))
    return result


def recover_number_cells(
    image: np.ndarray,
    rows: list[dict],
    layout: dict,
    recognize: Callable[[list[np.ndarray]], object],
    *,
    ocr_lock=None,
    min_confidence: float = 0.20,
    max_cells: int = 16,
    only_rows: set[int] | None = None,
    save_crop: Callable[[int, int, np.ndarray], None] | None = None,
) -> tuple[list[dict], dict]:
    """Recognize suspect cells only; return *copies* and a transparent report.

    2 views per suspect cell, batched into one text_recognizer invocation.
    Two agreeing views from the *same* model are not independent confirmation;
    leave low-confidence or sequence-conflicting results unresolved.
    """
    if max_cells <= 0 or not 0 <= min_confidence <= 1:
        raise ValueError('Invalid limits')
    started = perf_counter()
    updated = deepcopy(rows)
    attempts: list[dict] = []
    planned = []
    for idx, row in enumerate(updated):
        if only_rows is not None and row.get('row') not in only_rows:
            continue
        flagged, reason = suspect(row)
        if not flagged:
            continue
        if len(planned) >= max_cells:
            attempts.append(dict(row=row.get('row'), reason=reason, status='skipped_limit'))
            continue
        extracted = make_views(image, row, layout)
        if extracted is None:
            attempts.append(dict(row=row.get('row'), reason=reason, status='no_readable_crop'))
            continue
        planned.append((idx, row, reason, extracted))
    if planned:
        batch = []
        for _, _, _, extracted in planned:
            for j, view in enumerate(extracted['views']):
                if save_crop is not None:
                    save_crop(int(extracted['row']), j, view)
                batch.append(view)
        with (ocr_lock if ocr_lock is not None else nullcontext()):
            recognized = _recognizer_results(recognize, batch)
        for i, (idx, row, reason, extracted) in enumerate(planned):
            views = recognized[2 * i:2 * i + 2]
            values = [parse_index(text) for text, _ in views]
            expected = row.get('row')
            try:
                expected = int(expected)
            except (TypeError, ValueError):
                expected = None
            agreed = values[0] is not None and values[0] == values[1]
            confident = agreed and min(c for _, c in views) >= min_confidence
            dominant_candidates = [
                (value, confidence, other_value, other_confidence)
                for (value, (_, confidence)), (other_value, (_, other_confidence))
                in ((pair, reversed_pair) for pair, reversed_pair in (
                    ((values[0], views[0]), (values[1], views[1])),
                    ((values[1], views[1]), (values[0], views[0])),
                ))
                if value is not None
                and confidence >= DOMINANT_CONFIDENCE
                and other_value is None
                and other_confidence < WEAK_ALTERNATIVE_CONFIDENCE
            ]
            dominant = dominant_candidates[0][0] if dominant_candidates else None
            candidate = values[0] if confident else dominant
            sequence_ok = expected is None or candidate == expected
            accepted = bool(candidate is not None and sequence_ok)
            if dominant is not None:
                status = 'recognized_dominant_view' if accepted else 'sequence_conflict_after_ocr'
            elif not agreed:
                status = 'unconfirmed_readings'
            elif not confident:
                status = 'insufficient_confidence'
            elif not sequence_ok:
                status = 'sequence_conflict_after_ocr'
            else:
                status = 'recognized'
            attempt = dict(row=row.get('row'), reason=reason, status=status,
                           raw_number=row.get('item_number_raw', ''),
                           expected_sequence=expected,
                           cell_box=extracted['cell_box'],
                           digit_box=extracted['digit_box'],
                           readings=[dict(text=text, confidence=round(conf, 4)) for text, conf in views],
                           recognized_number=candidate if accepted else None)
            attempts.append(attempt)
            if accepted:
                row['item_number_recovered'] = str(candidate)
                row['item_number_recovery_source'] = 'targeted_rapidocr'
                row['item_number_recovery_confidence'] = round(min(c for _, c in views), 4)
    report = dict(version=VERSION, total_rows=len(rows), selected_cells=len(planned),
                  elapsed_ms=round(1000*(perf_counter()-started), 1),
                  recognizer_calls=1 if planned else 0, views=len(planned) * 2,
                  recovered=sum(a['status'] in {'recognized', 'recognized_dominant_view'} for a in attempts),
                  attempts=attempts, modified_raw_ocr=False,
                  note='Row sequence is only a validation gate, never a generated OCR result')
    return updated, report
