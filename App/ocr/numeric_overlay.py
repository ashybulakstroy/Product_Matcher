"""Visual overlay for unresolved numeric cells on the rectified receipt table."""
from __future__ import annotations

import cv2

from App.ocr.receipt_editing import parse_number
from App.ocr.numeric_consensus_math import _supported

PAYABLE = {'product', 'service', 'tax'}


def _cell_bounds(layout: dict, row: dict, field: str):
    if field == 'quantity':
        left, right = layout['name_right'], layout['quantity_right']
    elif field == 'price':
        left, right = layout['quantity_right'], layout['right']
    else:
        return None
    top, bottom = row.get('y_range', (None, None))
    if top is None or bottom is None:
        return None
    return [int(round(left)), int(round(top)), int(round(right)), int(round(bottom))]


def overlay_invalid_numeric_cells(image, receipt: dict, layout: dict, *, alpha: float = 0.34,
                                  fill_bgr=(80, 80, 255), border_bgr=(20, 20, 220), render=True, numeric_report=None):
    """Return a rectified image with semi-transparent red highlight over bad numeric cells.

    Only unresolved table numeric cells are highlighted. Raw OCR data is not changed.
    """
    if image is None:
        raise ValueError('overlay_invalid_numeric_cells: image is required')
    output = image.copy() if render else image
    paint = image.copy() if render else None
    report = dict(status='ok', alpha=alpha, color_bgr=list(fill_bgr), cells=[])
    # User rule: mathematical validation (total AND VAT) suppresses ALL red cells.
    calculation = receipt.get('calculation', {}) if isinstance(receipt, dict) else {}
    if calculation.get('validation_status', calculation.get('status')) == 'match':
        report['status'] = 'mathematics_passed'
        return output, report
    rows = receipt.get('rows', []) if isinstance(receipt, dict) else []
    ids = {row.get('row'): row for row in rows}
    suspects = {}
    for row in rows:
        for field in row.get('invalid_fields') or []:
            if field in {'quantity', 'price'}:
                suspects[(row.get('row'), field)] = 'invalid_number'
    for attempt in (numeric_report or {}).get('all_numeric_attempts', []):
        field = attempt.get('field')
        row = ids.get(attempt.get('row'))
        if not row or field not in {'quantity', 'price'} or not attempt.get('agreed'):
            continue
        proposed = parse_number(attempt.get('candidate'))
        current = parse_number(row.get(field, row.get(field + '_raw', '')))
        if current is not None and proposed is not None and _supported(attempt, current):
            suspects.setdefault((row['row'], field), 'ocr_disagreement')
    for row in rows:
        if row.get('kind') not in PAYABLE:
            continue
        for (row_id, field), reason in suspects.items():
            if row_id != row.get('row'):
                continue
            bounds = _cell_bounds(layout, row, field)
            if not bounds:
                continue
            x0, y0, x1, y1 = bounds
            if render:
                cv2.rectangle(paint, (x0, y0), (x1, y1), fill_bgr, thickness=-1)
                cv2.rectangle(output, (x0, y0), (x1, y1), border_bgr, thickness=2)
            report['cells'].append(dict(row=row.get('row'), field=field, bounds=bounds,
                                         reason=reason))
    if report['cells'] and render:
        output = cv2.addWeighted(paint, alpha, output, 1.0 - alpha, 0.0)
    elif not report['cells']:
        report['status'] = 'no_invalid_numeric_cells'
    return output, report
