"""Visual overlay for unresolved numeric cells on the rectified receipt table."""
from __future__ import annotations

import cv2

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
                                  fill_bgr=(80, 80, 255), border_bgr=(20, 20, 220), render=True):
    """Return a rectified image with semi-transparent red highlight over bad numeric cells.

    Only unresolved table numeric cells are highlighted. Raw OCR data is not changed.
    """
    if image is None:
        raise ValueError('overlay_invalid_numeric_cells: image is required')
    output = image.copy() if render else image
    paint = image.copy() if render else None
    report = dict(status='ok', alpha=alpha, color_bgr=list(fill_bgr), cells=[])
    rows = receipt.get('rows', []) if isinstance(receipt, dict) else []
    for row in rows:
        if row.get('kind') not in PAYABLE:
            continue
        invalid = row.get('invalid_fields') or []
        for field in invalid:
            bounds = _cell_bounds(layout, row, field)
            if not bounds:
                continue
            x0, y0, x1, y1 = bounds
            if render:
                cv2.rectangle(paint, (x0, y0), (x1, y1), fill_bgr, thickness=-1)
                cv2.rectangle(output, (x0, y0), (x1, y1), border_bgr, thickness=2)
            report['cells'].append(dict(row=row.get('row'), field=field, bounds=bounds,
                                         reason=row.get('review_reason') or 'invalid_number'))
    if report['cells'] and render:
        output = cv2.addWeighted(paint, alpha, output, 1.0 - alpha, 0.0)
    elif not report['cells']:
        report['status'] = 'no_invalid_numeric_cells'
    return output, report
