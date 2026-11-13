"""Early, scale-independent evidence that a region is a receipt product table.

Geometry and header text are independent supporting signals. Unreadable labels
are not a rejection: a complete four-column grid can stand on its own.
"""
from __future__ import annotations

import re
import unicodedata

import cv2
import numpy as np
from rapidfuzz.fuzz import ratio

VERSION = 'receipt-table-verification-1.0'


def _groups(values, tolerance):
    groups = []
    for value in sorted(values):
        if not groups or value - np.mean(groups[-1]) > tolerance:
            groups.append([value])
        else:
            groups[-1].append(value)
    return [float(np.mean(group)) for group in groups]


def header_role(text):
    text = unicodedata.normalize('NFKC', str(text)).casefold().replace('ғ', 'г')
    key = re.sub(r'[^а-яәіңүұқөһa-z]', '', text)
    if any(part in key for part in ('атау', 'наименован', 'назван')):
        return 'name'
    for role, words in (('quantity', ('саны', 'колво', 'количество')),
                        ('price', ('багасы', 'багас', 'цена', 'стоимость'))):
        if key and max(ratio(key, word) for word in words) >= 80:
            return role
    return None


def _header_evidence(image, recognize):
    if recognize is None:
        return dict(state='not_requested', roles=[])
    try:
        readings = recognize(image)
        width = image.shape[1]
        labels = []
        for box, text, confidence in readings or []:
            box = np.asarray(box, dtype=float)
            if box.shape != (4, 2) or not np.isfinite(box).all() or float(confidence) < .3:
                continue
            role = header_role(text)
            cx, cy = box.mean(axis=0)
            fraction = cx / width
            position_ok = ((role == 'name' and .08 <= fraction <= .72) or
                           (role == 'quantity' and .43 <= fraction <= .90) or
                           (role == 'price' and .65 <= fraction <= 1.0))
            if position_ok:
                labels.append(dict(role=role, text=str(text), y=float(cy),
                                   height=max(1., float(np.ptp(box[:, 1])))))
        for anchor in labels:
            aligned = [item for item in labels if abs(item['y'] - anchor['y']) <=
                       max(8., .6 * max(item['height'], anchor['height']))]
            if len({item['role'] for item in aligned}) >= 2:
                return dict(state='confirmed', roles=sorted({item['role'] for item in aligned}),
                            labels=aligned)
        return dict(state='unreadable_or_incomplete', roles=sorted({item['role'] for item in labels}))
    except Exception as error:
        # A header check must not break an otherwise working OCR pipeline.
        return dict(state='unavailable', roles=[], error=str(error))


def verify_table_candidate(image, boundary, recognize_header=None):
    report = dict(version=VERSION, status='uncertain', reason='', geometry={},
                  header=dict(state='not_requested', roles=[]))
    polygon = boundary.get('polygon')
    if polygon is None:
        report.update(status='rejected', reason='no_boundary')
        return report
    points = np.asarray(polygon, dtype=float).reshape(-1, 2)
    if len(points) < 4 or not np.isfinite(points).all():
        report.update(status='rejected', reason='invalid_boundary')
        return report
    h, w = image.shape[:2]
    lo, hi = points.min(axis=0), points.max(axis=0)
    left, top = max(0, int(lo[0])-2), max(0, int(lo[1])-2)
    right, bottom = min(w, int(np.ceil(hi[0]))+3), min(h, int(np.ceil(hi[1]))+3)
    region = image[top:bottom, left:right]
    if min(region.shape[:2]) < 12:
        report.update(status='rejected', reason='region_too_small')
        return report
    # Relative header dimensions, not fixed camera pixels. Broad ranges allow
    # different fonts, wrapping, perspective and paper curvature.
    header_height = min(region.shape[0], max(20, round(region.shape[1] * .28)))
    header = region[:header_height]
    factor = min(3., max(1., 700. / region.shape[1]))
    sample = cv2.resize(header, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY) if sample.ndim == 3 else sample
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 3)
    sh, sw = binary.shape
    lines = cv2.HoughLinesP(binary, 1, np.pi/360, threshold=max(15, round(sw*.055)),
                            minLineLength=max(15, round(sw*.10)), maxLineGap=max(3, round(sw*.015)))
    vertical, horizontal = [], []
    for x0, y0, x1, y1 in ([] if lines is None else lines.reshape(-1, 4)):
        dx, dy = abs(x1-x0), abs(y1-y0)
        if dy >= sw*.10 and dx <= dy*.18:
            vertical.append((x0+x1)/2/sw)
        if dx >= sw*.45 and dy <= dx*.15:
            horizontal.append((y0+y1)/2/sw)
    columns = _groups(vertical, .025)
    rules = _groups(horizontal, .018)
    four_columns = (any(x <= .08 for x in columns) and any(x >= .92 for x in columns) and
                    any(.04 <= a <= .22 and .42 <= b <= .78 and b+.07 <= c <= .92
                        for a in columns for b in columns for c in columns if a+.20 < b))
    header_row = any(.045 <= b-a <= .23 for a in rules[:1] for b in rules[1:])
    # Repeated thick bars cannot be product columns, even if OCR is unreadable.
    persistent = np.mean(binary > 0, axis=0) >= .65
    dense_bars = bool(np.mean(persistent) >= .32 and
                      np.count_nonzero(np.diff(np.r_[False, persistent, False].astype(np.int8)) == 1) >= 4)
    report['geometry'] = dict(four_columns=bool(four_columns), header_row=bool(header_row),
                              column_fractions=columns, horizontal_rule_fractions=rules,
                              dense_bars=dense_bars, region=[left, top, right, bottom])
    if dense_bars:
        report.update(status='rejected', reason='barcode_not_product_grid')
        return report
    report['header'] = _header_evidence(sample, recognize_header)
    support = (boundary.get('selected') or {}).get('support') or [0, 0]
    grid_support = len(support) == 2 and support[0] >= 3 and 3 <= support[1] <= 6
    if report['header']['state'] == 'confirmed' and grid_support:
        report.update(status='confirmed', reason='header_words_and_grid')
    elif four_columns and header_row and grid_support:
        report.update(status='confirmed', reason='four_columns_and_header_geometry')
    else:
        report['reason'] = 'insufficient_evidence_preserve_existing_result'
    return report
