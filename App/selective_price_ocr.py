"""Targeted price OCR: read only missing prices on an already rectified receipt.

This production module depends on App, NEVER on experiments. It does not
initialize a second OCR engine, edit .env, or invent a price from arithmetic.
"""
from __future__ import annotations

import copy
from time import perf_counter

import cv2
import numpy as np

from App.receipt_editing import parse_number, update_receipt

VERSION = 'selective-price-ocr-1.0'
PAYABLE = {'product', 'service', 'tax'}


def _price_views(image, bounds):
    """Four BGR, aspect-preserving views; only first two run initially.

    Full cell is preserved (including currency glyph). Later preparations can
    separate digits better without guessing where the currency symbol starts.
    """
    x0, y0, x1, y1 = bounds
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return []
    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()
    if gray.shape[0] < 5 or gray.shape[1] < 5:
        return []
    # Do not distort the shape of a narrow '1'. OCR receives 3-channel BGR.
    scale = min(3.0, max(1.0, 56.0 / gray.shape[0]))
    size = (max(8, round(gray.shape[1] * scale)), max(8, round(gray.shape[0] * scale)))
    raw = cv2.resize(gray, size, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=1.7, tileGridSize=(4, 4)).apply(raw)
    _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    block = min(31, max(3, (min(raw.shape) // 2) * 2 - 1))
    if block < 3:
        adaptive = otsu.copy()
    else:
        adaptive = cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, block, 7)
    sharp = cv2.addWeighted(raw, 1.7, cv2.GaussianBlur(raw, (3, 3), 0), -.7, 0)
    sources = [raw, otsu, adaptive, sharp]
    results = []
    for source in sources:
        padded = cv2.copyMakeBorder(source, 10, 10, 12, 12, cv2.BORDER_CONSTANT, value=255)
        results.append(cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR))
    return results


def _parse_ocr(text):
    # Reuse the project's number parser, including trailing T/Т/₸ currency.
    return parse_number(text)


def _consensus(readings):
    """Two consistent OCR readings; lower confidence only for one digit.

    Exact total/VAT validation is a second mandatory guard, NOT a source of
    the value. Two views use the same trained model and are correlated.
    """
    votes = {}
    for reading in readings:
        if reading['number'] is not None:
            votes.setdefault(reading['number'], []).append(reading)
    if not votes:
        return None, 'no_numeric_reading'
    ranked = sorted(votes.items(), key=lambda p: (len(p[1]),
                    sum(r['confidence'] for r in p[1])), reverse=True)
    text, supporters = ranked[0]
    if len(supporters) < 2:
        return None, 'less_than_two_votes'
    if len(ranked) > 1 and len(supporters) <= len(ranked[1][1]):
        return None, 'conflicting_votes'
    threshold = .18 if len(text) == 1 and text.isdigit() else .55
    if sum(r['confidence'] >= threshold for r in supporters) < 2:
        return None, 'low_confidence'
    return text, 'two_consistent_ocr_reads'


def recover_missing_prices(receipt, image, layout, recognize, *, ocr_lock=None, max_images=250):
    """Return a new receipt and audit. Never change *_raw or valid prices.

    Batch is deliberately avoided: App's rec_model expects BGR per independent
    cell. Runs at most 2+2 recognition calls per *missing price*, and no OCR
    for other numbers or fields.
    """
    from App.numeric_recovery import _crop_cell  # runtime import avoids a cycle

    started = perf_counter()
    current = copy.deepcopy(receipt)
    audit = dict(version=VERSION, attempted_cells=0, ocr_calls=0,
                 attempts=[], applied=[], applied_count=0, warnings=[], elapsed_ms=0)
    try:
        max_images = max(0, int(max_images))
        tasks = []
        for row in current.get('rows', []):
            if row.get('kind') not in PAYABLE:
                continue
            # Existing, valid price MUST NOT be scanned or overwritten.
            value = row.get('price', row.get('price_raw', ''))
            if parse_number(value) is not None:
                continue
            bounds = _crop_cell(image, layout, row, 'price')
            if bounds is None:
                audit['warnings'].append(f"Строка {row.get('row')}: геометрия цены недоступна")
                continue
            tasks.append((row, bounds))

        # Build image only for problematic cells; never scan nonprice fields.
        proposals = []
        for row, bounds in tasks:
            record = dict(row=row['row'], kind=row['kind'], field='price',
                          bounds=list(bounds), views=[], candidate=None,
                          reason='', accepted=False)
            images = _price_views(image, bounds)
            if len(images) != 4 or audit['ocr_calls'] + 2 > max_images:
                record['reason'] = 'invalid_crop_or_budget'
                audit['attempts'].append(record)
                continue
            audit['attempted_cells'] += 1
            def recognize_views(start, count):
                if ocr_lock is None:
                    return [recognize([view]) for view in images[start:start+count]]
                with ocr_lock:
                    return [recognize([view]) for view in images[start:start+count]]

            for start in (0, 2):
                if audit['ocr_calls'] + 2 > max_images:
                    record['reason'] = 'ocr_image_budget_reached'
                    break
                response = recognize_views(start, 2)
                for view_idx, raw in enumerate(response, start=start):
                    if len(raw) != 1:
                        raise ValueError('OCR must return one reading per image')
                    text, confidence = raw[0]
                    number = _parse_ocr(text)
                    record['views'].append(dict(view=view_idx+1, text=str(text), confidence=float(confidence),
                                                number=str(number) if number is not None else None))
                audit['ocr_calls'] += 2
                candidate, reason = _consensus(record['views'])
                record['candidate'], record['reason'] = candidate, reason
                if candidate is not None:
                    proposals.append(record)
                    break
            audit['attempts'].append(record)

        if proposals:
            trial = copy.deepcopy(current)
            for proposal in proposals:
                row = next((r for r in trial['rows'] if r['row'] == proposal['row']), None)
                if row is None or parse_number(row.get('price', row.get('price_raw', ''))) is not None:
                    continue
                row['price'] = proposal['candidate']
                row['price_source'] = 'selective_price_ocr'
                row['price_ocr'] = dict(views=proposal['views'], bounds=proposal['bounds'])
            # Validate on a detached copy: do NOT mutate untouched rows.
            verification = update_receipt(copy.deepcopy(trial))['calculation']
            audit['candidate_total_status'] = verification['status']
            audit['candidate_vat_status'] = verification['vat']['status']
            if verification['status'] == 'match' and verification['vat']['status'] in {'match', 'not_applicable'}:
                current = trial
                for proposal in proposals:
                    proposal['accepted'] = True
                    audit['applied'].append(dict(row=proposal['row'], field='price',
                                                 after=proposal['candidate'],
                                                 reason='ocr_consensus_and_full_math'))
            else:
                audit['warnings'].append('Подтверждённого совпадения НДС и печатного итога нет; цены не изменены')
        audit['applied_count'] = len(audit['applied'])
    finally:
        audit['elapsed_ms'] = round((perf_counter() - started) * 1000)
    return current, audit
