"""Replay a REAL diagnostic OCR trace, with the source receipt mathematics."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import cv2

from App.ocr.numeric_consensus_math import accept_unique_single_cell_correction
from App.ocr.numeric_overlay import overlay_invalid_numeric_cells
from App.ocr.receipt_editing import update_receipt

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'data/ocr_regression/row3_069_690'


def fixture():
    source = DATA / 'source'
    receipt = json.loads((source / 'receipt_before.json').read_text(encoding='utf-8'))
    numeric_report = json.loads((source / 'numeric_recovery.json').read_text(encoding='utf-8'))
    layout = json.loads((source / 'layout.json').read_text(encoding='utf-8'))['layout']
    image = cv2.imread(str(source / 'rectified.png'))
    assert image is not None
    return receipt, numeric_report, layout, image


def test_real_photo_ocr_evidence_fixes_069_to_690_without_modifying_raw():
    receipt, audit, layout, image = fixture()
    original = copy.deepcopy(receipt)
    fixed, changes, report = accept_unique_single_cell_correction(receipt, audit['all_numeric_attempts'])
    assert report['status'] == 'unique_verified', report
    assert len(changes) == 1 and changes[0]['row'] == 3 and changes[0]['field'] == 'price'
    row3 = next(row for row in fixed['rows'] if row['row'] == 3)
    assert row3['price_raw'].strip() == '069'  # OCR evidence untouched
    assert row3['price'] == '690'
    assert fixed.get('expected_total') == original.get('expected_total')
    assert receipt == original  # no in-place mutation
    calc = update_receipt(fixed)['calculation']
    assert calc['products_subtotal'] == '134160.00'
    assert calc['computed_total'] == '155626.00'
    assert calc['expected_total'] == '155626.00'
    assert calc['vat']['actual_tax'] == '21466.00'
    assert calc['validation_status'] == 'match'


def test_real_photo_overlay_shows_suspect_while_math_fails_but_not_after_recovery():
    receipt, audit, layout, image = fixture()
    unchanged, report = overlay_invalid_numeric_cells(image, receipt, layout, render=False, numeric_report=audit)
    assert unchanged is image  # non-destructive browser SVG overlay
    assert report['status'] == 'ok', report
    assert {(x['row'], x['field']) for x in report['cells']} == {(3, 'price')}, report
    assert report['cells'][0]['reason'] == 'ocr_disagreement'

    fixed, _, _ = accept_unique_single_cell_correction(receipt, audit['all_numeric_attempts'])
    fixed = update_receipt(fixed)
    image_after, empty = overlay_invalid_numeric_cells(image, fixed, layout, render=False, numeric_report=audit)
    assert image_after is image
    assert empty['status'] == 'mathematics_passed'
    assert empty['cells'] == []


def test_never_fix_solely_to_make_wrong_printed_total_match():
    receipt, audit, _, _ = fixture()
    receipt['expected_total'] = '155627'  # neither OCR alternative reconciles
    fixed, changes, report = accept_unique_single_cell_correction(receipt, audit['all_numeric_attempts'])
    assert fixed is None and changes == []
    assert report['status'] == 'not_confirmed'


def test_no_change_when_math_already_passes_even_if_ocr_disagrees():
    receipt, audit, _, _ = fixture()
    next(row for row in receipt['rows'] if row['row'] == 3)['price'] = '690'
    receipt = update_receipt(receipt)
    fixed, changes, report = accept_unique_single_cell_correction(receipt, audit['all_numeric_attempts'])
    assert fixed is None and changes == []
    assert report['status'] == 'already_matched'


def test_low_support_is_not_enough_to_overwrite_existing_numeric_value():
    receipt, audit, _, _ = fixture()
    proposals = copy.deepcopy(audit['all_numeric_attempts'])
    p = next(p for p in proposals if p['row'] == 3 and p['field'] == 'price')
    p['views'][0]['confidence'] = 0.1
    fixed, changes, report = accept_unique_single_cell_correction(receipt, proposals)
    assert fixed is None and changes == []


def test_manual_number_must_never_be_overwritten_even_with_math_support():
    receipt, audit, _, _ = fixture()
    target = next(row for row in receipt['rows'] if row['row'] == 3)
    target['price_source'] = 'user'
    original = copy.deepcopy(receipt)
    fixed, changes, report = accept_unique_single_cell_correction(receipt, audit['all_numeric_attempts'])
    assert fixed is None and not changes
    assert receipt == original


def test_conflicting_valid_single_cell_solutions_are_rejected():
    from decimal import Decimal
    from App.ocr.numeric_consensus_math import accept_unique_single_cell_correction

    receipt = dict(rows=[
        dict(row=1, kind='product', quantity_raw='2', price_raw='10'),
        dict(row=2, kind='product', quantity_raw='2', price_raw='10'),
    ], total_raw='60', expected_total='60')
    # Either price correction (+20) would balance, so it must not choose one.
    candidates = [dict(row=n, kind='product', field='price', candidate='20', agreed=True,
                       views=[{'number':'20', 'confidence':.9} for _ in range(4)])
                  for n in (1, 2)]

    def arithmetic(current):
        summed = sum(Decimal(str(row.get('quantity', row['quantity_raw']))) *
                     Decimal(str(row.get('price', row['price_raw']))) for row in current['rows'])
        match = summed == Decimal('60')
        return dict(calculation=dict(status='match' if match else 'mismatch',
                                     validation_status='match' if match else 'mismatch',
                                     expected_total='60', computed_total=str(summed), vat={'status':'not_applicable'}))

    fixed, changes, report = accept_unique_single_cell_correction(receipt, candidates, validate_fn=arithmetic)
    assert fixed is None and changes == []
    assert report['solutions'] == 2 and report['status'] == 'ambiguous'


def test_manual_edit_audit_log_blocks_recovery_without_source_flag():
    receipt, audit, _, _ = fixture()
    receipt['audit_log'] = [{'changes': [{'row': 3, 'field': 'price', 'after': '069'}]}]
    fixed, changes, _ = accept_unique_single_cell_correction(receipt, audit['all_numeric_attempts'])
    assert fixed is None and changes == []


def test_rounded_total_is_not_enough_to_accept_consensus_change():
    receipt, audit, _, _ = fixture()
    def rounded_only(current):
        checked = update_receipt(current)
        checked['calculation'].update(status='match', validation_status='match',
            computed_total='155626.00', exact_computed_total='155625.60')
        # Keep the input receipt failing so candidate validation is exercised.
        if next(r for r in current['rows'] if r['row'] == 3)['price'] != '690':
            checked['calculation']['status'] = 'mismatch'
        return checked
    fixed, changes, _ = accept_unique_single_cell_correction(receipt, audit['all_numeric_attempts'], validate_fn=rounded_only)
    assert fixed is None and not changes
