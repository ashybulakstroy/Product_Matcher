import json
from pathlib import Path

import cv2

from App.ocr.numeric_overlay import overlay_invalid_numeric_cells
from App.ocr.numeric_recovery import _accept_non_total_before_total
from App.ocr.receipt_editing import update_receipt
from App.ocr.receipt_editing import parse_number
from decimal import Decimal

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'data' / 'ocr_regression' / 'row5_numeric_overlay'


def test_real_row5_replay_accepts_quantity_three_and_keeps_original_total():
    receipt = json.loads((DATA / 'source' / 'receipt_before.json').read_text(encoding='utf-8'))
    replay = json.loads((DATA / 'source' / 'numeric_recovery_replay.json').read_text(encoding='utf-8'))
    trial, updates = _accept_non_total_before_total(receipt, replay['proposals'])
    assert trial is not None
    assert any(item['field'] == 'quantity' and item['after'] == '3' for item in updates)
    assert trial['expected_total'] == receipt['expected_total']
    assert parse_number(trial['expected_total']) == Decimal('69600.00')
    checked = update_receipt(trial)
    row5 = next(row for row in checked['rows'] if row['row'] == 5)
    assert row5['quantity'] == '3'
    assert checked['calculation']['computed_total'] == '69600.00'
    assert checked['calculation']['validation_status'] == 'match'


def test_real_row5_overlay_marks_quantity_cell():
    image = cv2.imread(str(DATA / 'source' / 'rectified.png'))
    assert image is not None
    receipt = json.loads((DATA / 'source' / 'receipt_before.json').read_text(encoding='utf-8'))
    layout = json.loads((DATA / 'source' / 'layout.json').read_text(encoding='utf-8'))['layout']
    overlaid, report = overlay_invalid_numeric_cells(image, receipt, layout)
    assert report['cells']
    row5 = next(item for item in report['cells'] if item['row'] == 5 and item['field'] == 'quantity')
    x0, y0, x1, y1 = row5['bounds']
    px = overlaid[(y0 + y1) // 2, (x0 + x1) // 2]
    assert int(px[2]) > int(px[0])
