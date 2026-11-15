"""Synthetic and geometric real-photo verification of V3 printed-grid warp."""
from pathlib import Path
import hashlib
import json

import cv2
import numpy as np

from App.ocr.printed_grid_rectification import (
    rectify_by_printed_grid, horizontal_rules, project_cell_boundary,
    select_five_dividers, vertical_candidates,
)

ROOT = Path(__file__).resolve().parents[3]
PHOTO = ROOT / 'data/ocr_regression/printed_grid_v3/photos/receipt_20261115.png'
EXPECTED = ROOT / 'data/ocr_regression/printed_grid_v3/expected/receipt_20261115.json'


def test_blank_and_random_grid_do_not_force_rectification():
    blank = np.full((1350, 800, 3), 255, np.uint8)
    result, report = rectify_by_printed_grid(blank)
    assert result is None and report['status'] == 'rejected'
    few = [dict(x=x, slope=0, top=50, bottom=500, support=1000.) for x in (100, 200, 400, 600)]
    assert select_five_dividers(few, blank.shape)[0] is None


def test_synthetic_five_columns_map_without_crossings():
    img = np.full((1100, 700, 3), 255, np.uint8)
    # Four-column Torgsoft sheet, slanted physical rule lines with shifted x.
    xs = [120, 158, 400, 475, 580]
    slopes = [-.08, -.07, -.02, .005, .025]
    for x, slope in zip(xs, slopes):
        cv2.line(img, (round(x+slope*(220-550)), 220),
                 (round(x+slope*(980-550)), 980), (70,70,70), 2)
    for y in range(220, 981, 65):
        cv2.line(img, (round(xs[0]+slopes[0]*(y-550)), y),
                 (round(xs[-1]+slopes[-1]*(y-550)), y+10), (65,65,65), 2)
    result, report = rectify_by_printed_grid(img)
    assert report['status'] == 'candidate_ready', report
    output, mx, my, meta = result
    assert output.shape[:2] == mx.shape == my.shape
    assert meta['method'] == 'printed_grid_rectified'
    assert len(meta['printed_grid_columns']) == 5
    assert all(a < b for a,b in zip(meta['printed_grid_columns'], meta['printed_grid_columns'][1:]))
    assert np.isfinite(mx).all() and np.isfinite(my).all()


def test_real_photo_rectifies_five_columns_and_horizontal_rules():
    assert PHOTO.is_file(), 'Real user photo is mandatory'
    assert EXPECTED.is_file(), 'Human-transcribed evidence is mandatory'
    expected = json.loads(EXPECTED.read_text(encoding='utf-8'))
    assert hashlib.sha256(PHOTO.read_bytes()).hexdigest() == expected['sha256']
    original = cv2.imdecode(np.frombuffer(PHOTO.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    assert original is not None
    before,_ = vertical_candidates(original)
    result, audit = rectify_by_printed_grid(original)
    assert audit['status'] == 'candidate_ready', audit
    output,mx,my,meta = result
    assert len(meta['printed_grid_columns']) == 5
    assert len(horizontal_rules(output)) >= 5
    assert max(abs(row[1]) for row in horizontal_rules(output)) < .03
    assert max(abs(v) for v in audit['source_vertical_slopes']) > .07
    # A true rectangular correction is NOT the original, merely cropped.
    assert output.shape[1] < original.shape[1]
    assert np.std(mx[30, 30:output.shape[1]-30] - mx[output.shape[0]//2,30:output.shape[1]-30]) > 3
    points=[[70,80],[250,80],[250,200],[70,200]]
    curved=project_cell_boundary(mx,my,points,samples=25)
    assert len(curved) == 4 and all(len(e)==25 for e in curved)
    assert all(len(p)==2 for edge in curved for p in edge)


def test_printed_amounts_in_reference_are_arithmetically_consistent():
    from decimal import Decimal
    expected=json.loads(EXPECTED.read_text(encoding='utf-8'))
    subtotal=sum(Decimal(p['quantity'])*Decimal(p['price']) for p in expected['products'])
    vat=Decimal(expected['tax']['quantity'])*Decimal(expected['tax']['price'])
    assert len(expected['products']) == 12
    assert subtotal==Decimal(expected['products_subtotal'])==Decimal('79160')
    assert vat==Decimal('12666')
    assert subtotal+vat==Decimal(expected['printed_total'])==Decimal('91826.00')

def test_boundary_refinement_keeps_text_centres_and_raw_evidence(monkeypatch):
    from App.ocr import printed_grid_rectification as module
    from copy import deepcopy
    monkeypatch.setattr(module,'layout_rules',lambda _: [(y,0.,400.) for y in [110.,170.,230.,280.,330.,400.]])
    image=np.zeros((500,500,3),np.uint8)
    lines=[{'box':[[70.,81.],[260.,81.],[260.,107.],[70.,107.]],'text':'wrapped last line'},
           {'box':[[70.,107.],[260.,107.],[260.,128.],[70.,128.]],'text':'next product'}]
    before=deepcopy(lines)
    bounds=[40.,104.75,170.,230.,280.,330.,400.,450.]
    output=module.refine_printed_row_boundaries(image,bounds,{'name_left':60.,'name_right':300.},lines)
    assert len(output)==len(bounds)
    assert 105. <= output[1] <= 109.
    for line in lines:
        center=np.mean(np.asarray(line['box'])[:,1])
        assert np.searchsorted(bounds,center)==np.searchsorted(output,center)
    assert lines==before and bounds[1]==104.75
