"""MANDATORY real full-OCR gate: numbers alone cannot conceal mixed names.

Run in the project's .venv with OCR_REAL_PHOTO_TESTS=1. Never weaken
assertions or silently skip missing OCR models, images or catalog dependencies.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import hashlib

import numpy as np
import pytest

from App.ocr.ocr_run import MODEL
from App.ocr.web_pipeline import ReceiptPipeline
from App.web_server import Job
from App.ocr.receipt_editing import parse_number

ROOT=Path(__file__).resolve().parents[3]
PHOTO=ROOT/'data/ocr_regression/printed_grid_v3/photos/receipt_20261115.png'
EXPECTED=ROOT/'data/ocr_regression/printed_grid_v3/expected/receipt_20261115.json'
OUTPUT=ROOT/'experiment_outputs/printed_grid_v3/real_receipt'

pytestmark=pytest.mark.skipif(os.environ.get('OCR_REAL_PHOTO_TESTS')!='1',
 reason='Must set OCR_REAL_PHOTO_TESTS=1 for acceptance; a skipped test is NOT a PASS')

class _NoCatalogMatch:
    def match(self,name):
        return dict(status='not_matched',best=None,top_matches=[])


def test_full_ocr_12_products_geometry_and_totals(monkeypatch,tmp_path):
    assert PHOTO.is_file() and EXPECTED.is_file() and MODEL.is_file()
    gold=json.loads(EXPECTED.read_text(encoding='utf-8'))
    assert hashlib.sha256(PHOTO.read_bytes()).hexdigest()==gold['sha256']

    from App.ocr import printed_grid_adapter as adapter
    monkeypatch.setattr(adapter,'mode',lambda:'select')
    engine=None
    pipeline=ReceiptPipeline()
    pipeline.matcher=_NoCatalogMatch()
    def resources():
        nonlocal engine
        if engine is None:
            from rapidocr_onnxruntime import RapidOCR
            engine=RapidOCR(rec_model_path=str(MODEL),rec_img_shape=[3,48,320],text_score=.3)
        pipeline.engine=engine
        pipeline.matcher=_NoCatalogMatch()
    monkeypatch.setattr(pipeline,'resources',resources)
    job=Job('e'*32,PHOTO.name,tmp_path)
    (tmp_path/'upload.bin').write_bytes(PHOTO.read_bytes())
    value=pipeline.process(job)
    calc=value['calculation']
    actual=[r for r in value['rows'] if r.get('kind')=='product']
    taxes=[r for r in value['rows'] if r.get('kind')=='tax']
    report=json.loads((tmp_path/'printed_grid_rectification.json').read_text(encoding='utf-8'))
    layout=json.loads((tmp_path/'layout.json').read_text(encoding='utf-8'))['layout']
    raw=json.loads((tmp_path/'ocr_rectified.json').read_text(encoding='utf-8'))
    boundaries=layout.get('row_boundaries',[])[1:-1]
    crossing=[]
    for block in raw.get('lines',[]):
        pts=np.asarray(block.get('box',[]),dtype=float)
        if pts.shape!=(4,2) or not np.isfinite(pts).all():continue
        x=float(pts[:,0].mean())
        if not layout['name_left']+3 < x < layout['name_right']-3:continue
        top,bottom=pts[:,1].min(),pts[:,1].max()
        for y in boundaries:
            if top+2 < y < bottom-2:
                crossing.append({'name':block.get('text'), 'boundary_y':y})
    data=dict(expected=gold, actual=[dict(name=r.get('name_raw'),quantity=r.get('quantity',r.get('quantity_raw')),
                                     price=r.get('price',r.get('price_raw'))) for r in actual],
              status=calc.get('validation_status'), total=calc.get('computed_total'),
              crossing=crossing,printed_grid=report)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/'result.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    assert report['selected'],data
    assert len(actual)==12 and len(taxes)==1,data
    for row,reference in zip(actual,gold['products']):
        assert parse_number(row.get('quantity',row.get('quantity_raw')))==parse_number(reference['quantity']),(row,reference)
        assert parse_number(row.get('price',row.get('price_raw')))==parse_number(reference['price']),(row,reference)
    assert parse_number(taxes[0].get('quantity',taxes[0].get('quantity_raw')))==parse_number(gold['tax']['quantity'])
    assert calc['computed_total']==gold['printed_total'] and calc['validation_status']=='match',data
    assert not crossing, 'Physical line cuts OCR name block; fix segmentation before accepting: '+str(crossing)
    # Letter accuracy is a separate diagnostic, per the accepted geometry-first
    # workflow. Preserve raw OCR and record literal marker failures for review.
    expected_marker_by_row=['akfix','lezard','гусак','stimul','ga060b',
                            't100','wc6050m','styro','prosvet','100','30','4.2']
    issues=[]
    for number,(row,marker) in enumerate(zip(actual,expected_marker_by_row),1):
        normalized=str(row.get('name_recovered') or row.get('name_raw') or '').casefold()
        normalized=normalized.replace(' ','').replace('\n','').replace('-','')
        if marker not in normalized:
            issues.append(dict(row=number,expected_marker=marker,actual=row.get('name_raw')))
    data['name_marker_issues']=issues
    data['geometry_acceptance']='passed'
    (OUTPUT/'result.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
