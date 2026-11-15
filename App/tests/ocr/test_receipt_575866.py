"""575866: a wrapped quantity heading is not a goods row."""
import copy
import os
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from App.ocr.ocr_catalog_eval import aligned_lines, choose_row_boundaries, parse_rows
from App.ocr.grid_first_rows import _wrapped_numeric, reconstruct_rows
from App.ocr.table_headers import without_table_headers

LAYOUT=dict(body_left=16,name_left=51,name_right=321,quantity_right=364,right=452,
            body_top=16,body_bottom=260,row_boundaries=[17.,85.5,133.5,182.5,209.,258.5])

def block(text,x0,y0,x1,y1):
    box=[[x0,y0],[x1,y0],[x1,y1],[x0,y1]]
    return dict(text=text,box=box,box_original=copy.deepcopy(box),confidence=.98)

def header_and_goods():
    return aligned_lines(dict(lines=[block('Кon',323,28,362,49),
        block('Наименование Товара',83,42,290,60),block('Цена',383,39,434,61),
        block('-BO',326,54,360,72),block('Диск',55,89,267,109),
        block('1',350,101,361,118),block('4000',374,101,447,117)]),{})

def test_two_line_header_keeps_printed_separator_and_first_quantity():
    lines=header_and_goods()
    content=without_table_headers(lines,LAYOUT)
    assert [l['text'] for l in content]==['Диск','1','4000']
    boundaries,_=choose_row_boundaries(lines,LAYOUT)
    assert boundaries==LAYOUT['row_boundaries']
    rows=parse_rows(lines,LAYOUT,boundaries)
    assert len(rows)==1
    assert rows[0]['quantity_raw']=='1'
    assert rows[0]['y_range']==[85.5,133.5]

def test_header_fragment_without_confirming_columns_is_not_discarded():
    lines=aligned_lines(dict(lines=[block('-BO',326,54,360,72)]),{})
    assert without_table_headers(lines,LAYOUT)==lines

@pytest.mark.parametrize('second_x', [351,325])
def test_wrapped_numeric_needs_right_edge_alignment(second_x):
    lines=aligned_lines(dict(lines=[block('633',329,211,364,230),
                                   block('6',second_x,233,second_x+13,253)]),{})
    before=[l['text'] for l in lines]
    assert _wrapped_numeric(lines,321,364)==('6336' if second_x==351 else None)
    assert [l['text'] for l in lines]==before

def test_equal_count_rows_are_refined_from_physical_separators(monkeypatch):
    from App.ocr import grid_first_rows as grid
    layout=copy.deepcopy(LAYOUT)
    lines=header_and_goods()
    before=[dict(kind='product',y_range=[90,140],quantity_raw='1',price_raw='4000')]
    # Same count but the provisional separator cuts through the physical cell.
    monkeypatch.setattr(grid,'find_printed_grid_bands',lambda *a,**kw:
        ([85.5,133.5],dict(status='ok',header_band=[16.5,85.5])))
    rows,report=reconstruct_rows(np.full((280,470,3),255,np.uint8),layout,lines,before)
    assert report['strategy']=='printed_grid'
    assert len(rows)==1 and rows[0]['y_range']==[85.5,133.5]
    assert layout['row_boundaries']==[16.5,85.5,133.5]

@pytest.mark.skipif(os.environ.get('OCR_REAL_PHOTO_TESTS')!='1',reason='Explicit real-photo OCR acceptance')
def test_original_photo_575866_has_three_goods_and_vat(tmp_path,monkeypatch):
    from rapidocr_onnxruntime import RapidOCR
    from App.ocr.ocr_run import MODEL
    from App.ocr.web_pipeline import ReceiptPipeline
    from App.ocr.receipt_editing import parse_number
    from App.web_server import Job
    photo=Path(__file__).parent/'fixtures/receipt_575866.png'
    assert photo.is_file() and MODEL.is_file()
    pipeline=ReceiptPipeline()
    class Matcher:
        def match(self,name):return dict(status='not_matched',best=None,top_matches=[])
    def resources():
        if pipeline.engine is None:
            pipeline.engine=RapidOCR(rec_model_path=str(MODEL),rec_img_shape=[3,48,320],text_score=.3)
        pipeline.matcher=Matcher()
    monkeypatch.setattr(pipeline,'resources',resources)
    (tmp_path/'upload.bin').write_bytes(photo.read_bytes())
    receipt=pipeline.process(Job('f'*32,photo.name,tmp_path))
    assert [r['kind'] for r in receipt['rows']]==['product','product','product','tax']
    values=[(parse_number(r['quantity']),parse_number(r['price'])) for r in receipt['rows']]
    assert values==[(Decimal('1'),Decimal('4000')),(Decimal('1'),Decimal('31600')),
                    (Decimal('10'),Decimal('400')),(Decimal('6336'),Decimal('1'))]
    assert receipt['calculation']['validation_status']=='match'
    assert receipt['calculation']['computed_total']=='45936.00'
    import json
    layout=json.loads((tmp_path/'layout.json').read_text(encoding='utf-8'))['layout']
    assert 64. not in layout['row_boundaries']
    assert np.allclose(layout['row_boundaries'],[16.5,85.5,133.5,182.5,209.,258.5],atol=2)
