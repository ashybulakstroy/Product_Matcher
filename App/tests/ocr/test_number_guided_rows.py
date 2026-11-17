from copy import deepcopy
import numpy as np
import pytest
from App.ocr import number_guided_rows as ng


@pytest.mark.parametrize('text',['1 2','17824 0','1.0','I','0','1000'])
def test_separate_digit_groups_are_not_invented_item_numbers(text):
    assert ng.parse_index(text) is None


def test_default_mode_is_shadow(tmp_path,monkeypatch):
    monkeypatch.delenv('OCR_NUMBER_GUIDED_ROWS_MODE',raising=False)
    assert ng.configured_mode(tmp_path)=='shadow'


def test_off_keeps_numeric_and_name_evidence():
    rows=[dict(row=1,kind='product',name_raw='Товар',quantity='500',price='2228',y_range=[20,70])]
    layout=dict(row_boundaries=[0,20,70]);before=deepcopy((rows,layout))
    returned,report=ng.refine_number_guided_rows(np.zeros((100,100,3),np.uint8),layout,[],rows,mode='off')
    assert returned is rows and (rows,layout)==before
    assert report['status']=='disabled'


def test_price_and_name_column_numbers_cannot_anchor_rows():
    layout=dict(body_left=0,name_left=35,body_top=10,body_bottom=140)
    def line(text,x,y):return dict(text=text,confidence=.99,eval_box=[[x-3,y-5],[x+3,y-5],[x+3,y+5],[x-3,y+5]])
    lines=[line('1',15,35),line('2',15,75),line('500',100,35),line('17824',230,75),line('0',230,90)]
    assert [a['number'] for a in ng.number_anchors(lines,layout)]==[1,2]


def test_number_sequence_alone_cannot_create_physical_rows():
    layout=dict(body_left=0,name_left=35,name_right=150,quantity_right=190,right=240,body_top=10,body_bottom=140)
    lines=[dict(text=str(n),confidence=.99,center=[15,y]) for n,y in [(1,35),(2,75),(3,110)]]
    rows=[dict(row=1,kind='product',name_raw='Товар',quantity_raw='1',price_raw='10',y_range=[10,140])]
    result,report=ng.refine_number_guided_rows(np.full((160,260,3),255,np.uint8),layout,lines,rows,mode='select')
    assert result is rows and not report['selected']
    assert report['reason']=='no_independent_printed_grid'
