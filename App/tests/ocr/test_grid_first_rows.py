"""Production safety gates; real-photo acceptance lives in experiment_outputs."""
import copy

import numpy as np
import pytest

from App.ocr import grid_first_rows as grid


LAYOUT=dict(body_left=0,name_left=20,name_right=180,quantity_right=240,right=300,body_top=0,body_bottom=500)


def evidence(text,x,y,width=20,height=16):
    box=[[x-width/2,y-height/2],[x+width/2,y-height/2],[x+width/2,y+height/2],[x-width/2,y+height/2]]
    return dict(text=text,confidence=.95,box=box,box_original=copy.deepcopy(box),eval_box=np.array(box))


def setup_bands(monkeypatch,support=4):
    monkeypatch.setattr(grid,'find_price_anchors',lambda *a:([50,150,250,350,450],dict(status='ok')))
    monkeypatch.setattr(grid,'price_bands',lambda *a:([0,100,200,300,400,500],dict(horizontal_rule_supported=support)))


def test_already_separated_rows_are_untouched(monkeypatch):
    setup_bands(monkeypatch)
    before=[dict(row=i,kind='product',name_raw=str(i)) for i in range(5)]
    rows,report=grid.reconstruct_rows(np.zeros((500,300,3),np.uint8),LAYOUT,[],before)
    assert rows==before
    assert report['status']=='not_needed'
    assert report['name_calls']==report['numeric_calls']==0


def test_refinement_needs_observed_printed_separators(monkeypatch):
    setup_bands(monkeypatch,support=2)
    before=[dict(row=i,kind='product',name_raw=str(i)) for i in range(5)]
    rows,report=grid.reconstruct_rows(np.zeros((500,300,3),np.uint8),LAYOUT,[],before,prefer_printed_bands=True)
    assert rows==before
    assert report['status']=='not_needed'


def test_crossing_block_not_copied_to_multiple_rows(monkeypatch):
    setup_bands(monkeypatch)
    before=[dict(row=1,kind='product',name_raw='MERGED',quantity_raw='1',price_raw='90')]
    snapshot=copy.deepcopy(before)
    block=evidence('MERGED',100,100,height=100)
    rows,report=grid.reconstruct_rows(np.zeros((500,300,3),np.uint8),LAYOUT,[block],before)
    assert len(rows)==5
    assert all(r['name_raw']=='' for r in rows)
    assert len(report['ambiguous_blocks'])==2
    assert before==snapshot


def test_cell_ocr_limits_and_preservation_of_numeric_raw(monkeypatch):
    setup_bands(monkeypatch)
    def read(_):return [('4',.99)]
    rows,report=grid.reconstruct_rows(np.full((500,300,3),250,np.uint8),LAYOUT,[],[dict(kind='product')],
                                     read_numeric=read,max_numeric_crops=2)
    assert report['numeric_calls']==2
    assert rows[0]['quantity_raw']==''
    assert rows[0]['quantity']=='4'
    assert 'price' not in rows[0]
    assert all('quantity' not in r for r in rows[1:])


def test_numeric_disagreement_does_not_change_quantity(monkeypatch):
    setup_bands(monkeypatch)
    values=iter([('4',.99),('7',.99)])
    rows,report=grid.reconstruct_rows(np.full((500,300,3),250,np.uint8),LAYOUT,[],[dict(kind='product')],
                                     read_numeric=lambda _:[next(values)],max_numeric_crops=2)
    assert all('quantity' not in r for r in rows)


@pytest.mark.parametrize('method',['wave_grid','grid_curvature_v2','rotation_crop'])
def test_other_validated_geometry_is_not_changed(method):
    im=np.full((100,100,3),255,np.uint8)
    yy,xx=np.indices((100,100),dtype=np.float32)
    meta=dict(method=method)
    result,report=grid.preserve_source_columns(im,[],im,xx,yy,meta)
    assert result[0] is im and result[3] is meta
    assert report['status']=='not_needed'


def test_complete_contour_grid_is_not_changed(monkeypatch):
    from App.ocr import table_verification
    monkeypatch.setattr(table_verification,'verify_table_candidate',lambda *a:dict(geometry=dict(four_columns=True)))
    im=np.full((100,100,3),255,np.uint8)
    yy,xx=np.indices((100,100),dtype=np.float32)
    meta=dict(method='contour_coons_patch')
    result,report=grid.preserve_source_columns(im,[],im,xx,yy,meta)
    assert result[0] is im and result[3] is meta


@pytest.mark.parametrize('value',['1 10 4','4 2','NaN','-1'])
def test_ambiguous_numeric_strings_rejected(value):
    assert grid._number(value) is None
