"""Safety gates for the additional damaged-grid/name fallback."""
import copy

import numpy as np
import pytest

from App.ocr import damaged_receipt_geometry as grid
from App.ocr import damaged_receipt_text as damage
from App.ocr import selective_text_recovery as text


def candidate():
    return dict(polygon=np.array([[10,20],[910,20],[910,420],[10,420]],np.float32),
                status='needs_review',warnings=[])


def uncertain():
    return dict(status='uncertain',geometry=dict(four_columns=False),
                header=dict(state='unreadable_or_incomplete'))


def prepare(monkeypatch):
    monkeypatch.setattr(grid,'detect_damaged_grid',lambda im:(candidate(),dict(
        status='recovered',selected_columns=[10,100,550,700,910],ocr_calls=0)))
    monkeypatch.setattr(damage,'detect_damage_stripe',lambda *a:dict(forced_ocr=True))
    return dict(polygon=np.array([[550,30],[915,30],[915,420],[550,420]],np.float32))


@pytest.mark.parametrize('verification',[None,dict(status='confirmed'),dict(status='rejected'),
                                       dict(status='uncertain',geometry=dict(four_columns=True))])
def test_existing_table_untouched_without_specific_fragment_evidence(monkeypatch,verification):
    previous = candidate()
    def forbidden(*args):
        pytest.fail('Successful table must not invoke fallback')
    monkeypatch.setattr(grid,'detect_damaged_grid',forbidden)
    actual,report=grid.recover_damaged_boundary(None,previous,previous_verification=verification)
    assert actual is previous
    assert report['status']=='not_needed'


def test_full_grid_and_physical_tear_replace_right_fragment(monkeypatch):
    previous=prepare(monkeypatch)
    actual,report=grid.recover_damaged_boundary(None,previous,lambda *a:dict(status='confirmed'),uncertain())
    assert report['replaced_fragment'] is True
    assert actual['polygon'][0,0]==10
    assert previous['polygon'][0,0]==550
    assert report['ocr_calls']==0


@pytest.mark.parametrize('status',['uncertain','rejected'])
def test_unconfirmed_full_candidate_cannot_replace_fragment(monkeypatch,status):
    previous=prepare(monkeypatch)
    actual,report=grid.recover_damaged_boundary(None,previous,lambda *a:dict(status=status),uncertain())
    assert actual is previous
    assert report['status']=='not_recovered'


def test_no_fragment_replacement_without_verifier(monkeypatch):
    previous=prepare(monkeypatch)
    assert grid.recover_damaged_boundary(None,previous,previous_verification=uncertain())[0] is previous


def test_no_fragment_replacement_without_photo_damage(monkeypatch):
    previous=prepare(monkeypatch)
    monkeypatch.setattr(damage,'detect_damage_stripe',lambda *a:dict(forced_ocr=False))
    assert grid.recover_damaged_boundary(None,previous,lambda *a:dict(status='confirmed'),uncertain())[0] is previous


def test_unrelated_or_full_width_uncertain_table_preserved(monkeypatch):
    prepare(monkeypatch)
    previous=candidate()
    actual,report=grid.recover_damaged_boundary(None,previous,lambda *a:dict(status='confirmed'),uncertain())
    assert actual is previous
    assert report['status']=='not_recovered'


def test_blank_and_small_images_have_no_candidate():
    for image in [None,np.full((100,100,3),255,np.uint8),np.full((1280,958,3),255,np.uint8)]:
        boundary,report=grid.detect_damaged_grid(image)
        assert boundary['polygon'] is None
        assert report['ocr_calls']==0


def test_observed_rules_define_bands_and_leave_total_outside():
    yy,xx=np.indices((300,800),dtype=np.float32)
    layout=dict(name_left=80,name_right=480,right=780,body_top=16,body_bottom=280,
                row_boundaries=[16,280])
    rules=[dict(ends=[0,y,799,y],slope=0) for y in [20,80,180,240]]
    updated,audit=grid.observed_row_layout(layout,xx,yy,dict(horizontal_rules=rules))
    assert audit['status']=='applied'
    assert updated['row_boundaries']==[20,80,180,240]
    assert updated['body_bottom']==240
    assert updated['grid_confirmed'] is True
    assert layout['body_bottom']==280


@pytest.mark.parametrize('levels',[[20,80,180],[20,80,180,500],[20,80,81,240]])
def test_missing_outside_or_ambiguous_rules_preserve_layout(levels):
    yy,xx=np.indices((300,800),dtype=np.float32)
    layout=dict(name_left=80,right=780)
    rules=[dict(ends=[0,y,799,y],slope=0) for y in levels]
    updated,audit=grid.observed_row_layout(layout,xx,yy,dict(horizontal_rules=rules))
    assert updated is layout
    assert audit['status']=='not_applied'


def test_damage_cannot_force_tax_or_nonoverlapping_rows():
    mapping=np.tile(np.arange(200,dtype=np.float32)[:,None],(1,100))
    rows=[dict(row=1,kind='product',y_range=[10,30]),dict(row=2,kind='tax',y_range=[40,60]),
          dict(row=3,kind='product',y_range=[40,60])]
    selected=damage.affected_name_rows(rows,dict(name_left=10,name_right=80),mapping,
        dict(status='confirmed_bright_gap',stripes=[dict(y0=35,y1=70)]))
    assert selected==[3]


@pytest.mark.parametrize('confirmed',[False,True])
def test_forced_name_ocr_preserves_raw_numbers_and_manual_names(monkeypatch,confirmed):
    original=dict(row=1,kind='product',name_raw='Кабель медный обычный',quantity_raw='2',price_raw='3900',
                  quantity='2',price='3900',ocr_confidence=.99,y_range=[20,100],raw_lines=[],
                  name_confirmed=confirmed)
    before=copy.deepcopy(original)
    monkeypatch.setattr(text,'_rapid_read',lambda *a:dict(status='ok',text='Не подтверждено',confidence=.3))
    monkeypatch.setattr(text,'_tesseract_read',lambda *a:dict(status='binary_missing'))
    rows,report=text.recover_text_names([original],[],np.full((150,450,3),255,np.uint8),
        dict(name_left=30,name_right=350),settings=dict(enabled=True,max_rows=3),
        rapidocr=lambda _:[],force_rows=[1])
    assert original==before
    for key in before:
        assert rows[0][key]==before[key]
    assert len(report['stage2_attempts'])==(0 if confirmed else 1)
    assert report['stage2_applied']==[]
