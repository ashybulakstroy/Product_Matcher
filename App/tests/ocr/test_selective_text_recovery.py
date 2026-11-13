import copy

import numpy as np
import pytest

from App.ocr import selective_text_recovery as text

LAYOUT = dict(name_left=30, name_right=350, quantity_right=395, right=450)
TAIL = 'DIN-изоляторе типа Стойка ШНИ-6х9-12-С-С IEK'
GOOD = 'Шина N ноль на ' + TAIL
BAD = 'н О N ит ' + TAIL
SETTINGS = dict(enabled=True, max_rows=3, min_confidence=.6, languages='rus+eng', tesseract_exe=None)


def row(name=BAD, confidence=.6, number=12):
    return dict(row=number, kind='product', name_raw=name, quantity_raw='2', price_raw='850',
                quantity='2', price='850', ocr_confidence=confidence, y_range=[30, 145], raw_lines=[])


def line(value, y, x=45, width=240):
    return dict(text=value, confidence=.95, box=[[x,y],[x+width,y],[x+width,y+12],[x,y+12]])


def image():
    return np.full((200,460,3), 245, np.uint8)


def reading(value):
    return [(np.array([[4,4],[250,4],[250,25],[4,25]]),value,.95)]


def test_missing_words_from_existing_blocks_need_no_ocr():
    original = row(name=TAIL, confidence=.99)
    before = copy.deepcopy(original)
    blocks = [line('Шина N ноль на',44), line('DIN-изоляторе типа',65),
              line('Стойка ШНИ-6х9-12-С-С IEK',88), line('СОСЕДНИЙ ТОВАР',156)]
    def forbidden(_):
        raise AssertionError('Existing blocks are sufficient')
    rows, audit = text.recover_text_names([original],blocks,image(),LAYOUT,rapidocr=forbidden,settings=SETTINGS)
    assert rows[0]['name_recovered'] == GOOD
    assert rows[0]['name_raw'] == TAIL
    assert rows[0]['name_recovery_source'] == 'original_ocr_blocks'
    assert audit['rapidocr_calls'] == audit['tesseract_calls'] == 0
    assert original == before
    for key in ('quantity','price','quantity_raw','price_raw','raw_lines','y_range'):
        assert rows[0][key] == before[key]


def test_independent_engines_can_confirm_damaged_name(monkeypatch):
    monkeypatch.setattr(text,'_tesseract_read',lambda *a,**k:dict(status='ok',text=GOOD,confidence=.9))
    rows,audit = text.recover_text_names([row()],[],image(),LAYOUT,rapidocr=lambda _:reading(GOOD),settings=SETTINGS)
    assert rows[0]['name_recovered'] == GOOD
    assert audit['rapidocr_calls'] == audit['tesseract_calls'] == 2
    assert rows[0]['name_raw'] == BAD


@pytest.mark.parametrize('replacement', ['Шина M ноль на '+TAIL, GOOD.replace('6х9-12','6х9-13')])
def test_conflicting_models_cannot_be_accepted(monkeypatch,replacement):
    monkeypatch.setattr(text,'_tesseract_read',lambda *a,**k:dict(status='ok',text=replacement,confidence=.99))
    rows,audit = text.recover_text_names([row()],[],image(),LAYOUT,rapidocr=lambda _:reading(GOOD),settings=SETTINGS)
    assert 'name_recovered' not in rows[0]
    assert rows[0]['name_recovery_status'] == 'needs_review'


def test_two_engines_cannot_override_existing_single_letter_model(monkeypatch):
    original='Шина N стойка'
    replacement='Шина M ноль на стойка'
    monkeypatch.setattr(text,'_tesseract_read',lambda *a,**k:dict(status='ok',text=replacement,confidence=.99))
    rows,_=text.recover_text_names([row(original,.2)],[],image(),LAYOUT,rapidocr=lambda _:reading(replacement),settings=SETTINGS)
    assert 'name_recovered' not in rows[0]
    assert rows[0]['name_raw'] == original


def test_short_numeric_codes_are_protected_in_both_stages():
    assert not text._preserves_words('Стойка 12','Шина Стойка 13')
    assert not text._text_better('Стойка 12','Шина Стойка 13')
    assert text._preserves_words('Стойка 12','Шина N Стойка 12')


def test_clean_or_confirmed_names_do_not_invoke_ocr():
    confirmed=row();confirmed['name_confirmed']=True
    def forbidden(_):
        raise AssertionError('Do not rescan clean or confirmed names')
    rows,audit=text.recover_text_names([row(GOOD,.99),confirmed],[],image(),LAYOUT,rapidocr=forbidden,settings=SETTINGS)
    assert audit['rapidocr_calls'] == audit['tesseract_calls'] == 0
    assert not any('name_recovered' in r for r in rows)


def test_units_and_prepositions_are_not_damaged_single_letters():
    original = 'Гофра ПВХ ЧЕРНАЯ ф20 (100 м в бухте)'
    assert text._damaged(original, .99) == []
    assert 'fragmented_single_letters' in text._damaged(BAD, .9)
    rows,audit=text.recover_text_names([row(original,.99)],[],image(),LAYOUT,settings=SETTINGS)
    assert audit['stage2_attempts'] == []
    assert audit['rapidocr_calls'] == audit['tesseract_calls'] == 0


def test_missing_tesseract_or_failed_rapidocr_keeps_original(monkeypatch):
    monkeypatch.setattr(text,'_tesseract_read',lambda *a,**k:dict(status='binary_missing'))
    def failed(_):
        raise RuntimeError('OCR failed')
    rows,audit=text.recover_text_names([row()],[],image(),LAYOUT,rapidocr=failed,settings=SETTINGS)
    assert rows[0]['name_raw'] == BAD
    assert 'name_recovered' not in rows[0]
    assert audit['stage2_attempts'][0]['rapidocr'][0]['status'] == 'error'


def test_retry_limit_is_enforced(monkeypatch):
    monkeypatch.setattr(text,'_tesseract_read',lambda *a,**k:dict(status='binary_missing'))
    rows,audit=text.recover_text_names([row(number=i) for i in range(1,8)],[],image(),LAYOUT,
                                     rapidocr=lambda _:[],settings=SETTINGS)
    assert len(audit['stage2_attempts']) == 3
    assert audit['rapidocr_calls'] == 6
    assert len([v for v in audit['skipped'] if v['reason']=='retry_limit']) == 4


def test_disabled_recovery_is_exact_passthrough():
    original=[row()]
    rows,audit=text.recover_text_names(original,[],image(),LAYOUT,settings=dict(SETTINGS,enabled=False))
    assert rows == original
    assert audit['status']=='disabled'
