"""Offline tests, real App receipt mathematics, synthetic crops and fake recognizer.
Not an end-to-end test of the ONNX model.
"""
import copy

import cv2
import numpy as np
import pytest

from App.ocr.receipt_editing import update_receipt
from App.ocr.selective_price_ocr import recover_missing_prices, _price_views, _consensus


LAYOUT = dict(name_right=320, quantity_right=470, right=650)


def sample(price1='', price2=''):
    return dict(rows=[
        dict(row=1, kind='service', name_raw='Доставка Рынок', quantity_raw='15000',
             price_raw=price1, y_range=(45, 104)),
        dict(row=2, kind='tax', name_raw='НДС 16 %', quantity_raw='2400',
             price_raw=price2, y_range=(105, 165)),
    ], total_raw='17 400,00')


def image():
    img=np.full((220, 700, 3), 238, np.uint8)
    cv2.putText(img, '1 T', (545,86), cv2.FONT_HERSHEY_SIMPLEX,.95,(32,32,32),2, cv2.LINE_AA)
    cv2.putText(img, '1 T', (545,145), cv2.FONT_HERSHEY_SIMPLEX,.95,(32,32,32),2, cv2.LINE_AA)
    return img


def responder(values):
    calls=[]
    iterator=iter(values)
    def recognize(images):
        assert len(images)==1
        assert images[0].ndim==3 and images[0].shape[2]==3
        calls.append(images[0].shape)
        return [next(iterator)]
    return recognize,calls


def test_only_missing_prices_recognized_and_verified_against_total_and_vat():
    receipt=sample()
    orig=copy.deepcopy(receipt)
    recognize,calls=responder([('1 ₸',.27),('1',.29),('1T',.33),('1',.34)])
    result,audit=recover_missing_prices(receipt,image(),LAYOUT,recognize)
    assert len(calls)==4
    assert audit['attempted_cells']==2
    assert audit['ocr_calls']==4
    assert audit['applied_count']==2
    assert [r['price'] for r in result['rows']]==['1','1']
    assert result['rows'][0]['price_raw']=='' and result['rows'][1]['price_raw']==''
    assert receipt==orig
    assert update_receipt(result)['calculation']['status']=='match'
    assert update_receipt(result)['calculation']['vat']['status']=='match'


def test_no_second_pass_for_correct_prices():
    recognize,calls=responder([])
    result,audit=recover_missing_prices(sample('1','1'),image(),LAYOUT,recognize)
    assert not calls and audit['attempted_cells']==0 and audit['applied_count']==0
    assert result['rows'][0]['price_raw']=='1'


def test_no_guess_from_total_if_ocr_proposes_wrong_price():
    recognize,calls=responder([('2',.40),('2',.42),('2',.43),('2',.44)])
    result,audit=recover_missing_prices(sample(),image(),LAYOUT,recognize)
    assert audit['applied_count']==0
    assert 'price' not in result['rows'][0]
    assert audit['candidate_total_status']=='mismatch'


def test_unreadable_first_price_triggers_only_two_extra_views():
    recognize,calls=responder([('?',.85),('I',.78),('?',.80),('l',.65),('1',.3),('1',.4)])
    result,audit=recover_missing_prices(sample(),image(),LAYOUT,recognize)
    assert audit['ocr_calls']==6
    assert audit['applied_count']==0  # Unconfirmed overall total; do not auto-fill.
    assert result['rows'][0]['price_raw']==''
    assert audit['attempts'][0]['candidate'] is None
    assert audit['attempts'][1]['candidate']=='1'


def test_no_printed_total_means_no_automatic_acceptance():
    receipt=sample()
    receipt['total_raw']=''
    recognize,calls=responder([('1',.32),('1',.41),('1',.32),('1',.41)])
    result,audit=recover_missing_prices(receipt,image(),LAYOUT,recognize)
    assert audit['applied_count']==0
    assert audit['candidate_total_status']=='missing_original_total'


def test_budget_limits_cost_without_change():
    recognize,calls=responder([('1',.32),('1',.41)])
    result,audit=recover_missing_prices(sample(),image(),LAYOUT,recognize,max_images=2)
    assert audit['ocr_calls']==2 and len(calls)==2
    assert audit['applied_count']==0
    assert result['rows'][1]['price_raw']==''
    assert any(rec['reason']=='invalid_crop_or_budget' for rec in audit['attempts'])


def test_views_preserve_three_channels_and_avoid_stretching():
    views=_price_views(image(),(478,50,642,99))
    assert len(views)==4
    assert all(view.ndim==3 and view.shape[2]==3 for view in views)
    assert all(view.shape[0]>=49 for view in views)


def test_conflicts_not_resolved_by_arithmetic():
    assert _consensus([dict(number='1',confidence=.9),dict(number='2',confidence=.9)])[0] is None
    assert _consensus([dict(number='1',confidence=.25),dict(number='1',confidence=.28)])[0]=='1'


def test_no_unnecessary_ocr_for_missing_quantity_alone():
    receipt=sample('1','1')
    receipt['rows'][0]['quantity_raw']=''
    recognize,calls=responder([])
    result,audit=recover_missing_prices(receipt,image(),LAYOUT,recognize)
    assert audit['attempted_cells']==0
    assert calls==[]
    assert result['rows'][0]['quantity_raw']==''


def test_bad_crop_is_reported_without_ocr():
    receipt=sample()
    receipt['rows'][0]['y_range']=(1,5)
    recognize,calls=responder([('1',.4),('1',.4)])
    result,audit=recover_missing_prices(receipt,image(),LAYOUT,recognize)
    assert audit['attempted_cells']==1
    assert any('геометрия цены' in w for w in audit['warnings'])
    assert result['rows'][0]['price_raw']==''


def test_integrated_numeric_recovery_no_full_rescan_when_two_prices_recovered():
    from App.ocr.numeric_recovery import recover_numeric_receipt
    recognize,calls=responder([('1',.29),('1',.29),('1',.29),('1',.29)])
    result,audit=recover_numeric_receipt(sample(),image(),LAYOUT,[],recognize,
                                          settings=dict(vat_rate_percent='16',
                                                        vat_rate_source='default_16_percent',max_images=250))
    assert [row['price'] for row in result['rows']]==['1','1']
    assert len(calls)==4
    assert audit['selective_price']['applied_count']==2
    assert audit['total_triggered'] is False
    assert audit['final_total_status']=='match'
    assert audit['final_vat_status']=='match'


def test_integrated_numeric_recovery_retains_existing_fallback_when_prices_unconfirmed(monkeypatch):
    from App.ocr.numeric_recovery import recover_numeric_receipt
    import App.ocr.numeric_recovery as recovery
    import App.ocr.numeric_fallback_ensemble as ensemble
    # Isolate the legacy fallback; the optional ensemble has its own tests.
    monkeypatch.setattr(ensemble, 'load_ensemble_options', lambda: {'enabled': False})
    fallback=[]
    def no_fallback_proposals(image,tasks,*args):
        fallback.extend(tasks)
        return []
    monkeypatch.setattr(recovery,'_recognition_pairs',no_fallback_proposals)
    # Both missing price cells are read, but OCR says '2', contradicted by total.
    recognize,calls=responder([('2',.44),('2',.45),('2',.44),('2',.45)])
    result,audit=recover_numeric_receipt(sample(),image(),LAYOUT,[],recognize,
                                          settings=dict(vat_rate_percent='16',
                                                        vat_rate_source='default_16_percent',max_images=250))
    assert len(calls)==4
    assert audit['total_triggered'] is True
    assert fallback  # Existing master recovery is retained, unlike the installer.
    assert audit['selective_price']['applied_count']==0
    assert audit['final_total_status']=='incomplete'
    assert all('price' not in row for row in result['rows'])
