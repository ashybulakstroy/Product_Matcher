import numpy as np
import cv2

from App.ocr.numeric_recovery import _apply_verified, _numeric_views, _recognition_pairs
from App.ocr.numeric_settings import load_numeric_settings
from App.ocr.receipt_editing import update_receipt


def test_numeric_views_are_four_three_channel_preparations():
    image=np.full((80,240,3),245,np.uint8)
    cv2.putText(image,'1',(205,57),cv2.FONT_HERSHEY_SIMPLEX,1.2,(20,20,20),2,cv2.LINE_AA)
    views=_numeric_views(image,(0,0,240,80))
    assert len(views)==4
    assert all(view.ndim==3 and view.shape[2]==3 for view in views)


def test_recognition_is_independent_and_single_digit_uses_consensus_rule():
    image=np.full((80,240,3),245,np.uint8)
    cv2.putText(image,'1',(205,57),cv2.FONT_HERSHEY_SIMPLEX,1.2,(20,20,20),2,cv2.LINE_AA)
    calls=[]
    values=iter([('1',.42),('1',.39),('1',.21),('I',.60)])
    def recognize(images):
        calls.append(len(images))
        return [next(values)]
    result=_recognition_pairs(image,[dict(row=1,kind='product',field='quantity',bounds=[0,0,240,80])],
                              recognize,None,250)
    assert calls==[1,1,1,1]
    assert result[0]['agreed']
    assert result[0]['candidate']=='1'
    assert result[0]['confidence_rule']=='single_digit_consensus'


def test_valid_product_numbers_are_immutable_but_missing_and_tax_can_change():
    receipt=dict(rows=[
        dict(row=1,kind='product',quantity_raw='',price_raw='3000'),
        dict(row=2,kind='product',quantity_raw='2',price_raw='3000'),
        dict(row=3,kind='tax',quantity_raw='1307',price_raw='1')],total_raw='19071')
    proposals=[
        dict(row=1,kind='product',field='quantity',agreed=True,candidate='2',views=[],bounds=[]),
        dict(row=2,kind='product',field='price',agreed=True,candidate='300000',views=[],bounds=[]),
        dict(row=3,kind='tax',field='quantity',agreed=True,candidate='13071',views=[],bounds=[])]
    changed=_apply_verified(receipt,proposals)
    assert changed['rows'][0]['quantity']=='2'
    assert 'price' not in changed['rows'][1]
    assert changed['rows'][1]['price_raw']=='3000'
    assert changed['rows'][2]['quantity']=='13071'


def test_printed_vat_label_does_not_override_configured_rate(tmp_path):
    assert load_numeric_settings(tmp_path)['vat_rate_percent']=='16'
    receipt=dict(rows=[
        dict(row=1,kind='product',name_raw='Товар',quantity_raw='1',price_raw='100'),
        dict(row=2,kind='tax',name_raw='НДС 12 %',quantity_raw='16',price_raw='1')],total_raw='116')
    result=update_receipt(receipt)
    assert result['calculation']['vat']['rate_percent']=='16'
    assert result['calculation']['vat']['status']=='match'
