import copy

import cv2
import numpy as np

from App.ocr.quantity_cells import recover_quantities
from App.ocr.receipt_editing import update_receipt


def fixture(total="100",quantity=""):
    image=np.full((250,500,3),240,np.uint8)
    cv2.putText(image,"1",(340,155),0,.8,(20,20,20),2)
    box=[[395,130],[475,130],[475,160],[395,160]]
    row=dict(row=2,kind="service",name_raw="Доставка",quantity_raw=quantity,price_raw="100",
             y_range=[80,210],raw_lines=[dict(text="100",confidence=.9,box=box,box_original=box)])
    receipt=update_receipt(dict(rows=[row],total_raw=total))
    layout=dict(name_right=300,quantity_right=380,right=490)
    return receipt,image,layout


def test_recovery_reads_cell_and_keeps_raw_ocr_immutable():
    receipt,image,layout=fixture()
    original=copy.deepcopy(receipt)
    result,record,_=recover_quantities(receipt,image,layout,lambda images:[("1",.48),("1",.49)])
    assert record["applied"]
    assert result["rows"][0]["quantity"]=="1"
    assert result["rows"][0]["quantity_raw"]==""
    assert result["rows"][0]["raw_lines"]==original["rows"][0]["raw_lines"]
    assert receipt==original
    assert result["calculation"]["validation_status"]=="match"


def test_disagreement_or_low_confidence_never_fills_quantity():
    for readings in ([("1",.9),("7",.9)],[("1",.1),("1",.2)],[("I",.9),("1",.9)]):
        receipt,image,layout=fixture()
        result,record,_=recover_quantities(receipt,image,layout,lambda images:readings)
        assert not record["applied"]
        assert result==receipt


def test_printed_total_is_a_gate_not_a_source_of_invented_numbers():
    receipt,image,layout=fixture(total="200")
    result,record,_=recover_quantities(receipt,image,layout,lambda images:[("1",.8),("1",.8)])
    assert not record["applied"]
    assert result["rows"][0]["quantity"]==""
    assert result["rows"][0]["quantity"]!="2"


def test_valid_existing_quantity_is_never_replaced():
    receipt,image,layout=fixture(quantity="3",total="300")
    def forbidden(images):
        raise AssertionError("No OCR call for an existing quantity")
    result,record,_=recover_quantities(receipt,image,layout,forbidden)
    assert result==receipt and not record["attempts"]


def test_empty_cell_cannot_generate_a_quantity():
    receipt,image,layout=fixture()
    image[:]=240
    def forbidden(images):
        raise AssertionError("No OCR call for a blank cell")
    result,record,_=recover_quantities(receipt,image,layout,forbidden)
    assert result==receipt and not record["attempts"]
