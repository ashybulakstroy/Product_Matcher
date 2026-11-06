from decimal import Decimal
from App.ocr.receipt_editing import apply_edit,update_receipt


def document(base="100",vat="16",total="116",label="НДС"):
    return dict(total_raw=total,rows=[
        dict(row=1,kind="service",name_raw="Доставка",quantity_raw="1",price_raw=base),
        dict(row=2,kind="tax",name_raw=label,quantity_raw=vat,price_raw="1")])


def test_vat_and_grand_total_both_pass():
    result = update_receipt(document(base="554100",vat="88656",total="642756"))
    assert result["calculation"]["validation_status"]=="match"
    assert result["calculation"]["vat"]["status"]=="match"
    assert result["calculation"]["vat"]["expected_tax"]=="88656.00"
    assert result["calculation"]["vat"]["rate_percent"]=="16"
    assert result["rows"][1]["vat_status"]=="match"


def test_equal_sum_cannot_hide_wrong_vat():
    result = update_receipt(document(vat="15",total="115"))
    assert result["calculation"]["status"]=="match"
    assert result["calculation"]["validation_status"]=="vat_mismatch"
    assert result["calculation"]["vat"]["difference"]=="-1.00"
    assert result["rows"][1]["needs_review"]
    assert not result["can_finalize"]


def test_missing_quantity_prevents_vat_and_total_verification():
    value = document()
    value["rows"][0]["quantity_raw"]=""
    result = update_receipt(value)
    assert result["calculation"]["validation_status"]=="incomplete"
    assert result["calculation"]["vat"]["status"]=="incomplete"
    assert result["calculation"]["vat"]["expected_tax"] is None


def test_printed_rate_does_not_override_configured_default():
    result = update_receipt(document(vat="12",total="112",label="НДС 12 %"))
    assert result["calculation"]["validation_status"]=="vat_mismatch"
    assert result["calculation"]["vat"]["rate_percent"]=="16"
    assert result["calculation"]["vat"]["rate_source"] in {"default_16_percent","env_file"}


def test_no_vat_row_is_not_a_failed_check():
    value=document(total="100")
    value["rows"].pop()
    result=update_receipt(value)
    assert result["calculation"]["vat"]["status"]=="not_applicable"
    assert result["calculation"]["validation_status"]=="match"


def test_correcting_tax_clears_error_and_restores_finalization():
    value=update_receipt(document(vat="15",total="116"))
    result=apply_edit(value,dict(revision=0,row=2,quantity="16"),lambda key:None)
    assert result["calculation"]["validation_status"]=="match"
    assert result["can_finalize"]
    assert not result["rows"][1]["needs_review"]


def test_mixed_printed_rates_are_ignored_for_calculation():
    value=document(label="НДС 16 %")
    value["rows"].append(dict(row=3,kind="tax",name_raw="НДС 12 %",quantity_raw="1",price_raw="1"))
    result=update_receipt(value)
    assert result["calculation"]["vat"]["status"]=="mismatch"
    assert not result["can_finalize"]


def test_vat_respects_integer_precision_printed_on_receipt():
    result=update_receipt(document(base="37270",vat="5963",total="43233",label="HDC 16 %"))
    assert result["calculation"]["validation_status"]=="match"
    assert result["calculation"]["vat"]["rounding_quantum"]=="1"
    assert Decimal(result["calculation"]["vat"]["unrounded_tax"])==Decimal("5963.2")


def test_explicit_cents_are_checked_with_cent_precision():
    result=update_receipt(document(base="37270",vat="5963.00",total="43233.00",label="НДС 16 %"))
    assert result["calculation"]["validation_status"]=="vat_mismatch"
    assert result["calculation"]["vat"]["rounding_quantum"]=="0.01"
