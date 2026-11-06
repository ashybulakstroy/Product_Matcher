import json

import pytest

from App.ocr.receipt_editing import apply_edit, parse_number, save_receipt, update_receipt


def receipt(score=98, quantity="1", articles=None):
    best = dict(name="Пила CROWN CT15074", normalized="пила crown ct15074", score=score,
                sources=[dict(article=a) for a in (articles or ["123"])])
    return update_receipt(dict(total_raw="36 656,00 Т", pass_similarity=87, rows=[
        dict(row=1, kind="product", name_raw="Пила @ CT15074", quantity_raw=quantity,
             price_raw="31600Т", match=dict(best=best), needs_review=True, review_reason="Близкие варианты"),
        dict(row=2, kind="tax", name_raw="НДС 16 %", quantity_raw="5056", price_raw="1Т")]))


def edit(value, **payload):
    return apply_edit(value, dict(revision=value["revision"], **payload), lambda key: None)


@pytest.mark.parametrize("text,expected", [("1,5", "1.5"), ("36 656,00 Т", "36656.00"),
    ("1.234,56", "1234.56"), ("1,234.56", "1234.56"), ("1O", None), ("-1", None), (True, None)])
def test_number_parsing_does_not_guess_ocr(text, expected):
    assert str(parse_number(text)) == expected if expected else parse_number(text) is None


def test_threshold_inclusive_replaces_name_despite_close_candidates():
    value = receipt(score=87)
    assert value["rows"][0]["name"] == "Пила CROWN CT15074"
    assert value["rows"][0]["name_raw"] == "Пила @ CT15074"
    assert not value["rows"][0]["needs_review"]
    assert value["calculation"]["status"] == "match"
    assert value["calculation"]["computed_total"] == "36656.00"
    assert value["calculation"]["tax_subtotal"] == "5056.00"
    assert edit(value, finalize=True)["finalized"]


def test_low_match_requires_confirmation_and_preserves_original():
    value = receipt(score=86.99)
    assert value["review_count"] == 1 and value["rows"][0]["name_origin"] == "ocr"
    with pytest.raises(ValueError):
        edit(value, finalize=True)
    updated = edit(value, row=1, name="Правильная пила", confirm_name=True)
    assert updated["review_count"] == 0 and updated["can_finalize"]
    assert updated["rows"][0]["name_raw"] == value["rows"][0]["name_raw"]
    assert value["rows"][0]["name"] != updated["rows"][0]["name"]
    assert updated["revision"] == 1 and updated["audit_log"]
    with pytest.raises(RuntimeError):
        apply_edit(updated, dict(revision=0, row=1, quantity="2"), lambda key:None)


def test_missing_quantity_and_mismatch_can_be_corrected():
    value = receipt(quantity="")
    assert value["calculation"]["status"] == "incomplete"
    confirmed = edit(value, row=1, confirm_name=True)
    assert confirmed["review_count"] == 1
    corrected = edit(confirmed, row=1, quantity="1")
    assert corrected["calculation"]["status"] == "match"
    changed = edit(corrected, row=1, price="30000")
    assert changed["calculation"]["status"] == "mismatch"
    assert changed["calculation"]["difference"] == "-1600.00"
    assert not changed["can_finalize"]
    with pytest.raises(ValueError):
        edit(changed, finalize=True)
    fixed = edit(changed, row=1, price="31600")
    assert fixed["can_finalize"] and fixed["rows"][0]["price_raw"] == "31600Т"


def test_multiple_articles_require_explicit_selection():
    value = receipt(articles=["123", "456"])
    entry = value["rows"][0]["match"]["best"]
    assert value["review_count"] == 1
    with pytest.raises(ValueError, match="артикул"):
        apply_edit(value, dict(revision=0, row=1, catalog_key=entry["normalized"], confirm_name=True), lambda key:entry)
    updated = apply_edit(value, dict(revision=0, row=1, catalog_key=entry["normalized"], article="456", confirm_name=True), lambda key:entry)
    assert updated["rows"][0]["article"] == "456" and updated["can_finalize"]


def test_saved_exports_contain_current_and_raw_values(tmp_path):
    value = edit(receipt(score=70), row=1, name="Верное имя", confirm_name=True)
    save_receipt(tmp_path, value)
    assert json.loads((tmp_path/"receipt.json").read_text(encoding="utf-8")) == value
    csv = (tmp_path/"receipt.csv").read_text(encoding="utf-8-sig")
    assert "Верное имя" in csv and "Пила @ CT15074" in csv


def test_services_and_decimal_line_rounding():
    value = dict(rows=[dict(row=1,kind="service",name_raw="Доставка",quantity_raw="2,5",price_raw="10,01")], total_raw="25,03")
    assert update_receipt(value)["calculation"]["status"] == "match"
    assert value["calculation"]["services_subtotal"] == "25.03"


def test_printed_total_row_is_used_as_ocr_evidence_without_double_counting(tmp_path):
    value = dict(total_raw="",rows=[
        dict(row=2,kind="product",name_raw="Дверь 51103",quantity_raw="3",price_raw="112700 T"),
        dict(row=3,kind="product",name_raw="Дверь 50951",quantity_raw="1",price_raw="216000 T"),
        dict(row=4,kind="tax",name_raw="НДС",quantity_raw="88656",price_raw="1 T"),
        dict(row=5,kind="total",name_raw="Корытынды :",quantity_raw="",price_raw="642 756,00 T")])
    update_receipt(value)
    assert value["calculation"]["status"] == "match"
    assert value["calculation"]["computed_total"] == "642756.00"
    assert value["calculation"]["expected_total"] == "642756.00"
    assert value["expected_total_source"] == "ocr_total_row"
    assert value["expected_total_ocr_row"] == 5
    assert value["total_raw"] == ""
    assert value["rows"][-1]["price_raw"] == "642 756,00 T"
    assert len(value["rows"]) == 4  # OCR evidence retained, presentation filters it.
    save_receipt(tmp_path,value)
    assert "Корытынды" not in (tmp_path/"receipt.csv").read_text(encoding="utf-8-sig")


def test_ocr_total_fallback_never_overrides_a_user_value_or_invents_a_total():
    value = dict(total_raw="",expected_total="123",expected_total_source="user",rows=[
        dict(row=1,kind="total",name_raw="Итого",quantity_raw="",price_raw="456")])
    assert update_receipt(value)["expected_total"] == "123"
    missing = dict(total_raw="",rows=[dict(row=1,kind="service",name_raw="Доставка",quantity_raw="1",price_raw="123")])
    assert update_receipt(missing)["calculation"]["status"] == "missing_original_total"
