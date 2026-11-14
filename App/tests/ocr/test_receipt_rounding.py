"""Whole-tenge reconciliation regressions for the photographed receipt format."""
from App.ocr.receipt_editing import update_receipt


CONTROL_LINES = [
    ("10", "230"), ("2", "680"), ("2", "680"), ("10", "470"),
    ("20", "530"), ("1", "7800"), ("9", "450"), ("6", "720"),
    ("15", "450"), ("3", "760"), ("20", "55"), ("5", "830"),
    ("3", "1270"), ("15", "835.71"), ("5", "260"), ("5", "250"),
]


def row(number, kind, quantity, price, name=""):
    return dict(row=number, kind=kind, name_raw=name or f"Строка {number}",
                quantity_raw=quantity, price_raw=price, name_confirmed=True)


def control_receipt():
    rows = [row(index, "product", quantity, price)
            for index, (quantity, price) in enumerate(CONTROL_LINES, 1)]
    rows.append(row(17, "tax", "11147", "1", "НДС 16 %"))
    return dict(total_raw="80 813,00 ₸", rows=rows)


def test_control_receipt_rounding_is_accepted_without_changing_source():
    value = control_receipt()
    source = [(item["quantity_raw"], item["price_raw"]) for item in value["rows"]]
    result = update_receipt(value)
    calculation = result["calculation"]

    assert calculation["status"] == "match"
    assert calculation["validation_status"] == "match"
    assert calculation["vat"]["status"] == "match"
    assert calculation["computed_total"] == "80813.00"
    assert calculation["exact_computed_total"] == "80812.65"
    assert calculation["expected_total"] == "80813.00"
    assert calculation["difference"] == "0.00"
    assert calculation["exact_difference"] == "-0.35"
    assert calculation["rounding_mode"] == "sum_then_integer"
    assert calculation["rounding_adjustment"] == "0.35"
    assert calculation["rounding"]["matched_modes"] == ["sum_then_integer", "each_line_integer"]
    assert calculation["rounding"]["fractional_rows"] == [dict(
        row=14, kind="product", line_amount="12535.65",
        whole_tenge_amount="12536.00", adjustment="0.35")]
    assert result["rows"][13]["line_total"] == "12535.65"
    assert [(item["quantity_raw"], item["price_raw"]) for item in result["rows"]] == source
    assert result["can_finalize"]


def test_exact_total_keeps_previous_behavior():
    value = dict(total_raw="116", rows=[row(1, "product", "1", "100"),
                                        row(2, "tax", "16", "1", "НДС")])
    calculation = update_receipt(value)["calculation"]

    assert calculation["status"] == "match"
    assert calculation["rounding_mode"] == "exact"
    assert calculation["rounding_adjustment"] == "0.00"
    assert calculation["computed_total"] == calculation["exact_computed_total"] == "116.00"
    assert calculation["rounding"]["matched_modes"] == ["exact"]


def test_each_line_rounding_is_disclosed_when_grand_rounding_differs():
    value = dict(total_raw="300", rows=[row(1, "product", "1", "100.40"),
                                        row(2, "product", "1", "200.40")])
    calculation = update_receipt(value)["calculation"]

    assert calculation["status"] == "match"
    assert calculation["rounding_mode"] == "each_line_integer"
    assert calculation["rounding_adjustment"] == "-0.80"
    assert calculation["rounding"]["candidates"] == {
        "sum_then_integer": "301.00", "each_line_integer": "300.00"}
    assert calculation["rounding"]["matched_modes"] == ["each_line_integer"]


def test_integer_only_one_tenge_error_is_not_masked():
    value = dict(total_raw="301", rows=[row(1, "product", "1", "100"),
                                        row(2, "product", "1", "200")])
    calculation = update_receipt(value)["calculation"]

    assert calculation["status"] == "mismatch"
    assert calculation["validation_status"] == "mismatch"
    assert calculation["rounding"]["reason"] == "no_fractional_tenge_lines"
    assert calculation["computed_total"] == "300.00"
    assert calculation["difference"] == "-1.00"
    assert not update_receipt(value)["can_finalize"]


def test_fractional_printed_total_is_never_rounded_to_whole_tenge():
    value = dict(total_raw="301.10", rows=[row(1, "product", "1", "100.65"),
                                           row(2, "product", "1", "200")])
    calculation = update_receipt(value)["calculation"]

    assert calculation["status"] == "mismatch"
    assert calculation["rounding"]["reason"] == "printed_total_has_tiyn"
    assert calculation["computed_total"] == "300.65"


def test_unverified_rounding_identity_is_rejected():
    value = dict(total_raw="302", rows=[row(1, "product", "1", "100.40"),
                                        row(2, "product", "1", "200.40")])
    calculation = update_receipt(value)["calculation"]

    assert calculation["status"] == "mismatch"
    assert calculation["rounding"]["reason"] == "no_exact_whole_tenge_rounding_identity"
    assert calculation["rounding"]["candidates"] == {
        "sum_then_integer": "301.00", "each_line_integer": "300.00"}


def test_missing_numeric_field_blocks_rounding():
    value = dict(total_raw="301", rows=[row(1, "product", "1", "100.65"),
                                        row(2, "product", "", "200")])
    calculation = update_receipt(value)["calculation"]

    assert calculation["status"] == "incomplete"
    assert calculation["rounding_mode"] == "not_checked"
    assert calculation["rounding"]["accepted_total"] is None


def test_wrong_vat_is_not_masked_by_total_rounding():
    value = dict(total_raw="316", rows=[row(1, "product", "1", "100.65"),
                                        row(2, "product", "1", "200"),
                                        row(3, "tax", "15", "1", "НДС")])
    calculation = update_receipt(value)["calculation"]

    assert calculation["status"] == "match"
    assert calculation["rounding_mode"] == "sum_then_integer"
    assert calculation["vat"]["status"] == "mismatch"
    assert calculation["validation_status"] == "vat_mismatch"
    assert not update_receipt(value)["can_finalize"]
