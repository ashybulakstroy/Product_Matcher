"""Invoice draft generation must flag unresolved OCR cells, not crash."""
from App.invoicing.invoice_generation import build_generation


def source(quantity, price, total, vat="0"):
    return {
        "items": [dict(row_id="1", name="Товар", ocr_name="Товар",
                       quantity=quantity, price=price, line_kind="product")],
        "vat_amount": vat,
        "total": total,
        "validation_status": "match" if total is not None else "incomplete",
    }


def test_invalid_numeric_cells_do_not_crash_draft_generation():
    value = source("Бағасы", "", total=None)
    result = build_generation(value, suggest=False, vat_in_line_amounts=False,
                              preserve_receipt_quantities=True)

    warnings = result["warnings"]
    assert any("нечисловое поле quantity" in item for item in warnings)
    assert any("нечисловое поле price" in item for item in warnings)
    assert any("Проверьте количество и цену" in item for item in warnings)
    assert result["items"][0]["quantity"] == "0"
    assert result["items"][0]["price"] == "0"
    assert result["items"][0]["amount"] == "0.00"
    # The caller's OCR evidence is never rewritten.
    assert value["items"][0]["quantity"] == "Бағасы"
    assert value["items"][0]["price"] == ""


def test_verifiable_numbers_are_not_changed_by_the_guard():
    value = source("1", "100", total="100")
    result = build_generation(value, suggest=False, vat_in_line_amounts=False,
                              preserve_receipt_quantities=True)

    assert not any("нечисловое поле" in item for item in result["warnings"])
    assert result["items"][0]["quantity"] == "1"
    assert result["items"][0]["price"] == "100"
    assert result["items"][0]["amount"] == "100.00"
    assert value["items"][0]["quantity"] == "1"
    assert value["items"][0]["price"] == "100"
