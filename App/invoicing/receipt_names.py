"""A confirmed reference name always wins over raw OCR evidence in the invoice stage.

Mirrors the donor helper; `edited`/`ocr_name` fields come verbatim from
`App.invoicing.receipt_adapter`. Nothing else governs which name is displayed.
"""


def torgsoft_name(item):
    """A confirmed reference overrides OCR, including previously edited rows."""
    if item.get("from_mapping") and item.get("reference_name"):
        return item["reference_name"]
    return item.get("name") if item.get("edited") else item.get("ocr_name") or item.get("name")
