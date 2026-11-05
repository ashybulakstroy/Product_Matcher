"""Exact VAT verification for this receipt format, separate from OCR geometry."""
from decimal import Decimal, ROUND_HALF_UP, localcontext
import re

CENT = Decimal(".01")


def amount(value):
    if value is None:
        return None
    with localcontext() as context:
        context.prec = 60
        return format(value.quantize(CENT,rounding=ROUND_HALF_UP),".2f")


def verify_vat(rows,taxable_base,actual_tax,parse_number,default_rate="16"):
    taxes = [row for row in rows if row["kind"]=="tax"]
    result = dict(status="not_applicable",taxable_base=amount(taxable_base),actual_tax=amount(actual_tax),
                  expected_tax=None,difference=None,rate_percent=None,rate_source=None,rows=[row["row"] for row in taxes])
    if not taxes:
        return result
    # The photographed label is OCR evidence, not configuration: the printed
    # percentage can itself be misread. Numeric recovery verifies against the
    # configured application rate instead.
    rate = parse_number(default_rate)
    if rate is None or rate > 100:
        result.update(status="invalid_rate",reason="Недействительная ставка НДС в конфигурации")
        return result
    result.update(rate_percent=format(rate,"f"),rate_source="env_file")
    if any(row.get("invalid_fields") for row in rows if row["kind"] in {"product","service","tax"}):
        result.update(status="incomplete",reason="Не все количества и цены прочитаны для проверки НДС")
        return result
    # Torgsoft often prints VAT as an integer amount in quantity × unit price 1.
    # Match that displayed precision rather than treating omitted cents as an
    # OCR error. Explicit decimals keep the corresponding stricter precision.
    precisions = []
    for row in taxes:
        quantity = parse_number(row.get("quantity",row.get("quantity_raw")))
        price = parse_number(row.get("price",row.get("price_raw")))
        value = quantity if price==1 else price if quantity==1 else quantity*price
        precisions.append(min(2,max(0,-value.as_tuple().exponent)))
    quantum = Decimal(1).scaleb(-max(precisions))
    with localcontext() as context:
        context.prec = 60
        unrounded = taxable_base*rate/100
        expected = unrounded.quantize(quantum,rounding=ROUND_HALF_UP)
        difference = actual_tax-expected
    result.update(status="match" if difference==0 else "mismatch",expected_tax=amount(expected),difference=amount(difference),
                  unrounded_tax=format(unrounded,"f"),rounding_quantum=format(quantum,"f"))
    if difference:
        result["reason"] = f"НДС по строкам {amount(actual_tax)}, ожидается {amount(expected)} при ставке {rate}%"
    return result
