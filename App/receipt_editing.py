"""Editable receipt values, immutable OCR evidence, and exact money arithmetic."""
from __future__ import annotations

import copy
import csv
import json
import re
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext

from App.receipt_math import verify_vat
from App.numeric_settings import load_numeric_settings

PAYABLE = {"product", "service", "tax"}
CENT = Decimal(".01")


def parse_number(value):
    if value is None or isinstance(value, (bool, list, dict)):
        return None
    text = re.sub(r"\s+", "", str(value)).strip()
    text = re.sub(r"(?:KZT|ТГ|ТЕНГЕ|[TТ₸])$", "", text, flags=re.IGNORECASE)
    if not text or len(text) > 40 or not re.fullmatch(r"\d+(?:[.,]\d+)*", text):
        return None
    if "." in text and "," in text:
        decimal_separator = "." if text.rfind(".") > text.rfind(",") else ","
        text = text.replace("," if decimal_separator == "." else ".", "")
    text = text.replace(",", ".")
    if text.count(".") > 1:
        return None
    try:
        result = Decimal(text)
        return result if result.is_finite() and Decimal(0) <= result < Decimal("1e15") else None
    except InvalidOperation:
        return None


def money(value):
    with localcontext() as context:
        context.prec = 60
        return format(value.quantize(CENT, rounding=ROUND_HALF_UP), ".2f")


def initialize_row(row, threshold=87):
    row.setdefault("name", row.get("name_raw", ""))
    row.setdefault("quantity", row.get("quantity_raw", ""))
    row.setdefault("price", row.get("price_raw", ""))
    row.setdefault("name_confirmed", False)
    row.setdefault("name_origin", "ocr")
    row.setdefault("article", "")
    row.setdefault("catalog_key", "")
    row.setdefault("match_review_reason", row.get("review_reason", ""))
    best = (row.get("match") or {}).get("best")
    if row["kind"] == "product" and best and not row.get("matching_applied"):
        articles = sorted({source["article"] for source in best.get("sources", []) if source.get("article")})
        row["articles"] = articles
        accepted = best["score"] >= threshold
        learned = (row.get("match") or {}).get("source") == "manual_mapping"
        if accepted:
            row.update(name=best["name"], name_origin="manual_mapping" if learned else "catalog", catalog_key=best["normalized"],
                       article=best.get("article", "") if learned else articles[0] if len(articles) == 1 else "")
        row["needs_name_review"] = not accepted or (len(articles) > 1 and not learned)
        row["matching_applied"] = True
    row.setdefault("needs_name_review", row["kind"] == "product" and not best)
    return row


def update_receipt(receipt):
    receipt.setdefault("revision", 0)
    receipt.setdefault("audit_log", [])
    receipt.setdefault("finalized", False)
    receipt.setdefault("expected_total", receipt.get("total_raw", ""))
    receipt.setdefault("expected_total_source", "ocr")
    if parse_number(receipt["expected_total"]) is None and receipt["expected_total_source"] != "user":
        # The OCR may have placed the printed total inside the last table band.
        # Reuse that evidence for comparison; never change the raw OCR fields or
        # derive an expected total by adding product prices.
        total_row = next((row for row in reversed(receipt["rows"])
                          if row.get("kind") == "total" and parse_number(row.get("price_raw")) is not None), None)
        if total_row:
            receipt["expected_total"] = total_row["price_raw"]
            receipt["expected_total_source"] = "ocr_total_row"
            receipt["expected_total_ocr_row"] = total_row["row"]
    threshold = receipt.get("pass_similarity", 87)
    totals = {kind:Decimal(0) for kind in PAYABLE}
    missing = []
    for row in receipt["rows"]:
        initialize_row(row, threshold)
        row["line_total"] = None
        fields = []
        if row["kind"] in PAYABLE:
            quantity, price = parse_number(row["quantity"]), parse_number(row["price"])
            if quantity is None:
                fields.append("quantity")
            if price is None:
                fields.append("price")
            if fields:
                missing.append(dict(row=row["row"], fields=fields, name=row["name"]))
            else:
                with localcontext() as context:
                    context.prec = 60
                    line_total = (quantity*price).quantize(CENT, rounding=ROUND_HALF_UP)
                    row["line_total"] = money(line_total)
                    totals[row["kind"]] += line_total
        row["invalid_fields"] = fields
        name_pending = row["kind"] == "product" and row["needs_name_review"] and not row["name_confirmed"]
        row["needs_review"] = name_pending or bool(fields)
        reasons = []
        if name_pending:
            reasons.append(row.get("match_review_reason") or "Выберите или подтвердите название товара")
        if fields:
            reasons.append("Проверьте " + ", ".join("количество" if field == "quantity" else "цену" for field in fields))
        row["review_reason"] = "; ".join(reasons)
    with localcontext() as context:
        context.prec = 60
        computed = sum(totals.values())
    expected = parse_number(receipt["expected_total"])
    if missing:
        status = "incomplete"
    elif expected is None:
        status = "missing_original_total"
    else:
        status = "match" if computed == expected.quantize(CENT, rounding=ROUND_HALF_UP) else "mismatch"
    settings = load_numeric_settings()
    receipt["vat_rate_percent"] = settings["vat_rate_percent"]
    receipt["vat_rate_source"] = settings["vat_rate_source"]
    vat = verify_vat(receipt["rows"],totals["product"]+totals["service"],totals["tax"],parse_number,
                     settings["vat_rate_percent"])
    vat["rate_source"] = settings["vat_rate_source"]
    for row in receipt["rows"]:
        if row["kind"]=="tax":
            row["vat_status"] = vat["status"]
            if vat["status"] in {"mismatch","invalid_rate"}:
                row["needs_review"] = True
                row["review_reason"] = "; ".join(filter(None,[row["review_reason"],vat.get("reason", "Проверьте НДС")]))
    validation_status = status if status!="match" or vat["status"] in {"match","not_applicable"} else "vat_mismatch"
    receipt["calculation"] = dict(status=status,validation_status=validation_status,vat=vat, products_subtotal=money(totals["product"]),
        services_subtotal=money(totals["service"]), tax_subtotal=money(totals["tax"]),
        computed_total=money(computed), expected_total=money(expected) if expected is not None else None,
        difference=money(computed-expected) if expected is not None and not missing else None,
        missing_rows=missing, missing_original_total=expected is None,
        rule="sum(quantity × price), including separate VAT rows")
    receipt["product_count"] = sum(row["kind"] == "product" for row in receipt["rows"])
    receipt["review_count"] = sum(row["needs_review"] for row in receipt["rows"])
    receipt["can_finalize"] = receipt["review_count"] == 0 and validation_status == "match"
    if receipt["finalized"] and validation_status!="match":
        receipt["finalized"] = False
    return receipt


def apply_edit(receipt, payload, catalog_lookup):
    result = update_receipt(copy.deepcopy(receipt))
    if payload.get("revision") != result["revision"]:
        raise RuntimeError("Чек уже изменён в другом окне. Обновите данные и повторите правку.")
    changes = []
    if "expected_total" in payload:
        value = payload["expected_total"]
        if parse_number(value) is None:
            raise ValueError("Введите итог с фотографии числом.")
        changes.append(dict(field="expected_total", before=result["expected_total"], after=str(value)))
        result["expected_total"], result["expected_total_source"] = str(value), "user"
    if "row" in payload:
        row = next((row for row in result["rows"] if row["row"] == payload["row"]), None)
        if row is None:
            raise ValueError("Строка чека не найдена.")
        key = payload.get("catalog_key")
        if key:
            entry = catalog_lookup(key)
            if entry is None:
                raise ValueError("Товар отсутствует в текущем справочнике.")
            articles = sorted({source["article"] for source in entry["sources"] if source["article"]})
            article = str(payload.get("article", ""))
            if len(articles) > 1 and article not in articles:
                raise ValueError("Выберите артикул: у этого названия их несколько.")
            if article and articles and article not in articles:
                raise ValueError("Артикул не относится к выбранному товару.")
            changes.append(dict(row=row["row"], field="name", before=row["name"], after=entry["name"]))
            row.update(name=entry["name"], catalog_key=key, article=article or (articles[0] if articles else ""), name_origin="catalog_user", name_confirmed=False)
        elif "name" in payload:
            value = str(payload["name"]).strip()
            if not value or len(value) > 1000:
                raise ValueError("Введите название товара.")
            if value != row["name"]:
                changes.append(dict(row=row["row"], field="name", before=row["name"], after=value))
                row.update(name=value, catalog_key="", article="", name_origin="user", needs_name_review=True, name_confirmed=False)
        for field in ("quantity", "price"):
            if field in payload:
                value = str(payload[field]).strip()
                if parse_number(value) is None:
                    raise ValueError("Количество и цена должны быть неотрицательными числами.")
                if value != row[field]:
                    changes.append(dict(row=row["row"], field=field, before=row[field], after=value))
                    row[field] = value
        if payload.get("confirm_name"):
            row["name_confirmed"] = True
            changes.append(dict(row=row["row"], field="name_confirmed", after=True))
    result["finalized"] = False
    result = update_receipt(result)
    if payload.get("finalize"):
        if not result["can_finalize"]:
            raise ValueError("Проверьте отмеченные строки и добейтесь совпадения итоговой суммы.")
        result["finalized"] = True
        result["finalized_at"] = datetime.now(timezone.utc).isoformat()
        changes.append(dict(field="finalized", after=True))
    result["revision"] += 1
    result["audit_log"].append(dict(at=datetime.now(timezone.utc).isoformat(), changes=changes))
    return result


def save_receipt(directory, receipt):
    temporary = directory / "receipt.json.tmp"
    temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(directory / "receipt.json")
    with (directory / "receipt.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["№", "Тип", "Название чека", "Артикул", "Количество", "Цена", "Сумма", "Проверить", "Название OCR"])
        for row in receipt["rows"]:
            if row["kind"] == "total":
                continue
            writer.writerow([row.get("item_number_raw", ""), row["kind"], row["name"], row["article"], row["quantity"],
                             row["price"], row["line_total"], row["review_reason"], row.get("name_raw", "")])
