"""Mapping rules built from confirmed user-supplied receipt/invoice pairs.

Adapted from the donor `learned_mapping.py`: same arithmetic (factors,
service quantities, conflict detection), imports pointed at `App.invoicing`,
paths relative to the №16 project root. A rule is only produced by pairs that
passed all readiness checks; proposals and suggestions are never rules.
"""
from __future__ import annotations

import json
import os
import re
from decimal import ROUND_HALF_UP, Decimal
from functools import lru_cache
from pathlib import Path

from App.invoicing.user_mapping import normalize_mapping_key

ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = ROOT / "Data" / "reference" / "Training_Pair"
# Tolerance is legal here ONLY for the historical training-pair bookkeeping and
# matching against user-supplied documents. The strict receipt validation of
# the №16 pipeline stays exact and is never relaxed by this constant.
VAT_TOLERANCE = Decimal("0.99")
CENT = Decimal("0.01")


def _trusted_reference(item):
    reference = item.get("reference_name") or ""
    observed = item["name"] if item.get("edited") else item.get("ocr_name") or item["name"]
    return bool(item.get("from_mapping") or (float(item.get("name_confidence") or 0) > 80 and
                re.findall(r"\d+", observed) == re.findall(r"\d+", reference)))


def prepare_training_pair(pair, reviewed_receipt=None):
    source = reviewed_receipt or pair.get("training_receipt") or pair["predzakaz"]
    pair["training_receipt"] = source
    pair["training_ready"] = False
    pair["training_notes"] = []
    pair["links"] = []
    invoice = pair["invoice"]
    header_warnings = {"Не распознана дата", "Отсутствует дата чека",
                       "Не распознан номер заказа", "Отсутствует номер заказа"}
    numeric_warnings = [message for message in source.get("warnings", []) if message not in header_warnings]
    if numeric_warnings or invoice.get("warnings") or not source.get("items"):
        pair["training_notes"].append("Есть ошибки извлечения; обучение отложено до исправления данных")
        return pair
    net = [Decimal(str(row["quantity"])) * Decimal(str(row["price"])) for row in source["items"]]
    net_total = sum(net, Decimal(0))
    if not net_total or any(amount <= 0 for amount in net) or source.get("total") is None:
        pair["training_notes"].append("Не заполнены числовые поля предзаказа")
        return pair
    vat = Decimal(str(source.get("vat_amount") or 0))
    if abs(net_total + vat - Decimal(str(source["total"]))) > VAT_TOLERANCE:
        pair["training_notes"].append("Арифметика предзаказа не сходится")
        return pair
    if invoice.get("total") is None or abs(Decimal(str(invoice["total"])) - Decimal(str(source["total"]))) > VAT_TOLERANCE:
        pair["training_notes"].append("Итоги документов различаются")
        return pair
    if not vat_matches(source.get("vat_amount"), invoice.get("vat_amount")):
        pair["training_notes"].append("НДС документов различается больше допустимой погрешности 0,99")
        return pair
    invoice_rows = invoice["items"]
    if len(invoice_rows) != len(net):
        pair["training_notes"].append("Объединение и разделение строк требует отдельного правила")
        return pair
    allocated = Decimal(0)
    used = set()
    for index, amount in enumerate(net):
        share = (vat * amount / net_total).quantize(CENT, rounding=ROUND_HALF_UP)
        if index == len(net) - 1:
            share = vat - allocated
        allocated += share
        candidates = [row for row in invoice_rows if row["row_id"] not in used
                      and abs(Decimal(row["amount"]) - amount - share) <= VAT_TOLERANCE]
        if len(candidates) > 1:
            same_quantity = [row for row in candidates
                             if Decimal(row["quantity"]) == Decimal(str(source["items"][index]["quantity"]))]
            if len(same_quantity) == 1:
                candidates = same_quantity
        if len(candidates) != 1 or candidates[0]["row_id"] in used:
            pair["links"] = []
            pair["training_notes"].append("Неоднозначное сопоставление строк по суммам")
            return pair
        target = candidates[0]
        if Decimal(target["quantity"]) <= 0:
            return pair
        used.add(target["row_id"])
        pair["links"].append({"source_index": index,
                              "predzakaz_row_id": source["items"][index].get("row_id", f"predzakaz_row_{index+1:03d}"),
                              "invoice_row_id": target["row_id"]})
    pair["training_ready"] = True
    pair["status"] = "learned"
    pair["training_basis"] = "user_supplied_real_pair"
    return pair


def vat_matches(left, right) -> bool:
    """Donor tolerance for separate document VATs.

    №16 adaptation: None means "no separate VAT" (the №16 receipt total already
    includes VAT), so two documents without a separate tax never conflict.
    """
    left = left or 0
    right = right or 0
    return abs(Decimal(str(left)) - Decimal(str(right))) <= VAT_TOLERANCE


@lru_cache(maxsize=4)
def _load_rules(signature, database_id):
    rules = {}
    cases = []
    for filename, _, _ in signature:
        case = json.loads(Path(filename).read_text(encoding="utf-8"))
        cases.extend([*case.get("mapping_history", []), case])
    for case in sorted(cases, key=lambda value: value.get("created_at", "")):
        if not case.get("training_ready"):
            continue
        if case.get("database_id") and case["database_id"] != database_id:
            continue
        source_rows = case.get("training_receipt", case["predzakaz"])["items"]
        invoice_rows = {item["row_id"]: item for item in case["invoice"]["items"]}
        for link in case.get("links", []):
            source = source_rows[link["source_index"]]
            target = invoice_rows[link["invoice_row_id"]]
            factor = Decimal(target["quantity"]) / Decimal(str(source["quantity"]))
            rule = {
                "case_id": case["case_id"],
                "name": target["name"],
                "unit": target["unit"],
                "quantity_factor": str(factor),
                "amount_factor": str(Decimal(target["amount"]) / (Decimal(str(source["quantity"])) * Decimal(str(source["price"])))),
                "quantity_mode": "service" if link.get("service") or target["unit"] == "усл" else "factor",
                "source_name": source["name"],
                "identity": "name",
                "source_index": link["source_index"],
                "origin_case_id": case.get("validation_original_case_id") or case["case_id"],
            }
            if rule["quantity_mode"] == "service":
                rule["service_quantity"] = str(Decimal(str(target["quantity"])).normalize())
            for field in ("source_identity", "source_article", "source_card", "catalog_id", "code"):
                if field in link:
                    rule[field] = link[field]
            aliases = {source["name"]}
            if _trusted_reference(source):
                aliases.add(source.get("reference_name"))
            raw_rows = {item.get("row_id", f"predzakaz_row_{index:03d}"): item
                        for index, item in enumerate(case["predzakaz"]["items"], 1)}
            raw = raw_rows.get(link["predzakaz_row_id"], {})
            aliases.update((raw.get("ocr_name"), raw.get("name")))
            for alias in aliases - {None, ""}:
                key = normalize_mapping_key(alias)
                previous = rules.get(key)
                if previous is None:
                    rules[key] = {**rule, "alternatives": [rule], "examples": [rule]}
                    continue
                if previous.get("conflict"):
                    continue
                same_source = (previous.get("source_identity") == rule.get("source_identity")
                               if previous.get("source_identity") and rule.get("source_identity") else
                               (normalize_mapping_key(previous["source_name"]) == normalize_mapping_key(rule["source_name"]) or
                                (previous["origin_case_id"], previous["source_index"]) == (rule["origin_case_id"], rule["source_index"])))
                variant = _variant_identity(rule)
                existing = next((value for value in previous["alternatives"]
                                 if _variant_identity(value) == variant), None)
                if not same_source or (existing and _quantity_signature(existing) != _quantity_signature(rule)):
                    rules[key] = {"conflict": True}
                    continue
                previous["examples"].append(rule)
                if existing is None:
                    previous["alternatives"].append(rule)
    return rules


def _variant_identity(rule):
    return (rule.get("code") or rule.get("catalog_id") or normalize_mapping_key(rule["name"]),
            rule["unit"], rule["quantity_mode"])


def _quantity_signature(rule):
    return Decimal(rule.get("service_quantity", "1")) if rule["quantity_mode"] == "service" else Decimal(rule["quantity_factor"])


def mapping_rules(training_dir=TRAINING_DIR, database_id=None):
    training_path = Path(training_dir)
    signature = tuple(
        (str(path.resolve()), path.stat().st_mtime_ns, path.stat().st_size)
        for path in sorted(training_path.glob("case_*/expected.json"))
    )
    return _load_rules(signature, database_id or os.environ.get("TORGSOFT_CATALOG_DATABASE_ID") or "local")


def source_rule(item, rules):
    names = (item.get("name"),) if item.get("edited") else (item.get("ocr_name"), item.get("name"))
    if _trusted_reference(item):
        names += (item.get("reference_name"),)
    matches = [rules[normalize_mapping_key(name)] for name in names
               if name and normalize_mapping_key(name) in rules]
    return next((match for match in matches if not match.get("conflict")), None)


def training_alternatives(item, training_dir=TRAINING_DIR, database_id=None):
    rule = source_rule(item, mapping_rules(training_dir, database_id))
    return rule.get("alternatives", []) if rule else []


def apply_training_pairs(receipt: dict, training_dir=TRAINING_DIR, *, case_id=None, database_id=None) -> dict:
    rules = mapping_rules(training_dir, database_id)
    result = {"status": "unmapped", "items": [], "warnings": [],
              "total": receipt.get("total"), "vat_amount": receipt.get("vat_amount")}
    source_items = receipt.get("items", [])
    net_total = sum((Decimal(str(item["quantity"])) * Decimal(str(item["price"])) for item in source_items), Decimal(0))
    vat = Decimal(str(receipt.get("vat_amount") or 0))
    if receipt.get("total") is None or abs(net_total + vat - Decimal(str(receipt["total"]))) > VAT_TOLERANCE:
        result["warnings"].append("Арифметика исходного чека не сходится; расчёт 1С отложен")
        return result
    for index, item in enumerate(source_items):
        # Prefer the observed name: a weak reference candidate must not redirect training.
        bundle = source_rule(item, rules)
        rule = bundle
        if bundle and case_id:
            rule = next((example for example in reversed(bundle["examples"])
                         if example["case_id"] == case_id and example["source_index"] == index), None)
        source_amount = Decimal(str(item["quantity"])) * Decimal(str(item["price"]))
        if rule is None:
            result["warnings"].append(f"Нет однозначной учебной связи: {item['name']}")
            continue
        quantity = Decimal(rule.get("service_quantity", "1")) if rule["quantity_mode"] == "service" \
            else Decimal(str(item["quantity"])) * Decimal(rule["quantity_factor"])
        if quantity <= 0:
            result["warnings"].append(f"Некорректное количество: {item['name']}")
            continue
        amount = (source_amount * Decimal(rule["amount_factor"])).quantize(CENT, rounding=ROUND_HALF_UP)
        result["items"].append({
            **{key: value for key, value in rule.items() if key not in {"alternatives", "examples"}},
            "alternatives": bundle["alternatives"], "source_index": index,
            "quantity": str(quantity),
            "price": str((amount / quantity).quantize(CENT, rounding=ROUND_HALF_UP)),
            "amount": str(amount.quantize(CENT, rounding=ROUND_HALF_UP)),
        })
    if receipt.get("warnings"):
        result["warnings"].append("В исходном OCR есть ошибки; требуется проверка")
    if len(result["items"]) == len(source_items) and result["items"]:
        generated_sum = sum((Decimal(row["amount"]) for row in result["items"]), Decimal(0))
        difference = Decimal(str(receipt["total"])) - generated_sum
        if abs(difference) <= VAT_TOLERANCE:
            last = result["items"][-1]
            amount = Decimal(last["amount"]) + difference
            last["amount"] = str(amount.quantize(CENT))
            last["price"] = str((amount / Decimal(last["quantity"])).quantize(CENT, rounding=ROUND_HALF_UP))
        else:
            result["warnings"].append("Сумма после применения учебных правил отличается от чека")
    if result["items"]:
        result["status"] = "review" if result["warnings"] else "applied"
    return result
