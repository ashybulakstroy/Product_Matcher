"""Frozen 1C result snapshot (contract 1.1).

Builds the document handed to 1C from the already verified invoice model and
the verified receipt snapshot. Nothing here runs OCR or re-validates receipt
amounts: ``invoice_model`` remains the single source of the invoice, and the
receipt fingerprint plus selections are captured in an input revision so a
later edit can invalidate the confirmation.

The checksum is the SHA-256 of the canonical UTF-8 JSON *without* the checksum
field: sorted keys, compact separators, Cyrillic not ASCII-escaped — the same
rule as ``receipt_adapter._canonical``.
"""
from __future__ import annotations

import hashlib
import json
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

from App.invoicing.invoice_service import _catalog_signature, _issuer_signature
from App.invoicing.receipt_adapter import snapshot_from_receipt

SCHEMA_VERSION = "1.1"
STATUS_CONFIRMED = "confirmed"
CURRENCY = "KZT"
CONFIRMED_TARGETS = {"manual", "remembered", "learned"}


def _canonical(payload) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      default=str).encode("utf-8")


def checksum(payload: dict) -> str:
    """Checksum of the snapshot body; the ``checksum`` field itself is ignored."""
    body = {key: value for key, value in payload.items() if key != "checksum"}
    return hashlib.sha256(_canonical(body)).hexdigest()


def fingerprint(payload: dict) -> str:
    """Canonical fingerprint of an arbitrary JSON body (receipt idempotency)."""
    return hashlib.sha256(_canonical(payload)).hexdigest()


def input_revision(receipt: dict, selections: dict | None = None, *, database_id=None) -> str:
    """Identifies the inputs a confirmation was made from.

    Changes to the receipt, saved selections, catalog files or the issuer make
    the confirmation stale even when no explicit invalidation happened.
    """
    snapshot = snapshot_from_receipt(receipt)
    payload = {
        "receipt_fingerprint": snapshot.fingerprint,
        "tax_basis": "net-line-prices-v3-dual-mode",
        "selections": selections or {},
        "catalog_signature": _catalog_signature(database_id=database_id),
        "issuer_signature": _issuer_signature(),
    }
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _decimal(value, default: Decimal | None = None) -> Decimal | None:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return default


def _factor_text(row: dict, quantity: Decimal | None, source: Decimal | None) -> str:
    factor = row.get("quantity_factor")
    if factor in (None, ""):
        if row.get("service") or not source:
            factor = Decimal(1)
        else:
            factor = (quantity or Decimal(0)) / source
    number = _decimal(factor)
    if number is None:
        return "1"
    return format(number.normalize(), "f")


def _target(row: dict, allow_missing=False) -> dict | None:
    """One 1C target row; ``None`` means the invoice row has no 1C product."""
    if not row.get("name") and not allow_missing:
        return None
    service = bool(row.get("service"))
    quantity = _decimal(row.get("quantity"))
    source = _decimal(row.get("source_quantity"))
    return {
        "code": row.get("code"),
        "uuid": row.get("uuid"),
        "catalog_id": row.get("catalog_id"),
        "name": row.get("name") or "",
        "unit": row.get("unit") or ("усл" if service else ""),
        "service": service,
        "quantity_mode": row.get("quantity_mode") or ("service" if service else "factor"),
        "quantity_factor": _factor_text(row, quantity, source),
        "quantity": str(row.get("quantity")),
        "price": str(row.get("price")),
        "amount": str(row.get("amount")),
    }


def _allowed_targets(row: dict) -> list[dict]:
    """Confirmed alternatives of the same source card, never search proposals."""
    allowed, seen = [], set()
    for option in row.get("approved_targets") or []:
        target = _target(option)
        if target is None:
            continue
        if option.get("conditions"):
            target["conditions"] = option["conditions"]
        key = (target["code"], target["name"], target["unit"], target["service"],
               target["quantity_mode"], target["quantity"])
        if key in seen:
            continue
        seen.add(key)
        allowed.append(target)
    return allowed


def _source(row: dict, line=None) -> dict:
    article = row.get("source_article") or (getattr(line, "article", "") if line else "") or None
    identity = row.get("source_identity") or (f"torgsoft:article:{article}" if article else None)
    return {
        "name": row.get("source_name"),
        "article": article,
        "identity": identity,
        "quantity": str(row.get("source_quantity")),
        "price": str(row.get("source_price")),
        "amount": str(row.get("source_amount")),
    }


def build_snapshot(task, model: dict, receipt: dict, version: int, now: float | None = None, *, status="confirmed") -> dict:
    """JSON 1.1 snapshot for one confirmation of ``task``.

    ``model`` is the current ``invoice_service.invoice_model`` result; the
    caller supplies a formed invoice; operator confirmation is not required.
    """
    receipt_snapshot = snapshot_from_receipt(receipt)
    receipt_lines = {str(line.row_id): line for line in receipt_snapshot.lines}
    items = []
    for row in model["generation"]["items"]:
        match_status = row.get("match_status")
        target = _target(row, allow_missing=True)
        row_id = str(row.get("row_id"))
        allowed = [target] if match_status not in CONFIRMED_TARGETS else _allowed_targets(row)
        # Exchange uses original net receipt amounts, independently of gross PDF.
        net_amount = Decimal(str(row['source_amount']))
        for candidate in [target] + allowed:
            quantity = Decimal(str(candidate['quantity']))
            candidate['price_includes_vat'] = False
            candidate['amount'] = format(net_amount.quantize(Decimal('.01'), rounding=ROUND_HALF_UP), '.2f')
            candidate['price'] = format((net_amount / quantity).quantize(Decimal('.01'), rounding=ROUND_HALF_UP), '.2f') if quantity else '0.00'
        items.append({
            "row_id": row_id,
            "match_status": match_status,
            "photo_id": task.job_id,
            "source": _source(row, receipt_lines.get(row_id)),
            "target": target,
            "allowed_targets": allowed,
            "selection_policy": "approved_alternatives" if match_status in {"remembered", "learned"} else "explicit",
        })
    document = model["document"]
    moment = time.time() if now is None else now
    confirmed_at = (datetime.fromtimestamp(moment, timezone.utc)
                    .isoformat(timespec="seconds").replace("+00:00", "Z"))
    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "ocr_complete": True,
        "confirmation_origin": "operator" if status == "confirmed" else None,
        "task_id": task.task_id,
        "database_id": task.database_id,
        "request_id": task.request_id,
        "document_uuid": task.document_uuid,
        "result_version": version,
        "revision": model.get("revision"),
        "currency": CURRENCY,
        "prices_include_vat": False,
        "price_includes_vat": False,
        "date": document.get("date"),
        "source_order_number": receipt_snapshot.order_number,
        "vat_rate": (str(receipt_snapshot.vat.rate_percent)
                     if receipt_snapshot.vat.rate_percent is not None else None),
        "vat_amount": document.get("vat_amount"),
        "total": document.get("total"),
        "totals": {
            "amount_without_vat": format(sum((Decimal(item['target']['amount']) for item in items), Decimal(0)), '.2f'),
            "vat": format(Decimal(str(document.get('vat_amount') or 0)), '.2f'),
            "amount_with_vat": document.get('total'),
        },
        "confirmed_at": confirmed_at if status == "confirmed" else None,
        "items": items,
    }
    snapshot["checksum"] = checksum(snapshot)
    return snapshot


def _xml_value(parent, key: str, value) -> None:
    element = ET.SubElement(parent, key)
    if value is None:
        element.set("nil", "true")
    elif key == "conditions" and isinstance(value, dict):
        # The 1C side expects conditions as JSON text in a single element.
        element.text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    elif isinstance(value, dict):
        for child_key, child in value.items():
            _xml_value(element, str(child_key), child)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _xml_value(element, "item", item)
    elif isinstance(value, bool):
        element.text = "true" if value else "false"
    else:
        element.text = str(value)


def to_xml(snapshot: dict) -> bytes:
    """XML form of the same snapshot; semantics of amounts are unchanged."""
    root = ET.Element("TorgsoftInvoice",
                      {"schemaVersion": snapshot.get("schema_version") or SCHEMA_VERSION})
    for key, value in snapshot.items():
        _xml_value(root, str(key), value)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)
