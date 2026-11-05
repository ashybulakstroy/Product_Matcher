"""Adapter between a verified №16 receipt snapshot and the invoice stage.

Reads the receipt produced by the existing pipeline (`App.web_pipeline`,
`App.receipt_editing`), never re-runs OCR, never mutates the source file and
never invents values. Every payable line stays visible; unresolved rows are
flagged, not dropped. Output field names follow the donor generation contract
(`items` / `vat_amount` / `total` / `date` / `order_number` / `warnings`) so
the planner and document builder can be adapted without re-validating amounts.
"""
from __future__ import annotations

import hashlib
import json
from decimal import ROUND_HALF_UP, Decimal, localcontext
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from App.receipt_editing import parse_number

CENT = Decimal(".01")
# Payable lines that become invoice items (tax/total/header are structural).
ITEM_KINDS = ("product", "service")
KNOWN_KINDS = ("product", "service", "tax", "total", "header")


def _decimal(values: Mapping[str, Any], key: str):
    """Parse a money/number field with the receipt's own number rules."""
    return parse_number(values.get(key))


def _canonical(payload) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      default=str).encode("utf-8")


class VatSummary(BaseModel):
    """Document-level VAT taken from the receipt's own tax rows."""

    model_config = ConfigDict(extra="forbid")

    amount: Decimal | None = None
    status: str = "not_applicable"
    rate_percent: Decimal | None = None
    rate_source: str | None = None
    tax_row_ids: list[int] = Field(default_factory=list)


class ReceiptLine(BaseModel):
    """One payable receipt line. Never derived, only mirrored from the snapshot."""

    model_config = ConfigDict(extra="forbid")

    row_id: int
    line_kind: str
    name: str
    ocr_name: str | None = None
    edited_name: bool = False
    name_confirmed: bool = False
    name_origin: str | None = None
    article: str = ""
    catalog_key: str = ""
    articles: list[str] = Field(default_factory=list)
    quantity: Decimal | None = None
    quantity_raw: str = ""
    price: Decimal | None = None
    price_raw: str = ""
    line_amount: Decimal | None = None
    line_amount_raw: str = ""
    invalid_fields: list[str] = Field(default_factory=list)
    needs_review: bool = False
    review_reason: str | None = None


class ReceiptSnapshot(BaseModel):
    """Immutable image of a verified receipt handed over to the invoice stage."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    revision: int = 0
    filename: str | None = None
    finalized: bool = False
    finalized_at: str | None = None
    date: str | None = None
    order_number: str | None = None
    validation_status: str = "incomplete"
    calculation_status: str = "incomplete"
    expected_total: Decimal | None = None
    computed_total: Decimal | None = None
    products_subtotal: Decimal | None = None
    services_subtotal: Decimal | None = None
    vat: VatSummary = VatSummary()
    lines: list[ReceiptLine] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    fingerprint: str = ""

    @model_validator(mode="after")
    def _build_fingerprint(self) -> "ReceiptSnapshot":
        payload = {
            "job_id": self.job_id,
            "revision": self.revision,
            "date": self.date,
            "order_number": self.order_number,
            "finalized": self.finalized,
            "validation_status": self.validation_status,
            "calculation_status": self.calculation_status,
            "expected_total": str(self.expected_total) if self.expected_total is not None else None,
            "vat_amount": str(self.vat.amount) if self.vat.amount is not None else None,
            "vat_status": self.vat.status,
            "lines": sorted(
                (
                    {
                        "row_id": line.row_id,
                        "line_kind": line.line_kind,
                        "name": line.name,
                        "ocr_name": line.ocr_name,
                        "edited_name": line.edited_name,
                        "name_confirmed": line.name_confirmed,
                        "article": line.article,
                        "catalog_key": line.catalog_key,
                        "quantity": str(line.quantity) if line.quantity is not None else None,
                        "quantity_raw": line.quantity_raw,
                        "price": str(line.price) if line.price is not None else None,
                        "price_raw": line.price_raw,
                        "line_amount": str(line.line_amount) if line.line_amount is not None else None,
                        "line_amount_raw": line.line_amount_raw,
                        "invalid_fields": line.invalid_fields,
                        "needs_review": line.needs_review,
                    }
                    for line in self.lines
                ),
                key=lambda item: item["row_id"],
            ),
        }
        self.fingerprint = hashlib.sha256(_canonical(payload)).hexdigest()
        return self


def snapshot_from_receipt(receipt: Mapping[str, Any], job_id: str | None = None) -> ReceiptSnapshot:
    """Build the snapshot from the receipt dict produced by the №16 pipeline.

    Raises ``ValueError`` on structurally impossible input; soft problems land
    in ``warnings`` and stay visible instead of being hidden or repaired.
    """
    if not isinstance(receipt, Mapping):
        raise ValueError("Снимок чека должен быть JSON-объектом.")
    rows = receipt.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Снимок чека не содержит строк.")
    resolved_job_id = job_id or receipt.get("job_id")
    if not resolved_job_id:
        raise ValueError("В снимке чека нет job_id (не заполнено задание).")
    calculation = (receipt.get("calculation") or {}) if isinstance(receipt.get("calculation"), dict) else {}
    warnings: list[str] = list(relation_warnings(receipt))

    lines: list[ReceiptLine] = []
    seen_rows: set[int] = set()
    tax_row_ids: list[int] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Строка чека не является объектом.")
        row_number = row.get("row")
        if not isinstance(row_number, int) or isinstance(row_number, bool) or row_number < 1:
            raise ValueError(f"У строки чека отсутствует устойчивый номер строки: {row_number!r}.")
        if row_number in seen_rows:
            raise ValueError(f"В снимке чека продублирован номер строки {row_number}.")
        seen_rows.add(row_number)
        kind = row.get("kind")
        if kind not in KNOWN_KINDS:
            warnings.append(f"Строка {row_number}: неизвестный тип {kind!r}, из счёта исключена.")
            continue
        if kind == "tax":
            tax_row_ids.append(row_number)
            continue
        if kind not in ITEM_KINDS:
            # tax/total/header stay out of payable items; they are structural.
            continue
        name = str(row.get("name") or "").strip()
        if not name:
            raise ValueError(f"Строка {row_number}: пустое название товарной строки.")
        # `quantity_raw` is the untouched OCR evidence; `quantity` is the
        # current (possibly edited) value. The adapter mirrors both.
        quantity_raw = str(row.get("quantity_raw") if row.get("quantity_raw") is not None
                           else row.get("quantity") if row.get("quantity") is not None else "")
        price_raw = str(row.get("price_raw") if row.get("price_raw") is not None
                        else row.get("price") if row.get("price") is not None else "")
        line_amount_raw = str(row.get("line_total") if row.get("line_total") is not None else "")
        quantity = _decimal(row, "quantity")
        price = _decimal(row, "price")
        line_amount = _decimal(row, "line_total")
        invalid_fields = list(row.get("invalid_fields") or [])
        if quantity is None and "quantity" not in invalid_fields:
            invalid_fields.append("quantity")
        if price is None and "price" not in invalid_fields:
            invalid_fields.append("price")
        if quantity is not None and price is not None:
            with localcontext() as context:
                context.prec = 60
                recomputed = (quantity * price).quantize(CENT, rounding=ROUND_HALF_UP)
            if line_amount is not None and recomputed != line_amount.quantize(CENT, rounding=ROUND_HALF_UP):
                warnings.append(f"Строка {row_number}: количество × цена не равно сумме строки.")
            elif line_amount is None:
                warnings.append(f"Строка {row_number}: сумма строки не вычислена.")
        elif line_amount is None:
            warnings.append(f"Строка {row_number}: не распознаны количество или цена.")
        lines.append(ReceiptLine(
            row_id=row_number,
            line_kind=kind,
            name=name,
            ocr_name=str(row.get("name_raw") or "") or None,
            edited_name=row.get("name_origin") not in (None, "ocr"),
            name_confirmed=bool(row.get("name_confirmed")),
            name_origin=str(row.get("name_origin") or "") or None,
            article=str(row.get("article") or ""),
            catalog_key=str(row.get("catalog_key") or ""),
            articles=list(row.get("articles") or []),
            quantity=quantity,
            quantity_raw=quantity_raw,
            price=price,
            price_raw=price_raw,
            line_amount=line_amount,
            line_amount_raw=line_amount_raw,
            invalid_fields=invalid_fields,
            needs_review=bool(row.get("needs_review")) or bool(invalid_fields),
            review_reason=str(row.get("review_reason") or "") or None,
        ))

    vat = VatSummary(
        amount=_decimal(calculation, "tax_subtotal") or None,
        status=str(calculation.get("vat", {}).get("status") if isinstance(calculation.get("vat"), dict)
                   else calculation.get("vat_status")) or "not_applicable",
        rate_percent=Decimal(str(receipt.get("vat_rate_percent"))) if receipt.get("vat_rate_percent") else None,
        rate_source=str(receipt.get("vat_rate_source") or "") or None,
        tax_row_ids=tax_row_ids,
    )
    if not tax_row_ids:
        vat.amount = None
    return ReceiptSnapshot(
        job_id=str(resolved_job_id),
        revision=int(receipt.get("revision") or 0),
        filename=str(receipt.get("filename") or "") or None,
        finalized=bool(receipt.get("finalized")),
        finalized_at=str(receipt.get("finalized_at") or "") or None,
        date=str(receipt.get("date") or "") or None,
        order_number=str(receipt.get("order_number") or "") or None,
        validation_status=str(calculation.get("validation_status") or receipt.get("calculation", {}).get("status")
                              or "incomplete"),
        calculation_status=str(calculation.get("status") or "incomplete"),
        expected_total=_decimal(calculation, "expected_total") or _decimal(receipt, "expected_total") or None,
        computed_total=_decimal(calculation, "computed_total") or None,
        products_subtotal=_decimal(calculation, "products_subtotal") or None,
        services_subtotal=_decimal(calculation, "services_subtotal") or None,
        vat=vat,
        lines=lines,
        warnings=warnings,
    )


def relation_warnings(receipt: Mapping[str, Any]) -> list[str]:
    """Warnings carried inside the receipt itself (never silently dropped)."""
    raw = receipt.get("warnings")
    if isinstance(raw, list):
        return [str(item) for item in raw]
    return []


def snapshot_from_directory(directory: Path) -> ReceiptSnapshot:
    """Read `receipt.json` of a finished job (materialized, not recreated)."""
    path = Path(directory) / "receipt.json"
    if not path.is_file():
        raise FileNotFoundError(f"Снимок чека не найден: {path.name} в {directory}.")
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Снимок чека не читается из {path}: {error}") from error
    return snapshot_from_receipt(receipt)


def to_generation_source(snapshot: ReceiptSnapshot) -> dict[str, Any]:
    """Donor-compatible document source: items/vat_amount/total/date/number."""
    items = []
    for line in snapshot.lines:
        items.append({
            "row_id": str(line.row_id),
            "name": line.name,
            "ocr_name": line.ocr_name,
            "edited": line.edited_name or line.name_confirmed,
            "name_confirmed": line.name_confirmed,
            "name_origin": line.name_origin,
            "article": line.article,
            "catalog_key": line.catalog_key,
            "articles": line.articles,
            "quantity": str(line.quantity) if line.quantity is not None else (line.quantity_raw or ""),
            "price": str(line.price) if line.price is not None else (line.price_raw or ""),
            "amount": str(line.line_amount) if line.line_amount is not None else (line.line_amount_raw or ""),
            "invalid_fields": line.invalid_fields,
            "needs_review": line.needs_review,
            "line_kind": line.line_kind,
        })
    return {
        "source_job_id": snapshot.job_id,
        "source_revision": snapshot.revision,
        "source_fingerprint": snapshot.fingerprint,
        "finalized": snapshot.finalized,
        "validation_status": snapshot.validation_status,
        "items": items,
        "vat_amount": str(snapshot.vat.amount) if snapshot.vat.amount is not None else None,
        "vat_status": snapshot.vat.status,
        "total": str(snapshot.expected_total) if snapshot.expected_total is not None else None,
        "products_subtotal": str(snapshot.products_subtotal) if snapshot.products_subtotal is not None else None,
        "services_subtotal": str(snapshot.services_subtotal) if snapshot.services_subtotal is not None else None,
        "date": snapshot.date,
        "order_number": snapshot.order_number,
        "warnings": list(snapshot.warnings),
    }
