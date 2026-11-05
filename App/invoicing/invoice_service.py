"""Per-job invoice model for the №16 web workspace.

Keeps the donor generation contract behind a thin service boundary for the
stdlib web server: builds the invoice model from the validated electronic
receipt snapshot, stores explicit per-row selections next to the job atomically,
and computes revision/token pairs so stale PDF requests are rejected with 409.
Nothing here creates or sends anything to 1C.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, StrictBool, StrictStr, ValidationError, model_validator

from App.invoicing import invoice_generation as generation
from App.invoicing.catalog_1c import catalog, with_catalog_snapshot
from App.invoicing.receipt_adapter import snapshot_from_receipt, to_generation_source
from App.invoicing.transform_candidates import proposals, resolve_source, search_sources

RESULT_1 = "1"

STATE_FILE = "invoice_state.json"
STATE_SCHEMA = 1
_write_lock = threading.Lock()


def _plain(value):
    """JSON-safe tree: Decimals become format(...) strings, everything else passes."""
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


class SelectionEntry(BaseModel):
    """An explicitly chosen 1C link for one receipt row."""

    name: StrictStr = Field(min_length=1)
    unit: StrictStr = Field(min_length=1)
    quantity: StrictStr
    service: StrictBool = False
    catalog_id: StrictStr | None = None
    code: StrictStr | None = None

    @model_validator(mode="after")
    def check_quantity(self):
        try:
            quantity = Decimal(self.quantity)
        except InvalidOperation as error:
            raise ValueError(f"Некорректное количество: {self.quantity!r}") from error
        if not quantity.is_finite() or quantity <= 0:
            raise ValueError(f"Количество должно быть положительным: {self.quantity!r}")
        return self


class SelectionPayload(BaseModel):
    selections: dict[StrictStr, SelectionEntry]


def state_path(job: Any) -> Path:
    return Path(job.directory) / STATE_FILE


def load_selections(job: Any) -> dict[str, dict]:
    path = state_path(job)
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(state, dict) or state.get("schema_version") != STATE_SCHEMA:
        return {}
    if state.get("receipt_fingerprint") != snapshot_from_receipt(job.result).fingerprint:
        return {}
    selections = state.get("selections")
    if not isinstance(selections, dict):
        return {}
    return {key: dict(value) for key, value in selections.items()
            if isinstance(value, dict)}


def save_selections(job: Any, selections: dict[str, dict]) -> dict[str, dict]:
    """Atomic, lock-guarded persistence of explicitly confirmed links."""
    validated = SelectionPayload(selections=selections)
    clean = {row_id: entry.model_dump() for row_id, entry in validated.selections.items()}
    with _write_lock, job.condition:
        state = {"schema_version": STATE_SCHEMA,
                 "saved_at": datetime.now(timezone.utc).isoformat(),
                 "receipt_fingerprint": snapshot_from_receipt(job.result).fingerprint,
                 "selections": clean}
        path = state_path(job)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    return clean


def _catalog_signature(directory=None, database_id=None) -> str:
    signature=catalog(directory, database_id=database_id)['signature']
    return hashlib.sha256(json.dumps(signature,ensure_ascii=False).encode('utf-8')).hexdigest()


def _issuer_signature(path=None) -> str:
    path = Path(path or generation.ISSUER_PATH)
    try:
        stat = path.stat()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return "missing"
    return f"{stat.st_mtime_ns}:{digest}"


def _revision(document: dict) -> str:
    """Receipt snapshot, selections, catalog files and issuer in one token pair."""
    payment = hashlib.sha256(b"\n".join([
        document["token"].encode("utf-8"),
        _catalog_signature().encode("utf-8"),
        _issuer_signature().encode("utf-8"),
    ])).hexdigest()
    return payment


def _blocked_reasons(source: dict, generation_result: dict, document: dict,
                     extra: list[str] | None = None) -> list[str]:
    reasons = []
    if source.get("validation_status") and source["validation_status"] != "match":
        reasons.append(f"Чек не сверен полностью ({source['validation_status']}); подтверждение заблокировано.")
    reasons.extend(generation_result["warnings"])
    reasons.extend(document["errors"])
    reasons.extend(extra or [])
    return list(dict.fromkeys(reasons))


def _row_candidates(row: dict) -> dict[str, Any]:
    article = row.get("torgsoft_article")
    resolved = resolve_source(row, article)
    result: dict[str, Any] = {"status": resolved["status"], "items": [],
                              "source_options": resolved.get("source_options", []),
                              "remembered": [], "errors": [], "query": row.get("name") or ""}
    if resolved["status"] == "resolved":
        found = proposals(row, article)
        result["items"] = [{**item, "origin": "proposal"} for item in found["items"]]
        result["errors"] = list(found.get("errors", []))
    try:
        net = Decimal(str(row.get("quantity") or 0)) * Decimal(str(row.get("price") or 0))
        remembered = generation.confirmed_candidates(row, net)
        result["remembered"] = [{**item, "origin": "remembered"} for item in remembered]
    except (ValueError, TypeError, InvalidOperation):
        pass
    from App.invoicing.catalog_1c import search_catalog
    existing = {item.get('id') or item.get('catalog_id') for item in result['items']}
    matches = search_catalog(row.get('name') or '', limit=10)
    if not matches['items'] and article:
        from App.invoicing.ai_matching import proposals as ai_proposals
        ai=ai_proposals(row.get('name') or '',catalog(),article=article,limit=10)
        matches['items']=ai['items']
        matches['errors']=matches['errors']+ai['errors']
    result['errors'].extend(matches['errors'])
    for item in matches['items']:
        if item['id'] not in existing:
            result['items'].append({**item, 'origin': 'ai_matching' if item.get('match_method')=='ai_matching' else 'search'})
            existing.add(item['id'])
    result['items'] = result['items'][:10]
    # Proposal/remembered contracts may omit stock metadata; use live identity only.
    from App.invoicing.catalog_1c import catalog_item
    for candidate in result['items'] + result['remembered']:
        card = catalog_item(candidate.get('catalog_id') or candidate.get('id'))
        if card:
            for field in ('uuid','stock_quantity','stock_amount','accounting_price','account_code','stock_as_of','snapshot_date','warehouse','warehouse_id','organization_id','database_id'):
                candidate[field]=card.get(field)
    return result


def _exact_amounts(source: dict, generation_result: dict) -> list[str]:
    """Check receipt amounts, with printed VAT allocated when included in prices."""
    CENT = Decimal("0.01")
    UNIT_PRICE_STEP = generation.UNIT_PRICE_STEP
    lines = {item["row_id"]: item for item in source["items"]}
    additions = {}
    if generation_result.get('vat_in_line_amounts'):
        net_total = sum((Decimal(str(item.get('amount') or 0)) for item in source['items']),Decimal(0))
        vat = Decimal(str(source.get('vat_amount') or 0))
        allocated = Decimal(0)
        for index,item in enumerate(source['items']):
            share=(vat*Decimal(str(item.get('amount') or 0))/net_total).quantize(CENT,rounding=ROUND_HALF_UP) if net_total else Decimal(0)
            if index==len(source['items'])-1:share=vat-allocated
            additions[item['row_id']]=share
            allocated+=share
    problems = []
    for row in generation_result["items"]:
        line = lines.get(row["row_id"])
        if line is None:
            continue
        raw = str(line.get("amount") or "").strip()
        try:
            exact = Decimal(raw) if raw else None
        except InvalidOperation:
            exact = None
        if exact is None:
            problems.append(f"Строка {row['row_id']}: в проверенном чеке нет суммы строки; счёт не может быть построен.")
            continue
        exact = (exact+additions.get(row['row_id'],Decimal(0))).quantize(CENT, rounding=ROUND_HALF_UP)
        try:
            current = Decimal(str(row["amount"]))
        except InvalidOperation:
            current = None
        if current is not None and abs(current - exact) <= CENT:
            continue
        problems.append(f"Строка {row['row_id']}: сумма счёта {current} не равна проверенной сумме чека {exact}.")
        row["amount"] = str(exact)
        quantity = Decimal(str(row["quantity"]))
        if quantity > 0:
            row["price"] = str((exact / quantity).quantize(UNIT_PRICE_STEP, rounding=ROUND_HALF_UP))
    return problems


def exchange_reasons(model: dict) -> list[str]:
    """Completeness for exchange, independently of mapping learning/finalizing."""
    reasons=[]
    if model.get('validation_status') != 'match':
        reasons.append('Числа чека не прошли контрольную сверку.')
    items=model.get('generation', {}).get('items', [])
    if not items:
        reasons.append('Нет завершённых строк счёта.')
    for row in items:
        if not row.get('name') or not row.get('unit'):
            reasons.append(f"Строка {row.get('row_id')}: выберите товар 1С и единицу.")
        for field in ('quantity','price','amount','source_amount'):
            try:
                number=Decimal(str(row.get(field)))
                valid=number.is_finite() and number>0
            except (InvalidOperation, ValueError, TypeError):
                valid=False
            if not valid:
                reasons.append(f"Строка {row.get('row_id')}: заполните {field}.")
    return list(dict.fromkeys(reasons))


@with_catalog_snapshot
def invoice_model(receipt: dict, selections: dict[str, dict] | None = None,
                  suggest: bool = True, database_id: str | None = None) -> dict:
    """Full model for GET/preview: generation, document, candidates, readiness."""
    snapshot = snapshot_from_receipt(receipt)
    source = to_generation_source(snapshot)
    vat_text = source["vat_amount"]
    for row in source["items"]:
        row["torgsoft_article"] = row.get("article") or None
        if row.get("torgsoft_article"):
            row["source_article"] = row["torgsoft_article"]
    clean = {}
    for row_id, entry in (selections or {}).items():
        entry = dict(entry)
        row = next((item for item in source["items"] if item["row_id"] == str(row_id)), None)
        if row is None:
            continue
        entry.pop("source_fingerprint", None)
        entry["source_fingerprint"] = generation.source_fingerprint(row)
        if entry.get("catalog_id"):
            entry.setdefault("code", None)
        clean[str(row_id)] = entry
    generation_result = generation.build_generation(source,
                                                    selections=clean or None,
                                                    suggest=suggest, database_id=database_id, vat_in_line_amounts=vat_text is not None,
                                                    preserve_receipt_quantities=True)
    amount_problems = _exact_amounts(source, generation_result)
    rounding_notes=[]
    if vat_text is not None:
        generation_result['price_decimal_places']=2
        for row in generation_result['items']:
            price=Decimal(str(row['price'])).quantize(Decimal('.01'),rounding=ROUND_HALF_UP)
            row['price']=format(price,'.2f')
            difference=(Decimal(str(row['quantity']))*price-Decimal(str(row['amount']))).quantize(Decimal('.01'),rounding=ROUND_HALF_UP)
            row['price_rounding_difference']=str(difference)
            if difference:
                rounding_notes.append(f"Строка {row['row_id']}: округление цены до 2 знаков даёт разницу {difference} ₸ при умножении на количество; сумма строки и итог сохранены из чека с НДС.")
    if vat_text:
        # Keep the printed receipt tax; never take a tax amount from a catalog.
        generation_result["vat_amount"] = vat_text
    document = generation.invoice_document(source, generation_result)
    candidates = {row["row_id"]: _row_candidates(row) for row in source["items"]}
    blocked = _blocked_reasons(source, generation_result, document, amount_problems)
    revision = _revision(document)
    stored = {}
    for row_id, entry in clean.items():
        portable = {key: value for key, value in entry.items() if key != "source_fingerprint"}
        stored[row_id] = portable
    model = {
        "schema_version": 1,
        "job_id": snapshot.job_id,
        "receipt_revision": snapshot.revision,
        "finalized": bool(source.get("finalized")),
        "validation_status": source.get("validation_status"),
        "generation": generation_result,
        "document": document,
        "candidates": _plain(candidates),
        "search_errors": catalog()["errors"],
        "blocked": blocked,
        "ready": not blocked,
        "can_render": True,
        "revision": revision,
        "token": document["token"],
        "selections": stored,
        "rounding_notes": rounding_notes,
    }
    model['exchange_reasons'] = exchange_reasons(model)
    model['exchange_ready'] = not model['exchange_reasons']
    return model


@with_catalog_snapshot
def train_from_selections(receipt: dict, selections: dict[str, dict], *,
                          database_id: str | None = None,
                          knowledge_path: Path | None = None,
                          training_dir: Path | None = None) -> dict:
    """Explicit operator action: save a training pair and remember the links.

    Requires a fully ready invoice (finalized receipt, validation match, every
    row linked). The pair is written to the donor package layout so
    `learned_mapping.mapping_rules()` learns from it; card knowledge is updated
    through `remember_selections` (with alternatives and backup). The real case
    store/knowledge are only touched when the caller does not override paths.
    """
    from App.invoicing import training_pairs
    model = invoice_model(receipt, selections=selections, database_id=database_id)
    if not receipt.get("finalized"):
        raise ValueError("Для обучения подтвердите исходный электронный чек.")
    if not model["ready"]:
        raise ValueError("Обучение возможно только для полностью готового счёта: "
                         + "; ".join(model["blocked"]))
    pair, memory_pairs = training_pairs.screen_pair(model, receipt, database_id=database_id)
    if not memory_pairs:
        raise ValueError("Нет подтверждённых связей для обучения.")
    case = training_pairs.save_screen_pair(pair, base_dir=training_dir)
    if knowledge_path is not None:
        generation.remember_selections(memory_pairs, path=Path(knowledge_path),
                                       database_id=database_id)
    else:
        generation.remember_selections(memory_pairs, database_id=database_id)
    summary = training_pairs.rules_summary(training_dir, database_id)
    return {"case": case, "remembered": len(memory_pairs),
            "cases_total": _count_cases(training_dir), **summary}


def _count_cases(training_dir: Path | None) -> int:
    base = Path(training_dir) if training_dir is not None else training_pairs_dir()
    return len(list(base.glob("case_*/expected.json")))


def training_pairs_dir() -> Path:
    from App.invoicing.training_pairs import DATA_TRAINING_DIR
    return DATA_TRAINING_DIR


@with_catalog_snapshot
def search_catalog_1c(query: str, limit: int = 20, *, database_id=None) -> dict:
    from App.invoicing.catalog_1c import search_catalog
    result = search_catalog(query or "", limit=limit)
    return {"items": _plain(result["items"]), "errors": result["errors"]}


def render_invoice(model: dict, revision: str | None, token: str | None) -> bytes:
    """PDF of the current model; stale revision/token pairs are rejected."""
    from App.invoicing.invoice_pdf import render_invoice_pdf
    if not revision or not token:
        raise RuntimeError("Счёт выдаётся только с актуальной ревизией и меткой документа")
    if revision != model["revision"] or token != model["token"]:
        raise RuntimeError("Счёт изменился; получите новую ревизию и метку документа")
    document = dict(model["document"], errors=model["blocked"])
    return render_invoice_pdf(document, allow_draft=True)
