"""Screen-mapping training pairs built from the №16 verified receipt (Etap 6).

Adapted from the donor `training_workflow.py` screen-mapping branch: a pair is
the current verified electronic receipt (predzakaz + canonical training
receipt) and the generated 1C invoice with operator-confirmed links. Cases are
stored in the same donor package layout
(`Data/reference/Training_Pair/case_*/expected.json`, schema 2), so the
existing `learned_mapping.mapping_rules()` picks them up without changes.

No PDF import or inbox queue yet: №16 invoices live as generated documents;
«обучение» is an explicit operator action and never runs automatically.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import threading
from datetime import datetime
from pathlib import Path

from App.invoicing import invoice_generation as generation
from App.invoicing.catalog_1c import catalog_named_item
from App.invoicing.learned_mapping import prepare_training_pair
from App.invoicing.receipt_adapter import snapshot_from_receipt, to_generation_source
from App.invoicing.transform_candidates import resolve_source
from App.invoicing.user_mapping import normalize_mapping_key

ROOT = Path(__file__).resolve().parents[2]
DATA_TRAINING_DIR = ROOT / "Data" / "reference" / "Training_Pair"
training_lock = threading.RLock()

SOURCE_KEYS = ("row_id", "name", "ocr_name", "edited", "name_confirmed", "quantity", "price")


def raw_predzakaz_items(receipt: dict) -> list[dict]:
    items = []
    for row in receipt.get("rows", []):
        if row.get("kind") not in ("product", "service"):
            continue
        items.append({
            "row_id": str(row.get("row")),
            "name": row.get("name") or "",
            "ocr_name": row.get("name_raw") or row.get("name") or "",
            "edited": bool(row.get("name_confirmed") or row.get("name_origin") == "catalog"),
            "quantity": row.get("quantity"),
            "price": row.get("price"),
            "name_confidence": row.get("match"),
            "reference_name": row.get("name") or "",
        })
    return items


def _invoice_items(generation_result: dict) -> list[dict]:
    items = []
    for row in generation_result["items"]:
        items.append({
            "row_id": str(row["row_id"]),
            "name": row.get("name"),
            "unit": row.get("unit") or "",
            "quantity": row.get("quantity"),
            "price": row.get("price"),
            "amount": row.get("amount"),
            "service": bool(row.get("service")),
        })
    return items


def screen_pair(model: dict, receipt: dict, *, database_id: str | None = None,
                training_basis: str = "screen_mapping") -> tuple[dict, list[tuple[dict, dict]]]:
    """Build and validate a training pair from the current invoice model.

    Returns (pair, memory_pairs): memory pairs are (source_row, selection)
    tuples enriched with card identity fields, ready for
    `invoice_generation.remember_selections`. Raises ValueError with visible
    notes when the pair must not be learned.
    """
    database_id = database_id or generation.database_context()
    snapshot = snapshot_from_receipt(receipt)
    source = to_generation_source(snapshot)
    for row in source["items"]:
        row["torgsoft_article"] = row.get("article") or None
        if row.get("torgsoft_article"):
            row["source_article"] = row["torgsoft_article"]
    generation_result = model["generation"]
    invoice_rows = {str(row["row_id"]): row for row in generation_result["items"]}
    selections = {str(row_id): dict(entry) for row_id, entry in (model.get("selections") or {}).items()}
    raw_items = raw_predzakaz_items(receipt)
    links = []
    memory_pairs = []
    for index, row in enumerate(source["items"]):
        target = invoice_rows.get(str(row["row_id"]))
        if target is None:
            raise ValueError(f"Строка {row['row_id']} не попала в счёт; обучение невозможно.")
        link = {"predzakaz_row_id": row["row_id"], "source_index": index,
                "invoice_row_id": target["row_id"]}
        resolved = resolve_source(row, row.get("torgsoft_article"))
        if resolved["status"] == "resolved":
            card = resolved["source"]
            link.update(source_identity=card["identity"], source_article=card["article"], source_card=card)
        selection = selections.get(str(row["row_id"]))
        if selection:
            product = catalog_item_selection(selection)
            if product:
                link.update(catalog_id=selection.get("catalog_id"), code=selection.get("code"),
                            service=bool(selection.get("service")))
            memory = dict(selection)
            for key in ("source_identity", "source_article", "source_card"):
                if key in link:
                    memory[key] = link[key]
            memory_pairs.append((dict(row), memory))
        links.append(link)
    invoice = {"items": _invoice_items(generation_result), "total": generation_result["total"],
               "vat_amount": None}
    training_receipt = {"items": [{key: row.get(key) for key in SOURCE_KEYS} for row in source["items"]],
                        "total": source["total"], "vat_amount": None, "warnings": []}
    pair = {
        "predzakaz": {"items": raw_items, "total": source["total"], "vat_amount": None},
        "training_receipt": training_receipt,
        "invoice": invoice,
        "links": links,
    }
    prepare_training_pair(pair)
    if not pair.get("training_ready"):
        raise ValueError("; ".join(pair.get("training_notes") or ["Пара не готова к обучению"]))
    pair.update(schema_version=2, status="learned", database_id=database_id,
                mapping_policy="enrich_confirmed_alternatives", training_basis=training_basis,
                screen_revision=snapshot.revision,
                confirmation={"ocr_checked": bool(snapshot.finalized), "invoice_checked": True,
                              "links_checked": True})
    return pair, memory_pairs


def catalog_item_selection(selection: dict):
    """Resolve the catalog card behind an operator selection (or None)."""
    catalog_id = selection.get("catalog_id")
    if catalog_id:
        return generation.catalog_item(catalog_id)
    name, unit = selection.get("name") or "", selection.get("unit") or ""
    return generation.catalog_named_item(name, unit) if name else None


def save_screen_pair(pair: dict, base_dir: Path | None = None) -> dict:
    """Write the case file atomically; same-job retraining updates in place."""
    base_dir = Path(base_dir) if base_dir is not None else DATA_TRAINING_DIR
    case_id = _case_identity_from_pair(pair)
    case_dir = (base_dir / case_id).resolve()
    if case_dir.parent.resolve() != base_dir.resolve():
        raise ValueError("Некорректный каталог учебной пары")
    expected_path = case_dir / "expected.json"
    now = datetime.now().isoformat(timespec="microseconds")
    previous = {}
    if expected_path.exists():
        try:
            previous = json.loads(expected_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(f"Существующий кейс повреждён: {case_id}") from error
    pair = dict(pair)
    pair["case_id"] = case_id
    pair["created_at"] = previous.get("created_at", now)
    pair["updated_at"] = now
    mapping_fields = ("predzakaz", "training_receipt", "invoice", "links")
    history = copy.deepcopy(previous.get("mapping_history", []))
    current = {key: copy.deepcopy(pair.get(key)) for key in mapping_fields}
    if previous.get("training_ready"):
        previous_snapshot = {key: copy.deepcopy(previous.get(key)) for key in mapping_fields}
        snapshot = copy.deepcopy(previous_snapshot)
        tags = dict(case_id=case_id, training_ready=True,
                    created_at=previous.get("created_at", now))
        snapshot.pop("training_ready", None)
        if any(previous_snapshot.get(key) != value for key, value in current.items()):
            snapshot.update(tags)
            snapshot["training_ready"] = True
            if snapshot not in history:
                history.append(snapshot)
    pair["mapping_history"] = history
    pair["sources"] = {}
    case_dir.mkdir(parents=True, exist_ok=True)
    temporary = expected_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(pair, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, expected_path)
    return {"case_id": case_id, "path": str(expected_path), "created": not previous,
            "history_count": len(history)}


def _case_identity_from_pair(pair: dict) -> str:
    links_payload = json.dumps([link for link in pair["links"]],
                               sort_keys=True, ensure_ascii=False, default=str)
    identity = normalize_mapping_key(json.dumps(pair.get("training_receipt", {}).get("items", []),
                                               ensure_ascii=False, default=str))
    digest = hashlib.sha256((links_payload + ":" + identity).encode("utf-8")).hexdigest()[:12]
    return "case_" + digest


def rules_summary(training_dir: Path | None = None, database_id: str | None = None) -> dict:
    from App.invoicing.learned_mapping import mapping_rules
    rules = mapping_rules(Path(training_dir) if training_dir else None, database_id)
    conflicts = [key for key, rule in rules.items() if rule.get("conflict")]
    return {"rules": len(rules) - len(conflicts), "conflicts": len(conflicts)}


def archive_case_photo(case_dir: Path, photo: Path, name: str = "predzakaz.jpg") -> str:
    """Keep the receipt photo next to its pair for later PDF-import work."""
    target = case_dir / name
    if photo.is_file():
        shutil.copy2(photo, target)
        return target.name
    return ""
