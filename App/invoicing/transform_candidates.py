"""Read-only, cached Torgsoft→1C proposals. Import loads nothing confirmed.

Adapted from the donor `transform_candidates.py`: same pydantic schema
(proposals file version 1.0), the same read-only lock and caches, imports
pointed at `App.invoicing.*`. Selections stored in this file are NEVER treated
as confirmed mappings; confirmed choices live in the mapping layer (Etap 3).
"""
from __future__ import annotations

import json
import os
import threading
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, StrictInt, StrictStr, ValidationError, field_validator, model_validator
from rapidfuzz import fuzz, process

from App.invoicing.catalog_1c import catalog
from App.invoicing.receipt_names import torgsoft_name
from App.invoicing.user_mapping import normalize_mapping_key

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "Data" / "reference" / "User_Mapping" / "torgsoft_1c_transform_top10.json"
_lock = threading.RLock()


class SourceCard(BaseModel):
    article: StrictStr | None
    name: StrictStr = Field(min_length=1, pattern=r"\S")
    sources: list[StrictStr]


class Candidate(BaseModel):
    rank: StrictInt = Field(ge=1, le=10)
    code_1c: StrictStr = Field(min_length=1, pattern=r"\S")
    name_1c: StrictStr = Field(min_length=1, pattern=r"\S")
    source_sheet: StrictStr
    unit_1c: StrictStr | None
    stock_snapshot: Decimal | None = Field(allow_inf_nan=False)
    stock_date: StrictStr | None
    review_notes: list[StrictStr]

    @field_validator("stock_date")
    @classmethod
    def date_format(cls, value):
        if value:
            datetime.strptime(value, "%d.%m.%Y")
        return value


class InitialChoice(BaseModel):
    selected_code_1c: None
    approved_codes_1c: list[StrictStr] = Field(max_length=0)
    mapping_status: Literal["not_confirmed"]
    quantity_coefficient: None


class ProposalRow(BaseModel):
    row_id: StrictStr = Field(min_length=1)
    torgsoft: SourceCard
    search_result: StrictStr
    candidates_1c: list[Candidate] = Field(max_length=10)
    user_choice: InitialChoice

    @model_validator(mode="after")
    def unique_candidates(self):
        for field in ("rank", "code_1c"):
            values = [getattr(row, field) for row in self.candidates_1c]
            if len(set(values)) != len(values):
                raise ValueError(f"Повтор кандидата: {field}")
        return self


@lru_cache(maxsize=2)
def _load(path_text, modified, size):
    path = Path(path_text)
    result = {"rows": [], "by_article": defaultdict(list), "by_name": defaultdict(list),
              "errors": [], "invalid_records": 0, "file": path.name}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if (not isinstance(data, dict) or data.get("schema_version") != "1.0" or
                data.get("candidate_limit") != 10 or data.get("confirmed_mapping") is not False or
                not isinstance(data.get("rows"), list) or
                any(not isinstance(data.get(key), list) or any(not isinstance(value, str) for value in data[key])
                    for key in ("sources", "notes"))):
            raise ValueError("Неверный формат каталога предложений 1.0")
        seen = set()
        duplicates = set()
        for index, raw in enumerate(data["rows"], 1):
            try:
                row = ProposalRow.model_validate(raw).model_dump(mode="json")
                if row["row_id"] in seen:
                    duplicates.add(row["row_id"])
                    raise ValueError("Повтор row_id")
                seen.add(row["row_id"])
                row["candidates_1c"].sort(key=lambda candidate: candidate["rank"])
                result["rows"].append(row)
            except (ValidationError, ValueError) as error:
                result["invalid_records"] += 1
                message = "; ".join(".".join(map(str, e["loc"])) + ": " + e["msg"]
                                    for e in error.errors(include_input=False)) if isinstance(error, ValidationError) else str(error)
                if len(result["errors"]) < 30:
                    result["errors"].append(f"Строка {index}: {message}")
        result["rows"] = [row for row in result["rows"] if row["row_id"] not in duplicates]
        for row in result["rows"]:
            source = row["torgsoft"]
            if source["article"]:
                result["by_article"][source["article"]].append(row)
            result["by_name"][normalize_mapping_key(source["name"])].append(row)
    except (OSError, ValueError, TypeError) as error:
        result["errors"].append(f"{path.name}: {error}")
    result["names"] = tuple(result["by_name"])
    return result


def candidate_index(path=None):
    path = Path(path or os.environ.get("TORGSOFT_TRANSFORM_CANDIDATES", DEFAULT_PATH)).resolve()
    with _lock:
        try:
            stat = path.stat()
            signature = stat.st_mtime_ns, stat.st_size
        except OSError:
            signature = 0, 0
        return _load(str(path), *signature)


def source_identity(article):
    return "torgsoft:article:" + article


def source_view(row):
    return {**row["torgsoft"], "proposal_row_id": row["row_id"],
            "identity": source_identity(row["torgsoft"]["article"]) if row["torgsoft"]["article"] else None}


def resolve_source(source, article=None, index=None):
    index = index if index is not None else candidate_index()
    explicit = article if article is not None else source.get("torgsoft_article")
    variants = (index["by_article"].get(explicit, []) if explicit is not None else
                index["by_name"].get(normalize_mapping_key(torgsoft_name(source) or ""), []))
    if len(variants) == 1 and variants[0]["torgsoft"]["article"]:
        row = variants[0]
        # An article collision is never repaired by picking the first matching name.
        if len(index["by_article"][row["torgsoft"]["article"]]) == 1:
            return {"status": "resolved", "source": source_view(row), "row": row}
    return {"status": "source_required", "source": None,
            "source_options": [source_view(row) for row in variants[:10]]}


def search_sources(query, limit=10):
    index = candidate_index()
    query = query.strip()
    if query in index["by_article"]:
        rows = index["by_article"][query]
    elif len(query) >= 3:
        matches = process.extract(normalize_mapping_key(query), index["names"], scorer=fuzz.WRatio,
                                  limit=limit, score_cutoff=55)
        rows = [row for key, _, _ in matches for row in index["by_name"][key]]
    else:
        rows = []
    return {"items": [source_view(row) for row in rows[:limit]], "errors": index["errors"]}


def proposals(source, article=None):
    index = candidate_index()
    resolved = resolve_source(source, article, index)
    result = {key: value for key, value in resolved.items() if key != "row"}
    result.update(items=[], errors=list(index["errors"]), invalid_records=index["invalid_records"])
    if resolved["status"] != "resolved":
        return result
    current_catalog = catalog()
    targets = current_catalog["by_code"]
    result['errors'].extend(current_catalog.get('errors', []))
    for candidate in resolved["row"]["candidates_1c"]:
        variants = targets.get(normalize_mapping_key(candidate["code_1c"]), [])
        if len(variants) != 1:
            result["errors"].append(f"Код 1С {candidate['code_1c']}: карточка отсутствует или неоднозначна")
            continue
        product = variants[0]
        notes = list(candidate["review_notes"])
        if product["name"] != candidate["name_1c"] or (candidate["unit_1c"] and product["unit"] != candidate["unit_1c"]):
            notes.append("Карточка изменилась после подготовки предложений: проверьте название и единицу")
        result["items"].append({**product, "rank": candidate["rank"],
                                "stock_snapshot": product.get('stock_quantity') if current_catalog.get('stock_source')=='json' else candidate['stock_snapshot'],
                                "stock_date": product.get('stock_as_of') if current_catalog.get('stock_source')=='json' else candidate['stock_date'],
                                "review_notes": notes, "proposal_name": candidate["name_1c"]})
    return result


def import_status():
    index = candidate_index()
    return {"file": index["file"], "rows": len(index["rows"]),
            "with_candidates": sum(bool(row["candidates_1c"]) for row in index["rows"]),
            "invalid_records": index["invalid_records"], "errors": index["errors"]}
