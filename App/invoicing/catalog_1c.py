"""Read-only index of all supplied 1C catalog workbooks into App/invoicing.

Adapted from the donor `catalog_1c.py` with the №16 root layout and the
extended inventory contract: cards keep their 1C codes, units and services,
and, when the workbook provides them, end-stock quantity/amount, accounting
price, accounting account, warehouse and snapshot date — stored separately and
never used as the sale price or as evidence of product identity.
"""
from __future__ import annotations

import csv
import hashlib
import io
import re
from decimal import Decimal, InvalidOperation
from functools import lru_cache, wraps
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from rapidfuzz import fuzz, process

from App.invoicing.user_mapping import normalize_mapping_key

CATALOG_DIR = Path(__file__).resolve().parents[2] / "Data" / "reference" / "1c_ref"

STOCK_TOKENS = {"конечный остаток"}
STOCK_QUANTITY_TOKENS = {"количество", "кол-во"}
ACCOUNTING_PRICE_TOKENS = {"учетная цена", "учётная цена"}
STOCK_AMOUNT_TOKENS = {"сумма"}
ACCOUNT_CODE_TOKENS = {"счет учета", "счёт учета", "счет учёта", "счёт учёта"}


def _text(value):
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _decimal(value):
    text = _text(value)
    if not text:
        return None
    try:
        result = Decimal(text.replace(",", ".").replace(" ", ""))
    except InvalidOperation:
        return None
    return result if result.is_finite() else None


def _name_id(name, unit):
    return hashlib.sha256((name + "\0" + unit).encode("utf-8")).hexdigest()[:24]


def _name_key(name, unit):
    return normalize_mapping_key(name), normalize_mapping_key(unit)


def _file_meta(path: Path) -> dict:
    """Warehouse and snapshot date live in the workbook file name and only there."""
    date_match = re.search(r"на\s+(\d{2}\.\d{2}\.\d{4})", path.stem)
    warehouse = path.stem.split(" на ")[0].strip() if date_match else None
    return {
        "warehouse": warehouse or None,
        "snapshot_date": date_match.group(1) if date_match else None,
    }


def workbook_rows(path):
    if path.suffix.lower() == ".xls":
        import xlrd
        book = xlrd.open_workbook(path, on_demand=True)
        try:
            for sheet in book.sheets():
                yield sheet.name, (sheet.row_values(i) for i in range(sheet.nrows))
        finally:
            book.release_resources()
    elif path.suffix.lower() == ".xlsx":
        from openpyxl import load_workbook
        book = load_workbook(path, read_only=True, data_only=True)
        try:
            for sheet in book:
                yield sheet.title, sheet.iter_rows(values_only=True)
        finally:
            book.close()
    else:
        data = path.read_bytes()
        for encoding in ("utf-8-sig", "cp1251"):
            try:
                content = data.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("Неизвестная кодировка CSV")
        dialect = csv.Sniffer().sniff(content[:8192], delimiters=";,\t")
        yield path.stem, csv.reader(io.StringIO(content), dialect)


def _find_column(keys, tokens):
    for index, key in enumerate(keys):
        if key in tokens:
            return index
    return None


def _sub_header_columns(values, keys):
    """Detect the second header row (Учетная цена | Количество | Сумма)."""
    price = _find_column(keys, ACCOUNTING_PRICE_TOKENS)
    quantity = _find_column(keys, STOCK_QUANTITY_TOKENS)
    amount = _find_column(keys, STOCK_AMOUNT_TOKENS)
    if price is None and quantity is None:
        return None
    return {"accounting_price": price, "stock_quantity": quantity, "stock_amount": amount}


@lru_cache(maxsize=2)
def _load(signature):
    entries, errors = {}, []
    for filename, _, _ in signature:
        path = Path(filename)
        meta = _file_meta(path)
        try:
            for sheet, rows in workbook_rows(path):
                name_col = unit_col = code_col = account_col = None
                columns = {"accounting_price": None, "stock_quantity": None, "stock_amount": None}
                service_sheet = normalize_mapping_key(sheet) in {"услуги", "services"}
                for row in rows:
                    values = [_text(value) for value in row]
                    keys = [normalize_mapping_key(value) for value in values]
                    header = next((i for i, key in enumerate(keys)
                                   if key in {"наименование", "номенклатура", "наименование номенклатуры", "name"}), None)
                    if header is not None:
                        name_col = header
                        unit_col = next((i for i, key in enumerate(keys)
                                         if key in {"ед изм", "ед", "единица измерения", "unit"}), None)
                        code_col = next((i for i, key in enumerate(keys)
                                         if key in {"код артикул", "код", "артикул", "code", "article"}), None)
                        account_col = _find_column(keys, ACCOUNT_CODE_TOKENS)
                        columns = {"accounting_price": _find_column(keys, ACCOUNTING_PRICE_TOKENS),
                                   "stock_quantity": _find_column(keys, STOCK_QUANTITY_TOKENS | STOCK_TOKENS),
                                   "stock_amount": _find_column(keys, STOCK_AMOUNT_TOKENS)}
                        continue
                    if name_col is None:
                        # Some service sheets are plain name lists without column headers.
                        populated = [value for value in values if value]
                        if not service_sheet or len(populated) != 1:
                            continue
                        name = populated[0]
                        key = normalize_mapping_key(name)
                        if key in {"услуги", "services"} or key.startswith(("услуги ", "services ")):
                            continue
                    elif len(values) <= name_col:
                        continue
                    else:
                        sub = _sub_header_columns(values, keys)
                        if sub is not None:
                            # A second header row: Учетная цена | Количество | Сумма.
                            for key, index in sub.items():
                                if index is not None:
                                    columns[key] = index
                            continue
                        name = values[name_col]
                    if not name or normalize_mapping_key(name) in {"итого", "всего"}:
                        continue
                    unit = values[unit_col] if unit_col is not None and len(values) > unit_col else ""
                    if service_sheet and not unit:
                        unit = "усл"
                    code = values[code_col] if code_col is not None and len(values) > code_col else ""
                    identity = "code:" + hashlib.sha256(code.encode("utf-8")).hexdigest()[:24] if code else _name_id(name, unit)
                    # Different codes must remain separate even when their names are identical.
                    service = service_sheet or unit.rstrip(".") == "усл"
                    record_key = identity, name, unit, service
                    entries.setdefault(record_key, {
                        "id": identity, "name": name, "unit": unit,
                        "code": code or None, "uuid": None,
                        "service": service,
                        "source": path.name, "sheet": sheet,
                        "stock_quantity": _decimal(values[columns["stock_quantity"]]) if columns["stock_quantity"] is not None else None,
                        "stock_amount": _decimal(values[columns["stock_amount"]]) if columns["stock_amount"] is not None else None,
                        "accounting_price": _decimal(values[columns["accounting_price"]]) if columns["accounting_price"] is not None else None,
                        "account_code": (values[account_col] if account_col is not None and len(values) > account_col else "") or None,
                        "warehouse": meta["warehouse"],
                        "snapshot_date": meta["snapshot_date"],
                    })
        except Exception as error:
            errors.append(f"{path.name}: {error}")
    coded_names = {_name_key(row["name"], row["unit"]) for row in entries.values() if row["code"]}
    items = [row for row in entries.values() if row["code"] or _name_key(row["name"], row["unit"]) not in coded_names]
    return _index(items, errors, signature)


def _index(items, errors, signature):
    by_name, by_id, by_code = {}, {}, {}
    for row in items:
        by_name.setdefault(_name_key(row["name"], row["unit"]), []).append(row)
        by_id.setdefault(row["id"], []).append(row)
        if row.get('uuid'):
            by_id.setdefault(row['uuid'].lower(), []).append(row)
            by_id.setdefault('uuid:'+row['uuid'].lower(), []).append(row)
        if row["code"]:
            by_code.setdefault(normalize_mapping_key(row["code"]), []).append(row)
    for variants in by_name.values():
        # Preserve old saved name-based IDs only when there is a single target.
        if len(variants) == 1:
            row = variants[0]
            by_id.setdefault(_name_id(row["name"], row["unit"]), variants)
    for identity, variants in by_id.items():
        if len(variants) > 1:
            errors.append(f"Неоднозначный идентификатор 1С: {variants[0].get('code') or identity}")
    return {"items": items, "keys": [normalize_mapping_key(row["name"]) for row in items],
            "by_id": by_id, "by_name": by_name, "by_code": by_code,
            "errors": errors, "files": len(signature), "signature": signature}


def invoice_catalog():
    from App.invoicing.learned_mapping import mapping_rules
    entries = {}
    for rule in mapping_rules().values():
        if rule.get("conflict"):
            continue
        identity = "invoice:" + hashlib.sha256((rule["name"] + "\0" + rule["unit"]).encode("utf-8")).hexdigest()[:24]
        entries.setdefault(identity, {"id": identity, "name": rule["name"], "unit": rule["unit"],
                                      "service": rule["unit"].rstrip(".") == "усл",
                                      "source": "Подтверждённые учебные счета 1С", "sheet": "",
                                      "stock_quantity": None, "stock_amount": None,
                                      "accounting_price": None, "account_code": None,
                                      "warehouse": None, "snapshot_date": None,
                                      "case_id": rule["case_id"]})
    return list(entries.values())


_DATABASE = ContextVar('catalog_database', default=None)
_PINNED = ContextVar('catalog_snapshot', default=None)


def catalog_signature(directory=None, database_id=None):
    return catalog(directory, database_id=database_id)['signature']


@lru_cache(maxsize=8)
def _load_current(signature):
    from App.invoicing.stock_sync import read, StockSyncError
    path=Path(signature[0][0])
    try:
        data=read(path)
    except StockSyncError as error:
        result=_index([], [str(error)], signature)
        result['stock_source']='json_error'
        return result
    items=[]
    for item in data['items']:
        row=dict(item, id='code:'+hashlib.sha256(item['code'].encode('utf-8')).hexdigest()[:24],
                 source=path.name, sheet='Услуги' if item['service'] else 'Товары',
                 warehouse=data['warehouse']['name'], warehouse_id=data['warehouse']['id'],
                 organization_id=data['organization']['id'], database_id=data['database_id'],
                 snapshot_date=data['stock_as_of'], stock_as_of=data['stock_as_of'])
        for field in ('stock_quantity','stock_amount','accounting_price'):
            row[field]=Decimal(item[field]) if item[field] is not None else None
        items.append(row)
    result=_index(items, [], signature)
    result.update(stock_source='json', stock_as_of=data['stock_as_of'], database_id=data['database_id'],
                  snapshot_id=data['snapshot_id'])
    return result


def catalog(directory=None, *, database_id=None):
    from App.invoicing.stock_sync import select_path, StockSyncError
    if (directory is None and _PINNED.get() is not None
            and (database_id is None or database_id==_DATABASE.get())):
        return _PINNED.get()
    directory=Path(directory or CATALOG_DIR)
    database_id=database_id if database_id is not None else _DATABASE.get()
    try:
        current=select_path(directory, database_id)
    except StockSyncError as error:
        result=_index([], [str(error)], ())
        result['stock_source']='json_error'
        return result
    if current:
        stat=current.stat()
        result = _load_current(((str(current.resolve()),stat.st_mtime_ns,stat.st_size),))
        if database_id and result.get('database_id') and result['database_id'].strip().casefold()!=database_id.strip().casefold():
            failed=_index([],['JSON относится к другой базе 1С.'],result['signature'])
            failed['stock_source']='json_error'
            return failed
        return result
    signature=tuple((str(p.resolve()),p.stat().st_mtime_ns,p.stat().st_size)
                    for p in sorted(directory.rglob('*'))
                    if p.is_file() and p.suffix.lower() in {'.xls','.xlsx','.csv'} and not p.name.startswith('~$'))
    return _load(signature)


@contextmanager
def catalog_scope(database_id=None):
    db_token=_DATABASE.set(database_id if database_id is not None else _DATABASE.get())
    try:
        snapshot=catalog(database_id=_DATABASE.get())
        pin_token=_PINNED.set(snapshot)
        try:
            yield snapshot
        finally:
            _PINNED.reset(pin_token)
    finally:
        _DATABASE.reset(db_token)


def with_catalog_snapshot(function):
    @wraps(function)
    def wrapped(*args,**kwargs):
        with catalog_scope(kwargs.get('database_id')):
            return function(*args,**kwargs)
    return wrapped


def invalidate_stock_cache(data=None):
    _load_current.cache_clear()
    _suggest.cache_clear()
    # Load the newly committed catalog before acknowledging the upload.
    if data is not None:
        current=catalog(database_id=data['database_id'])
        if current.get('stock_source')!='json' or len(current['items'])!=data['record_count']:
            from App.invoicing.stock_sync import StockSyncError
            raise StockSyncError('catalog_reload_failed','Не удалось прочитать опубликованный каталог.',500)


def default_search_catalog(query, limit=30, directory=None, include_invoices=False):
    index = catalog(directory)
    if index.get('stock_source')=='json_error':
        return dict(items=[],total=0,files=index['files'],errors=index['errors'])
    items = index["items"]
    choices = index["keys"]
    if include_invoices:
        keys = {(normalize_mapping_key(row["name"]), row["unit"].rstrip(".")) for row in items}
        extra = [row for row in invoice_catalog() if (normalize_mapping_key(row["name"]), row["unit"].rstrip(".")) not in keys]
        items = items + extra
        choices = choices + [normalize_mapping_key(row["name"]) for row in extra]
    raw_identity=query.strip().lower()
    query = normalize_mapping_key(query)
    identity_matches = index['by_id'].get(raw_identity, [])
    if identity_matches:
        return dict(items=[{**row,'score':100.0} for row in identity_matches][:limit],total=len(items),files=index['files'],errors=index['errors'])
    if query in index["by_code"]:
        return {"items": [{**row, "score": 100.0} for row in index["by_code"][query]][:limit],
                "total": len(items), "files": index["files"], "errors": index["errors"]}
    if query:
        words = [word for word in query.split() if len(word) >= 3 and word.isalpha()
                 and word not in {"для", "при", "под", "над", "без", "или"}]
        digits = set(re.findall(r"\d+", query))
        matches = process.extract(query, choices, scorer=fuzz.WRatio, limit=None, score_cutoff=35)
        ranked = []
        for key, score, position in matches:
            candidate_words = key.split()
            common = [any(word == candidate or candidate.startswith(word) or
                          (len(word) >= 4 and fuzz.ratio(word, candidate) >= 85)
                          for candidate in candidate_words) for word in words]
            # Numeric fragments alone must not suggest a different product category.
            if words and not any(common):
                continue
            weights = [3] + [1] * (len(words) - 1) if words else []
            coverage = sum(weight for weight, present in zip(weights, common) if present) / sum(weights) if weights else 0
            numeric = len(digits & set(re.findall(r"\d+", key))) / len(digits) if digits else 1
            rank = .65 * coverage + .25 * score / 100 + .10 * numeric
            ranked.append((rank, score, position))
        ranked.sort(key=lambda match: (-match[0], -match[1], items[match[2]]["name"]))
        selected = [{**items[position], "score": round(score, 1)} for _, score, position in ranked[:limit]]
    else:
        selected = items[:limit]
    return {"items": selected, "total": len(items), "files": index["files"], "errors": index["errors"]}


def search_catalog(query, limit=30, directory=None, include_invoices=False):
    """Matcher first, original search unchanged when the matcher finds nothing."""
    from App.invoicing.product_matcher import rank_products
    index = catalog(directory)
    if index.get('stock_source')=='json_error':
        return dict(items=[],total=0,files=index['files'],errors=index['errors'],match_method='catalog_error')
    items = index['items']
    if include_invoices:
        keys = {(normalize_mapping_key(row['name']), row['unit'].rstrip('.')) for row in items}
        items = items + [row for row in invoice_catalog()
                         if (normalize_mapping_key(row['name']), row['unit'].rstrip('.')) not in keys]
    exact=index['by_id'].get(query.strip().lower(), []) or index['by_code'].get(normalize_mapping_key(query), [])
    if exact:
        return dict(items=[{**row,'score':100.0,'match_method':'catalog_identity'} for row in exact][:limit],total=len(items),files=index['files'],errors=index['errors'],match_method='catalog_identity')
    matches = rank_products(query, items, limit=limit) if query.strip() else []
    if matches:
        return dict(items=[{**row, 'score': row['match_score'], 'match_method': 'product_matcher_v1'}
                           for row in matches], total=len(items), files=index['files'],
                    errors=index['errors'], match_method='product_matcher_v1')
    result = default_search_catalog(query, limit=limit, directory=directory, include_invoices=include_invoices)
    if not result['items']:
        from App.invoicing.ai_matching import proposals
        ai = proposals(query,index,limit=limit)
        if ai['items']:
            return {**result,'items':ai['items'],'errors':result['errors']+ai['errors'],
                    'match_method':'ai_matching'}
        result['errors']=result['errors']+ai['errors']
    return {**result, 'match_method': 'default_fallback',
            'items': [{**row, 'match_method': 'default_fallback'} for row in result['items']]}


def catalog_named_item(name, unit, directory=None):
    index = catalog(directory)
    variants = index["by_name"].get(_name_key(name, unit), [])
    if len(variants) == 1 and len(index["by_id"].get(variants[0]["id"], [])) == 1:
        return variants[0]
    return None


def catalog_item(identity):
    if not identity:
        return None
    if identity.startswith("invoice:"):
        row = next((row for row in invoice_catalog() if row["id"] == identity), None)
        return (catalog_named_item(row["name"], row["unit"]) or row) if row else None
    variants = catalog()["by_id"].get(identity.lower(), [])
    return variants[0] if len(variants) == 1 else None


def price_tie_key(item, receipt_price):
    from App.matching.ranking import _price
    price, target = _price(item.get('accounting_price')), _price(receipt_price)
    difference = abs(target-price)/price if price and target is not None else None
    return (difference is None, difference if difference is not None else Decimal(0))


def sort_candidate_ties(items, receipt_price):
    from App.matching.ranking import _price, candidate_rank_key
    if _price(receipt_price) is None:
        return list(items)
    # Source priorities stay fixed; price reorders only scored suggestions.
    groups = {}
    for index, item in enumerate(items):
        if item.get('score', item.get('match_score')) is not None:
            groups.setdefault(item.get('origin'), []).append(index)
    ordered = list(items)
    def key(item):
        candidate = dict(stock_id=item.get('id') or item.get('catalog_id') or '',
                         stock_price=item.get('accounting_price'),
                         similarity_pct=item.get('score', item.get('match_score')))
        return candidate_rank_key(candidate, _price(receipt_price))
    for slots in groups.values():
        candidates = sorted((items[index] for index in slots), key=key)
        for index, item in zip(slots, candidates):
            ordered[index] = item
    return ordered


@lru_cache(maxsize=256)
def _suggest(query, signature, invoice_signature, ai_signature, article=None, database_id=None, receipt_price=None):
    matches = search_catalog(query, limit=max(1, len(catalog()["items"])+len(invoice_catalog())), include_invoices=True)["items"]
    if not matches and article:
        from App.invoicing.ai_matching import proposals
        matches=proposals(query,catalog(),article=article,limit=max(1,len(catalog()['items'])))['items']
    matches = sort_candidate_ties(matches, receipt_price)
    return matches[0] if matches else None


def suggest_catalog(query, article=None, *, receipt_price=None):
    from App.invoicing.learned_mapping import TRAINING_DIR
    from App.invoicing.ai_matching import signature as ai_signature
    signature = tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size)
                      for path in sorted(TRAINING_DIR.glob("case_*/expected.json")))
    product = _suggest(query, catalog()["signature"], signature, ai_signature(), article, _DATABASE.get(), receipt_price)
    return dict(product) if product else None
