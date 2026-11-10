"""Read-only category assignments, scoped to the classified 1C database."""
from functools import lru_cache
from pathlib import Path
import re
import unicodedata

CLASSIFIED_FILE = Path(__file__).resolve().parents[2] / 'data' / 'reference' / '1c_ref' / '1C_classified_2026-10-16.xlsx'
CLASSIFIED_DATABASE = 'buh_kz_01'
UNCLASSIFIED = 'Без категории / требует проверки'


def _name(value):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(value or '')).strip()).casefold()


@lru_cache(maxsize=4)
def _load(path, modified_ns, size):
    from openpyxl import load_workbook
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = workbook.worksheets[0].iter_rows(values_only=True)
        headers = next(rows)
        result = {}
        for values in rows:
            row = dict(zip(headers, values))
            code = str(row.get('Код 1С') or '').strip()
            if not code or code in result:
                if code:
                    result[code] = None
                continue
            if row.get('needs_review') == 'да':
                result[code] = None
                continue
            parts = tuple(str(row.get(field) or '').strip() for field in ('category', 'subcategory', 'product_type'))
            result[code] = (_name(row.get('Название 1С')), parts) if all(parts) else None
        return result
    finally:
        workbook.close()


def assignments(stock, database_id, path=None):
    path = Path(path or CLASSIFIED_FILE)
    if str(database_id or '').strip().casefold() != CLASSIFIED_DATABASE or not path.is_file():
        return {}
    stat = path.stat()
    classified = _load(str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    output = {}
    for item in stock:
        code = str(item.get('code') or item.get('id') or '').strip()
        value = classified.get(code)
        if value and value[0] == _name(item.get('name')):
            output[str(item.get('id') or code)] = value[1]
    return output


def options(stock, database_id, path=None):
    assigned = assignments(stock, database_id, path)
    paths = set()
    for parts in assigned.values():
        paths.update(parts[:level] for level in (1, 2, 3))
    return [dict(label=' → '.join(parts), path=list(parts)) for parts in sorted(paths)] + [dict(label=UNCLASSIFIED, path=[UNCLASSIFIED])]


def filter_stock(stock, database_id, selected, path=None):
    if not selected:
        return list(stock)
    if isinstance(selected, str):
        term = _name(selected)
        if not term:
            return list(stock)
        assigned = assignments(stock, database_id, path)
        return [item for item in stock
                if any(term in _name(part) for part in assigned.get(str(item.get('id')), ()))]
    if not isinstance(selected, list) or not 1 <= len(selected) <= 3 or any(not isinstance(part, str) or not part for part in selected):
        raise ValueError('Выберите категорию из списка')
    assigned = assignments(stock, database_id, path)
    allowed = {tuple(item['path']) for item in options(stock, database_id, path)}
    if tuple(selected) not in allowed:
        raise ValueError('Выбранная категория недоступна для этой базы')
    return [item for item in stock if (str(item.get('id')) not in assigned if selected == [UNCLASSIFIED]
            else assigned.get(str(item.get('id')), ())[:len(selected)] == tuple(selected))]
