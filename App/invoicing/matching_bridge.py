"""Read-only bridge from invoice items to Product_Matcher voting engine.

Preserves 1C identifiers (including leading zeroes). Never writes invoice
selections, learning links, source OCR names, or stock snapshots.
"""
from __future__ import annotations

import math
import os
from decimal import Decimal, InvalidOperation

DEFAULT_ALGORITHMS = ('copilot_1', 'sql_str_3', 'opencode_4', 'token_set')


def _number(value, default=0.0):
    if value is None or value == '':
        return default
    try:
        d = Decimal(str(value).replace(',', '.').replace(' ', ''))
        result = float(d) if d.is_finite() else default
        return result if result is not None and math.isfinite(result) else default
    except (ValueError, TypeError, InvalidOperation):
        return default


def confident_default(items, *, threshold=85., gap=8.):
    """Conservative selection of when to ask the voting matcher for help.

    Lexical scores are NOT calibrated probabilities; threshold is a routing
    heuristic only, never a basis for auto-confirmation.
    """
    if not items:
        return False
    first = _number(items[0].get('score'), -1.)
    second = _number(items[1].get('score'), -1.) if len(items) > 1 else -1.
    return first >= threshold and (second < 0 or first-second >= gap)


def stock_rows(catalog_index):
    """Build stock entries from the catalog of the *current* 1C database."""
    result = []
    by_identity = {}
    ambiguous = set()
    for item in catalog_index.get('items', ()):
        identity = str(item.get('id') or '')
        if not identity or identity in ambiguous:
            continue
        if identity in by_identity:
            # Never map ambiguous IDs to the first matching card.
            by_identity.pop(identity, None)
            ambiguous.add(identity)
            continue
        by_identity[identity] = item
    for identity, item in by_identity.items():
        result.append({'id': identity, 'name': str(item.get('name') or ''),
                       'quantity': _number(item.get('stock_quantity')),
                       'accounting_price': item.get('accounting_price')})
    return result, by_identity


def vote_candidates(query, catalog_index, *, price=None, quantity=1, limit=10,
                    algorithms=None, matcher=None):
    """Run CPU voting, return candidates backed by the live catalog.

    `matcher` injection is for isolated synthetic tests. Semantic ONNX models
    are deliberately not run implicitly in the OCR pipeline.
    """
    needed = _number(quantity, None)
    if needed is None or needed <= 0:
        raise ValueError('Для поиска аналогов требуется положительное количество из чека')
    limit = max(1, min(int(limit), 10))
    if matcher is None:
        from App.matching.engine import match as matcher
    items, by_identity = stock_rows(catalog_index)
    if not query.strip() or not items:
        return {'items': [], 'recommended_id': None, 'is_tie': False,
                'voting_algorithms': 0, 'errors': ['Нет запроса или каталога 1С']}
    selected = list(algorithms or DEFAULT_ALGORITHMS)
    checked = [{'id': 'invoice_row', 'name': query, 'price': price}]
    result = matcher(checked, items, top_n=max(1, min(int(limit), 10)),
                     min_stock=needed,
                     algorithms=selected, parallel=True)
    positions = result.get('voting', {}).get('positions') or []
    position = positions[0] if positions else {}
    candidates = []
    for voted in position.get('candidates') or []:
        identity = str(voted.get('stock_id') or '')
        catalog_item = by_identity.get(identity)
        if catalog_item is None:
            continue
        if _number(catalog_item.get('stock_quantity')) < needed:
            continue
        candidates.append({**catalog_item, 'id': identity,
                           'score': voted.get('vote_pct', 0),
                           'votes': voted.get('votes', 0),
                           'vote_pct': voted.get('vote_pct', 0),
                           'voters': voted.get('voters', []),
                           'origin': 'analog_vote', 'match_method': 'analog_vote'})
    recommended = position.get('recommended_stock_id')
    if recommended is not None:
        recommended = str(recommended)
    if position.get('is_tie') or recommended not in {i['id'] for i in candidates[:limit]}:
        recommended = None
    return {'items': candidates[:limit], 'recommended_id': recommended,
            'is_tie': bool(position.get('is_tie')),
            'voting_algorithms': position.get('voting_algorithms', 0),
            'errors': [r.get('error') for r in result.get('results_by_algorithm', [])
                       if r.get('status') != 'ok' and r.get('error')]}


def fallback_suggestions(row, current_items, catalog_index, *, matcher=None, database_id=None):
    """Only append read-only alternatives when ordinary search is uncertain."""
    threshold = _number(os.getenv('ANALOG_FALLBACK_MIN_SCORE', '85'), 85.)
    gap = _number(os.getenv('ANALOG_FALLBACK_MIN_GAP', '8'), 8.)
    if confident_default(current_items, threshold=threshold, gap=gap):
        return []
    query = str(row.get('name') or '')
    if not query.strip():
        return []
    try:
        validate_database(catalog_index, database_id)
        result = vote_candidates(query, catalog_index, price=row.get('price'),
                                 quantity=row.get('quantity'), matcher=matcher)
    except Exception:
        # Optional matcher never interrupts receipt/invoice processing.
        import logging
        logging.exception('Optional analog fallback failed')
        return []
    seen = {str(item.get('id') or item.get('catalog_id') or '') for item in current_items}
    return [item for item in result['items'] if str(item['id']) not in seen]


def validate_database(index, database_id=None):
    if index.get('stock_source') == 'json_error':
        raise ValueError('Остатки базы 1С повреждены или недоступны')
    if database_id is not None:
        actual = str(index.get('database_id') or '').strip().casefold()
        if not actual or actual != str(database_id).strip().casefold():
            raise ValueError('Для этой базы 1С нет подходящего снимка остатков')


def find_receipt_row(receipt, row_id):
    """Resolve row ID from the existing validated electronic receipt."""
    from App.invoicing.receipt_adapter import snapshot_from_receipt, to_generation_source
    source = to_generation_source(snapshot_from_receipt(receipt))
    return next((item for item in source['items'] if str(item['row_id']) == str(row_id)), None)


def for_invoice_row(receipt, row_id, database_id=None, *, matcher=None):
    from App.invoicing.catalog_1c import catalog_scope
    with catalog_scope(database_id) as index:
        validate_database(index, database_id)
        row = find_receipt_row(receipt, row_id)
        if row is None:
            raise ValueError('Строка электронного чека не найдена')
        data = vote_candidates(str(row.get('name') or ''), index,
                               price=row.get('price'), quantity=row.get('quantity'),
                               matcher=matcher)
        return {'row_id': str(row['row_id']), 'database_id': database_id,
                'receipt_revision': receipt.get('revision', 0),
                'source_name': row.get('name'), **data}
