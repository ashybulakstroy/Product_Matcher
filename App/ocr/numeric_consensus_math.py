"""Conservative OCR-evidence numeric correction, guarded by exact receipt math.

Applies to a single existing, parsed product/service quantity or price only.
Re-reading the *same* model is not independent cross-engine evidence; requiring
four consistent views AND unique total/VAT reconciliation reduces false fixes.
No number is invented from the arithmetic difference.
"""
from __future__ import annotations

import copy
from decimal import Decimal
from typing import Any, Callable

from App.ocr.receipt_editing import parse_number, update_receipt

VERSION = 'single-cell-ocr-consensus-math-v2'


def _valid_math(calculation: dict) -> bool:
    return (calculation.get('status') == 'match' and
            calculation.get('validation_status', 'match') == 'match' and
            calculation.get('vat', {}).get('status') in {'match', 'not_applicable'} and
            parse_number(calculation.get('exact_computed_total', calculation.get('computed_total')))
            == parse_number(calculation.get('expected_total')))


def _supported(proposal: dict, before: Decimal) -> bool:
    """Four concordant actual OCR views (not just ``agreed=True``)."""
    field = proposal.get('field')
    if field not in {'quantity', 'price'} or proposal.get('kind') not in {'product', 'service'}:
        return False
    if not proposal.get('agreed'):
        return False
    candidate = parse_number(proposal.get('candidate'))
    if candidate is None or candidate == before:
        return False
    views = proposal.get('views') or []
    if len(views) < 4:
        return False
    # Single digit quantities are genuinely harder to recognize, and need the
    # existing special consensus floor. Prices require the stronger 0.55 gate.
    floor = Decimal('.18') if field == 'quantity' and candidate == candidate.to_integral_value() and candidate < 10 else Decimal('.55')
    for view in views:
        if parse_number(view.get('number', view.get('text'))) != candidate:
            return False
        try:
            confidence = Decimal(str(view.get('confidence', 0)))
        except Exception:
            return False
        if not confidence.is_finite() or confidence < floor:
            return False
    return True


def accept_unique_single_cell_correction(receipt: dict, proposals: list[dict],
                                          validate_fn: Callable[[dict], dict] = update_receipt
                                          ) -> tuple[dict | None, list[dict], dict]:
    """Correct a number ONLY if one OCR-supported change makes total+VAT match.

    Does not change the original `*_raw` values or the printed total. Never
    replaces a user/manual-edit field. Multiple valid alternatives are rejected.
    """
    before_check = validate_fn(copy.deepcopy(receipt))['calculation']
    report = {'version': VERSION, 'status': 'not_applicable', 'examined': [], 'solutions': 0}
    if (before_check.get('status') == 'match' and
            before_check.get('validation_status', 'match') == 'match' and
            before_check.get('vat', {}).get('status') in {'match', 'not_applicable'}):
        report['status'] = 'already_matched'
        return None, [], report
    if parse_number(before_check.get('expected_total')) is None:
        report['status'] = 'no_printed_total'
        return None, [], report

    by_id = {row['row']: row for row in receipt.get('rows', []) if 'row' in row}
    manual_fields = {(change.get('row'), change.get('field'))
                     for entry in receipt.get('audit_log', [])
                     for change in entry.get('changes', [])}
    options = {}
    for proposal in proposals:
        row = by_id.get(proposal.get('row'))
        field = proposal.get('field')
        if row is None or row.get('kind') not in {'product', 'service'} or field not in {'quantity', 'price'}:
            continue
        source = str(row.get(field + '_source', '')).casefold()
        if source in {'manual', 'user', 'human'} or (row['row'], field) in manual_fields:
            continue
        old = parse_number(row.get(field, row.get(field + '_raw', '')))
        if old is None or not _supported(proposal, old):
            continue
        new = parse_number(proposal['candidate'])
        key = (row['row'], field, new)
        options[key] = proposal

    accepted = []
    for (row_id, field, new), evidence in options.items():
        trial = copy.deepcopy(receipt)
        target = next(row for row in trial['rows'] if row['row'] == row_id)
        before = parse_number(target.get(field, target.get(field + '_raw', '')))
        target[field] = str(evidence['candidate'])
        target[field + '_source'] = 'numeric_consensus_math'
        target[field + '_ocr'] = {
            'consensus_views': len(evidence['views']),
            'bounds': evidence.get('bounds'),
            'evidence': evidence['views'],
            'rule': VERSION,
        }
        checked = validate_fn(copy.deepcopy(trial))['calculation']
        report['examined'].append({'row': row_id, 'field': field,
                                   'before': str(before), 'after': str(new),
                                   'math_valid': _valid_math(checked)})
        if _valid_math(checked):
            accepted.append((trial, {'row': row_id, 'field': field, 'before': str(before),
                                     'after': str(new), 'reason': VERSION}))

    report['solutions'] = len(accepted)
    if len(accepted) == 1:
        report['status'] = 'unique_verified'
        trial, update = accepted[0]
        return trial, [update], report
    report['status'] = 'ambiguous' if accepted else 'not_confirmed'
    return None, [], report
