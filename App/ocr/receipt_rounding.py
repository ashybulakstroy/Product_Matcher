"""Evidence-based whole-tenge reconciliation for photographed Torgsoft receipts.

This is a pure Decimal module. It does not read .env, run OCR, infer digits,
change source numbers, or alter VAT verification. No blanket tolerance applies.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext

TENGE = Decimal('1')
TIYN = Decimal('0.01')
VERSION = 'receipt-total-rounding-1.0'


def _decimal(value):
    if value is None:
        return None
    try:
        amount = Decimal(str(value))
    except (ValueError, InvalidOperation, TypeError):
        return None
    return amount if amount.is_finite() and amount >= 0 else None


def _money(value):
    return format(value.quantize(TIYN, rounding=ROUND_HALF_UP), '.2f') if value is not None else None


def validate_total_rounding(line_amounts, expected, exact_computed, *, missing=False):
    """Return verified total status and an auditable rounding explanation.

    line_amounts: iterable of (row_id, kind, line_amount), already computed by
    App receipt_editing from the actual OCR/user-verified quantity and price.
    The printed total is never inferred from the line amounts.

    Only these exact identities can validate a mismatch:
      a) sum of cent-accurate lines, rounded HALF_UP to whole tenge
      b) sum of *each* cent-accurate line, rounded HALF_UP to whole tenge

    This is enabled only for integer-valued printed totals and only if at
    least one *actual* line has fractional tenge. A 1-tenge error among wholly
    integer lines remains an error. VAT remains an independent hard gate.
    """
    expected_value = _decimal(expected)
    exact_value = _decimal(exact_computed)
    report = dict(version=VERSION, status='mismatch', mode='unverified',
                  accepted_total=None, exact_total=_money(exact_value),
                  printed_total=_money(expected_value), adjustment=None,
                  exact_difference=_money(exact_value - expected_value)
                  if exact_value is not None and expected_value is not None else None,
                  candidates={}, fractional_rows=[], matched_modes=[],
                  rule='exact Decimal equality or a verified whole-tenge HALF_UP identity')
    if missing:
        report.update(status='incomplete', mode='not_checked', reason='missing_numeric_fields')
        return report
    if expected_value is None:
        report.update(status='missing_original_total', mode='not_checked', reason='printed_total_missing')
        return report
    if exact_value is None:
        report.update(reason='invalid_calculated_total')
        return report
    entries=[]
    try:
        for row_id, kind, raw in line_amounts:
            value=_decimal(raw)
            if value is None:
                report.update(reason='invalid_line_amount', row=row_id)
                return report
            entries.append((row_id, str(kind), value))
    except (TypeError, ValueError):
        report.update(reason='invalid_line_ledger')
        return report
    with localcontext() as context:
        context.prec=60
        ledger_total=sum((v for _,_,v in entries), Decimal(0))
        if ledger_total != exact_value:
            report.update(reason='line_ledger_does_not_match_computed_total',
                          ledger_total=_money(ledger_total))
            return report
        if exact_value == expected_value:
            report.update(status='match', mode='exact', accepted_total=_money(expected_value),
                          adjustment='0.00', matched_modes=['exact'], reason='exact_amounts_equal')
            return report
        fractional=[(row_id, kind, amount) for row_id, kind, amount in entries
                    if amount != amount.to_integral_value()]
        report['fractional_rows']=[dict(row=row, kind=kind, line_amount=_money(v),
                                        whole_tenge_amount=_money(v.quantize(TENGE, rounding=ROUND_HALF_UP)),
                                        adjustment=_money(v.quantize(TENGE, rounding=ROUND_HALF_UP)-v))
                                  for row,kind,v in fractional]
        if not fractional:
            report.update(reason='no_fractional_tenge_lines')
            return report
        if expected_value != expected_value.to_integral_value():
            report.update(reason='printed_total_has_tiyn')
            return report
        total_round=ledger_total.quantize(TENGE, rounding=ROUND_HALF_UP)
        lines_round=sum((v.quantize(TENGE, rounding=ROUND_HALF_UP) for _,_,v in entries), Decimal(0))
        report['candidates']={'sum_then_integer': _money(total_round),
                              'each_line_integer': _money(lines_round)}
        matches=[mode for mode,value in [('sum_then_integer',total_round),
                                           ('each_line_integer',lines_round)] if value==expected_value]
        report['matched_modes']=matches
        if not matches:
            report.update(reason='no_exact_whole_tenge_rounding_identity')
            return report
        selected=matches[0]
        report.update(status='match', mode=selected, accepted_total=_money(expected_value),
                      adjustment=_money(expected_value-exact_value),
                      reason='printed_total_equals_verified_rounding_formula')
    return report
