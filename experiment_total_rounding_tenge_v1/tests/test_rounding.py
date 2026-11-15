"""Decimal rounding evidence and guard rails (no OCR mocks)."""
from decimal import Decimal
from pathlib import Path
import importlib.util
import json

import pytest

PKG=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('receipt_rounding',PKG/'src'/'receipt_rounding.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
validate=module.validate_total_rounding


def inspect(lines, expected, *, missing=False):
    vals=[(i+1,'product',Decimal(str(v))) for i,v in enumerate(lines)]
    exact=sum((val for _,_,val in vals), Decimal(0))
    return validate(vals,expected,exact,missing=missing)


def test_receipt_16_lines_and_vat_are_real_fixture_numbers():
    f=json.loads((PKG/'fixtures'/'receipt_16_rows.json').read_text(encoding='utf8'))
    rows=[(r['row'],'product',Decimal(r['quantity'])*Decimal(r['price'])) for r in f['rows']]
    rows.append((17,'tax',Decimal('11147')))
    exact=sum((v for _,_,v in rows),Decimal(0))
    assert sum(v for _,k,v in rows if k=='product')==Decimal('69665.65')
    assert Decimal('69665.65')*Decimal('.16')==Decimal('11146.5040')
    assert exact==Decimal('80812.65')
    result=validate(rows,Decimal('80813'),exact)
    assert result['status']=='match'
    assert result['mode']=='sum_then_integer'
    assert result['accepted_total']=='80813.00'
    assert result['adjustment']=='0.35'
    assert result['exact_total']=='80812.65'
    assert result['fractional_rows']==[{'row':14,'kind':'product','line_amount':'12535.65',
                                           'whole_tenge_amount':'12536.00','adjustment':'0.35'}]
    assert result['matched_modes']==['sum_then_integer','each_line_integer']


def test_exact_amount_no_rounding():
    r=inspect(['10','2.15'],'12.15')
    assert r['status']=='match' and r['mode']=='exact' and r['adjustment']=='0.00'


def test_an_integer_only_receipt_cannot_hide_a_wrong_tenge():
    r=inspect(['100','200'],'301')
    assert r['status']=='mismatch' and r['reason']=='no_fractional_tenge_lines'


def test_fractional_total_does_not_admit_arbitrary_one_tenge_difference():
    r=inspect(['100.35','200'],'301')
    assert r['status']=='mismatch'


def test_wrong_printed_decimal_value_is_not_rounded_to_integer():
    r=inspect(['100.65','200'],'301.10')
    assert r['status']=='mismatch' and r['reason']=='printed_total_has_tiyn'


def test_half_up_final_sum_acceptance():
    r=inspect(['100.65','200'],'301.00')
    assert r['status']=='match' and r['mode']=='sum_then_integer'


def test_each_line_rounding_without_total_rounding_is_disclosed():
    # 100.40+200.40=300.80 -> 301 by grand rounding; line rounds -> 300
    r=inspect(['100.40','200.40'],'300')
    assert r['status']=='match' and r['mode']=='each_line_integer'
    assert r['matched_modes']==['each_line_integer']
    assert r['adjustment']=='-0.80'


def test_total_rounding_and_each_line_can_disagree_but_only_exact_formula_passes():
    r=inspect(['100.40','200.40'],'302')
    assert r['status']=='mismatch'
    assert r['candidates']=={'sum_then_integer':'301.00','each_line_integer':'300.00'}


def test_missing_cell_blocks_rounding_even_if_computed_sum_rounds():
    r=inspect(['100.65','200'],'301',missing=True)
    assert r['status']=='incomplete' and r['accepted_total'] is None


def test_missing_printed_total_cannot_be_inferred():
    r=inspect(['100.65','200'],None)
    assert r['status']=='missing_original_total' and r['accepted_total'] is None


def test_ledger_immutable_and_inconsistent_ledger_rejected():
    entries=[(1,'product',Decimal('100.65'))]
    snapshot=list(entries)
    r=validate(entries,Decimal('101'),Decimal('150.65'))
    assert r['status']=='mismatch' and r['reason']=='line_ledger_does_not_match_computed_total'
    assert entries==snapshot


@pytest.mark.parametrize('bad', ['NaN','Infinity','-1','oops'])
def test_invalid_amounts_do_not_get_accepted(bad):
    r=inspect(['100.65','200'],bad)
    assert r['status']=='missing_original_total' or r['status']=='mismatch'


def test_source_images_included_for_real_regression():
    assert (PKG/'fixtures'/'receipt_original_2142822991.png').stat().st_size>100000
    assert (PKG/'fixtures'/'app_mismatch_0_35.png').stat().st_size>100000
