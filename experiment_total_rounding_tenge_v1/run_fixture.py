#!/usr/bin/env python3
"""Check the photographed 16-row example or revalidate an existing OCR job.

Default mode uses manually transcribed *real* receipt numbers as a regression
fixture; --full-ocr requires App + its locally installed RapidOCR model.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

PACKAGE=Path(__file__).resolve().parent
ROOT=PACKAGE.parent
sys.path.insert(0,str(ROOT))
from src.receipt_rounding import validate_total_rounding

OUT=ROOT/'data'/'output'/'receipt_total_rounding'
FIXTURE=PACKAGE/'fixtures'/'receipt_16_rows.json'


def _amounts(rows):
    return [(row['row'],row.get('kind','product'),(Decimal(row['quantity'])*Decimal(row['price'])).quantize(Decimal('.01'),rounding=ROUND_HALF_UP)) for row in rows]


def math_fixture():
    fixture=json.loads(FIXTURE.read_text(encoding='utf-8'))
    rows=[dict(item,kind='product') for item in fixture['rows']]+[fixture['vat_row']]
    amounts=_amounts(rows)
    computed=sum((amount for _,_,amount in amounts),Decimal(0))
    calculated_tax=sum(amount for _,kind,amount in amounts if kind=='tax')
    base=sum(amount for _,kind,amount in amounts if kind=='product')
    expected=Decimal(fixture['expected_total'])
    result=validate_total_rounding(amounts,expected,computed)
    rounded_vat=(base*Decimal(fixture['vat_rate_percent'])/100).quantize(Decimal('1'),rounding=ROUND_HALF_UP)
    report=dict(fixture=fixture['document'],products_total=str(base),vat=str(calculated_tax),
                configured_vat_expected=str(rounded_vat),math=result,
                fixture_expected=fixture['expected_total'],
                checks=dict(
                    lines_count=len(fixture['rows'])==16,
                    product_subtotal=str(base)==fixture['expected_products_subtotal_exact'],
                    vat_matches=rounded_vat==calculated_tax,
                    exact_total=str(computed)==fixture['expected_computed_exact'],
                    accepted_total=result['accepted_total']==fixture['expected_total'],
                    rounding_adjustment=result['adjustment']==fixture['expected_rounding_adjustment'],
                ))
    return report


def live_job(job_dir):
    from App.ocr.receipt_editing import update_receipt
    path=Path(job_dir)/'receipt.json'
    if not path.is_file():
        raise FileNotFoundError(path)
    receipt=json.loads(path.read_text(encoding='utf-8'))
    result=update_receipt(receipt)
    return dict(job_dir=str(job_dir),calculation=result['calculation'],
                rows=len(result['rows']), can_finalize=result['can_finalize'])


def full_ocr():
    from App.web_server import Job
    from App.ocr.web_pipeline import ReceiptPipeline
    photo=PACKAGE/'fixtures'/'receipt_original_2142822991.png'
    job_dir=OUT/'full_ocr_job'
    job_dir.mkdir(parents=True,exist_ok=True)
    (job_dir/'upload.bin').write_bytes(photo.read_bytes())
    job=Job('a'*32, photo.name, job_dir)
    receipt=ReceiptPipeline().process(job)
    from App.ocr.receipt_editing import parse_number
    source=json.loads(FIXTURE.read_text(encoding='utf8'))
    actual_products=[r for r in receipt['rows'] if r.get('kind')=='product']
    actual_tax=[r for r in receipt['rows'] if r.get('kind')=='tax']
    matching_rows=(len(actual_products)==len(source['rows']) and all(
        parse_number(actual.get('quantity',actual.get('quantity_raw'))) == Decimal(expected['quantity']) and
        parse_number(actual.get('price',actual.get('price_raw'))) == Decimal(expected['price'])
        for actual,expected in zip(actual_products,source['rows'])))
    vat_ok=(len(actual_tax)==1 and
            parse_number(actual_tax[0].get('quantity',actual_tax[0].get('quantity_raw')))==Decimal('11147') and
            parse_number(actual_tax[0].get('price',actual_tax[0].get('price_raw')))==Decimal('1'))
    calc=receipt['calculation']
    checks=dict(product_count=len(actual_products)==16,numeric_rows=matching_rows,
                vat_row=vat_ok, printed_total=calc.get('expected_total')=='80813.00',
                rounded_total=calc.get('computed_total')=='80813.00',
                exact_total=calc.get('exact_computed_total')=='80812.65',
                vat_valid=calc.get('vat',{}).get('status')=='match',
                validation_status=calc.get('validation_status')=='match')
    result=dict(mode='full_ocr', product_count=receipt.get('product_count'),
                rows=len(receipt['rows']),calculation=calc,checks=checks)
    (job_dir/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--full-ocr',action='store_true',help='Запустить всю программу с реальной моделью OCR')
    parser.add_argument('--job-dir',type=Path,help='Повторно проверить ранее обработанное receipt.json')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    name='math_fixture'
    try:
        if args.full_ocr:
            name='full_ocr'; report=full_ocr()
        elif args.job_dir:
            name='existing_job'; report=live_job(args.job_dir)
        else:
            report=math_fixture()
        ok=all(report['checks'].values()) if 'checks' in report else (report['calculation'].get('validation_status')=='match')
        report['success']=ok
    except Exception as e:
        report={'success':False,'error':type(e).__name__,'detail':str(e),'traceback':traceback.format_exc(limit=5)}
    (OUT/(name+'_report.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if report['success'] else 1


if __name__=='__main__':
    sys.exit(main())
