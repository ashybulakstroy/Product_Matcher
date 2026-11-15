#!/usr/bin/env python3
"""Preview/apply/restore the small App rounding integration.

No absolute paths, no .env writes, no dependency on experiments at runtime.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import shutil
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent
APP = ROOT / 'App' / 'ocr'
TARGET = APP / 'receipt_editing.py'
MODULE = APP / 'receipt_rounding.py'
SOURCE_MODULE = PACKAGE / 'src' / 'receipt_rounding.py'
OUTPUT = ROOT / 'data' / 'output' / 'receipt_total_rounding'
MANIFEST = OUTPUT / 'install_manifest.json'
BACKUP = OUTPUT / 'backups'

IMPORT_ANCHOR = 'from App.ocr.receipt_math import verify_vat\n'
NEW_IMPORT = 'from App.ocr.receipt_rounding import validate_total_rounding\n'
OLD_STATUS = '''    if missing:
        status = "incomplete"
    elif expected is None:
        status = "missing_original_total"
    else:
        status = "match" if computed == expected.quantize(CENT, rounding=ROUND_HALF_UP) else "mismatch"
'''
NEW_STATUS = '''    # Evaluate exact result first, then only two mathematically verified
    # whole-tenge rounding methods. No arbitrary <= 1 KZT tolerance.
    rounding = validate_total_rounding(
        [(row['row'], row['kind'], row['line_total'])
         for row in receipt['rows'] if row['kind'] in PAYABLE and row.get('line_total') is not None],
        expected, computed, missing=bool(missing),
    )
    status = rounding['status']
'''
OLD_CALC = '''    receipt["calculation"] = dict(status=status,validation_status=validation_status,vat=vat, products_subtotal=money(totals["product"]),
        services_subtotal=money(totals["service"]), tax_subtotal=money(totals["tax"]),
        computed_total=money(computed), expected_total=money(expected) if expected is not None else None,
        difference=money(computed-expected) if expected is not None and not missing else None,
        missing_rows=missing, missing_original_total=expected is None,
        rule="sum(quantity × price), including separate VAT rows")
'''
NEW_CALC = '''    # The primary computed_total reflects an accepted printed rounding policy,
    # while exact_computed_total preserves the independent cent-accurate sum.
    accepted_total = (parse_number(rounding['accepted_total']) if status == 'match'
                      else None)
    display_total = accepted_total if accepted_total is not None else computed
    receipt["calculation"] = dict(status=status,validation_status=validation_status,vat=vat, products_subtotal=money(totals["product"]),
        services_subtotal=money(totals["service"]), tax_subtotal=money(totals["tax"]),
        computed_total=money(display_total), exact_computed_total=money(computed),
        expected_total=money(expected) if expected is not None else None,
        difference=money(display_total-expected) if expected is not None and not missing else None,
        exact_difference=rounding['exact_difference'],
        rounding_mode=rounding['mode'], rounding_adjustment=rounding['adjustment'],
        rounding=rounding,
        missing_rows=missing, missing_original_total=expected is None,
        rule="sum(quantity × price), including separate VAT rows; verified whole-tenge rounding only")
'''


def sha(data):
    return hashlib.sha256(data).hexdigest()


def _replace_once(content, before, after, description):
    count = content.count(before)
    if count != 1:
        raise RuntimeError(f'{description}: ожидалось ровно одно место замены, найдено {count}. Отказ от изменений.')
    return content.replace(before, after, 1)


def patch_source(content):
    if NEW_IMPORT in content:
        raise RuntimeError('Интеграция уже присутствует или исходник отличается. Используйте --restore или проверьте вручную.')
    for old,new,description in [
        (IMPORT_ANCHOR, IMPORT_ANCHOR+NEW_IMPORT, 'импорт'),
        (OLD_STATUS, NEW_STATUS, 'проверка итоговой суммы'),
        (OLD_CALC, NEW_CALC, 'диагностическая структура calculation'),
    ]:
        content=_replace_once(content,old,new,description)
    return content


def check_project():
    if not TARGET.is_file():
        raise FileNotFoundError(f'Не найден {TARGET.relative_to(ROOT)}. Распакуйте эксперимент в корень Product_Matcher.')
    if not SOURCE_MODULE.is_file():
        raise FileNotFoundError(f'Не найден модуль: {SOURCE_MODULE}')


def ensure_preview():
    check_project()
    before=TARGET.read_text(encoding='utf-8')
    after=patch_source(before)
    staged_module=SOURCE_MODULE.read_bytes()
    if MODULE.exists() and MODULE.read_bytes() != staged_module:
        raise RuntimeError('App/ocr/receipt_rounding.py уже существует с другим содержимым. Не перезаписывать.')
    diff=''.join(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
             fromfile='App/ocr/receipt_editing.py (до)',tofile='App/ocr/receipt_editing.py (после)'))
    diff+='\n# Новый модуль App/ocr/receipt_rounding.py: '+str(len(staged_module))+' байт\n'
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/'preview.diff').write_text(diff,encoding='utf-8')
    print('Preview: data/output/receipt_total_rounding/preview.diff')
    return before.encode('utf-8'),after.encode('utf-8'),staged_module


def apply():
    check_project()
    if MANIFEST.exists():
        meta=json.loads(MANIFEST.read_text(encoding='utf-8'))
        if (TARGET.is_file() and sha(TARGET.read_bytes())==meta['target_after_sha256'] and
                MODULE.is_file() and sha(MODULE.read_bytes())==meta['module_after_sha256']):
            print('Уже установлено, никаких изменений.');return
        raise RuntimeError('Есть manifest предыдущей установки, но файлы изменились. Не перезаписывать изменения.')
    old,new,mod=ensure_preview()
    old_module=MODULE.read_bytes() if MODULE.exists() else None
    BACKUP.mkdir(parents=True,exist_ok=True)
    (BACKUP/'receipt_editing.py.before').write_bytes(old)
    if old_module is not None:
        (BACKUP/'receipt_rounding.py.before').write_bytes(old_module)
    manifest={
        'target':'App/ocr/receipt_editing.py', 'module':'App/ocr/receipt_rounding.py',
        'target_before_sha256':sha(old),'target_after_sha256':sha(new),
        'module_before_sha256':sha(old_module) if old_module is not None else None,
        'module_after_sha256':sha(mod), 'module_created':old_module is None,
        'version':'1.0',
    }
    try:
        # Syntax-check the future source before touching the working application.
        compile(new.decode('utf-8'),'App/ocr/receipt_editing.py','exec')
        compile(mod.decode('utf-8'),'App/ocr/receipt_rounding.py','exec')
        TARGET.write_bytes(new)
        MODULE.write_bytes(mod)
        MANIFEST.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    except Exception:
        TARGET.write_bytes(old)
        if old_module is None:
            MODULE.unlink(missing_ok=True)
        else:
            MODULE.write_bytes(old_module)
        raise
    print('Установлено: только App/ocr/receipt_editing.py и App/ocr/receipt_rounding.py')
    print('Резервные копии: data/output/receipt_total_rounding/backups/')


def restore():
    if not MANIFEST.exists():
        raise RuntimeError('Не найден manifest установки; безопасный откат невозможен.')
    meta=json.loads(MANIFEST.read_text(encoding='utf-8'))
    if not TARGET.exists() or sha(TARGET.read_bytes())!=meta['target_after_sha256']:
        raise RuntimeError('App/ocr/receipt_editing.py изменён после установки. Откат остановлен — требуется ручной merge.')
    if not MODULE.exists() or sha(MODULE.read_bytes())!=meta['module_after_sha256']:
        raise RuntimeError('App/ocr/receipt_rounding.py изменён после установки. Откат остановлен — требуется ручной merge.')
    prior=(BACKUP/'receipt_editing.py.before').read_bytes()
    if sha(prior)!=meta['target_before_sha256']:
        raise RuntimeError('Контрольная сумма резервной копии не совпадает.')
    old_module=None
    if not meta['module_created']:
        old_module=(BACKUP/'receipt_rounding.py.before').read_bytes()
        if sha(old_module)!=meta['module_before_sha256']:
            raise RuntimeError('Контрольная сумма резервной копии модуля не совпадает.')
    TARGET.write_bytes(prior)
    if meta['module_created']:
        MODULE.unlink()
    else:
        MODULE.write_bytes(old_module)
    MANIFEST.unlink()
    print('Выполнен побайтовый откат. Резервные копии сохранены.')


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    actions=cli.add_mutually_exclusive_group(required=True)
    for action in ('preview','apply','restore'):
        actions.add_argument('--'+action,action='store_true')
    args=cli.parse_args()
    if args.preview:ensure_preview()
    elif args.apply:apply()
    elif args.restore:restore()
