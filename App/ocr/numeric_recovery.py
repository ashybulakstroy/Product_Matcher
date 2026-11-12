"""Conservative second-pass numeric OCR for Torgsoft receipt grid.

No external OCR engine: `recognize` is App's already-loaded RapidOCR text_recognizer.
The module does not alter source OCR (`*_raw`), nor infer numbers from totals.
"""
from __future__ import annotations

import copy
from decimal import Decimal
import math
import re

import cv2
import numpy as np

from App.ocr.receipt_editing import parse_number, update_receipt
from App.ocr.numeric_settings import load_numeric_settings

VERSION = 'numeric-recovery-1.1'
NUMERIC_KINDS = {'product', 'service', 'tax'}


def _as_numeric(text):
    """Parse a recognizable *OCR string*. Never guess missing digits."""
    value = re.sub(r'\s+', '', str(text or '')).replace('₸', '').strip()
    value = re.sub(r'(?i)(?:KZT|ТЕНГЕ|ТГ|[TТ])$', '', value)
    if not re.fullmatch(r'\d+(?:[.,]\d+)*', value):
        return None
    return parse_number(value)


def _clip(value, minimum, maximum):
    return max(minimum, min(maximum, int(value)))


def _crop_cell(image, layout, row, field):
    """Crop *numeric* column at the actual row band, excluding grid borders."""
    if field not in {'quantity', 'price'}:
        return None
    top, bottom = row.get('y_range', (None, None))
    if top is None or bottom is None:
        return None
    if field == 'quantity':
        left, right = layout['name_right'], layout['quantity_right']
    else:
        left, right = layout['quantity_right'], layout['right']
    h, w = image.shape[:2]
    x0 = _clip(math.ceil(left) + 8, 0, w)
    x1 = _clip(math.floor(right) - 8, 0, w)
    y0 = _clip(math.ceil(top) + 7, 0, h)
    y1 = _clip(math.floor(bottom) - 7, 0, h)
    if x1 - x0 < 9 or y1 - y0 < 9:
        return None
    return (x0, y0, x1, y1)


def _printed_total_crop(image, lines, layout):
    """Locate printed total from OCR geometry; do not calculate and fabricate it."""
    candidates = []
    for item in lines:
        box = np.asarray(item.get('box', []), dtype=float)
        if box.shape != (4, 2):
            continue
        center = box.mean(axis=0)
        if center[1] < float(layout['body_bottom']) - 5 or center[0] < float(layout['name_right']):
            continue
        if not re.search(r'\d', str(item.get('text', ''))):
            continue
        candidates.append((float(center[1]), box))
    if not candidates:
        return None
    box = max(candidates, key=lambda pair: pair[0])[1]
    h, w = image.shape[:2]
    x0 = _clip(math.floor(box[:, 0].min()) - 7, 0, w)
    x1 = _clip(math.ceil(box[:, 0].max()) + 7, 0, w)
    y0 = _clip(math.floor(box[:, 1].min()) - 6, 0, h)
    y1 = _clip(math.ceil(box[:, 1].max()) + 6, 0, h)
    return (x0, y0, x1, y1) if x1 - x0 > 9 and y1 - y0 > 9 else None


def _numeric_views(image, bounds):
    """Return two independent preparations of every printed digit in a cell.

    Grid lines are used only for localization and never written over the
    source.  Wrapped numeric lines are concatenated in reading order, which is
    required for the photographed VAT value ``1307`` + ``1``.
    """
    x0,y0,x1,y1=bounds
    base=image[y0:y1,x0:x1].copy()
    if base.size == 0:return []
    gray=cv2.cvtColor(base,cv2.COLOR_BGR2GRAY) if base.ndim==3 else base.copy()
    height,width=gray.shape
    mask=cv2.threshold(cv2.GaussianBlur(gray,(3,3),0),0,255,
                       cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    hk=cv2.getStructuringElement(cv2.MORPH_RECT,(max(12,width//2),1))
    vk=cv2.getStructuringElement(cv2.MORPH_RECT,(1,max(12,height//2)))
    ruling=cv2.bitwise_or(cv2.morphologyEx(mask,cv2.MORPH_OPEN,hk),
                         cv2.morphologyEx(mask,cv2.MORPH_OPEN,vk))
    ink=cv2.bitwise_and(mask,cv2.bitwise_not(ruling))
    _,_,stats,_=cv2.connectedComponentsWithStats(ink,8)
    # Main glyphs define text lines. Tiny decimal marks remain inside the
    # expanded line crop, but cannot create a false extra line themselves.
    min_glyph_h=max(7,int(height*.075))
    glyphs=[]
    for x,y,cw,ch,area in stats[1:]:
        if area<4 or ch<min_glyph_h:continue
        if cw>width*.55 and ch<height*.18:continue
        if ch>height*.70 and cw<max(3,width*.06):continue
        if (y<height*.07 or y+ch>height*.93) and ch<height*.12:continue
        glyphs.append((int(x),int(y),int(cw),int(ch)))
    if not glyphs:return []
    median_h=sorted(ch for _,_,_,ch in glyphs)[len(glyphs)//2]
    groups=[]
    for glyph in sorted(glyphs,key=lambda v:(v[1]+v[3]/2,v[0])):
        cy=glyph[1]+glyph[3]/2
        if not groups or cy-groups[-1]['center']>max(8,median_h*.65):
            groups.append(dict(center=cy,items=[glyph]))
        else:
            groups[-1]['items'].append(glyph)
            groups[-1]['center']=sum(y+ch/2 for _,y,_,ch in groups[-1]['items'])/len(groups[-1]['items'])

    def build(source):
        strips=[]
        for group in groups:
            items=group['items']
            left=max(0,min(x for x,_,_,_ in items)-5)
            right=min(width,max(x+cw for x,_,cw,_ in items)+5)
            top=max(0,min(y for _,y,_,_ in items)-5)
            bottom=min(height,max(y+ch for _,y,_,ch in items)+5)
            strip=source[top:bottom,left:right]
            if strip.size==0:continue
            target=48
            ratio=target/max(1,strip.shape[0])
            strips.append(cv2.resize(strip,(max(8,round(strip.shape[1]*ratio)),target),
                                     interpolation=cv2.INTER_CUBIC))
        if not strips:return None
        combined=strips[0]
        for strip in strips[1:]:combined=np.hstack((combined,np.full((48,5),255,np.uint8),strip))
        combined=cv2.copyMakeBorder(combined,12,12,12,12,cv2.BORDER_CONSTANT,value=255)
        return cv2.cvtColor(combined,cv2.COLOR_GRAY2BGR)

    cleaned=gray.copy();cleaned[ruling>0]=255
    composite=build(cleaned)
    if composite is None:return []
    prepared=cv2.cvtColor(composite,cv2.COLOR_BGR2GRAY)
    # Two genuinely separate image preparations. Their parameters were not
    # selected from the expected totals: both simply preserve thin printed
    # strokes while suppressing the uneven paper background.
    variants=[cv2.adaptiveThreshold(prepared,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY,block,c)
              for block,c in ((15,5),(21,7),(31,9),(41,11))]
    return [cv2.cvtColor(view,cv2.COLOR_GRAY2BGR) for view in variants]


def _recognition_pairs(image, tasks, recognize, lock, max_images=250):
    """Recognize each preparation independently, without cross-cell resize coupling."""
    images, accepted_tasks = [], []
    for task in tasks:
        views = _numeric_views(image, task['bounds'])
        if len(views) < 2 or len(images) + len(views) > max_images:
            continue
        images.extend(views)
        accepted_tasks.append((task,len(views)))
    if not accepted_tasks:
        return []
    def run_independently():
        readings=[]
        for prepared in images:
            one=recognize([prepared])
            if len(one)!=1:
                raise ValueError('Numeric OCR must return one result for one independent crop')
            readings.append(one[0])
        return readings
    if lock is None:
        readings=run_independently()
    else:
        with lock:
            readings=run_independently()
    if len(readings) != len(images):
        raise ValueError('Numeric OCR returned different number of results from number of crops')
    result = []
    offset=0
    for task,view_count in accepted_tasks:
        views = []
        for text, confidence in readings[offset:offset+view_count]:
            number = _as_numeric(text)
            views.append(dict(text=str(text), confidence=float(confidence), number=str(number) if number is not None else None))
        offset+=view_count
        groups={}
        for view in views:
            if view['number'] is not None:groups.setdefault(view['number'],[]).append(view)
        ranked=sorted(groups.items(),key=lambda item:(len(item[1]),sum(v['confidence'] for v in item[1])),reverse=True)
        candidate=ranked[0][0] if ranked else None
        supporters=ranked[0][1] if ranked else []
        unopposed=not ranked or len(ranked)==1 or len(supporters)>len(ranked[1][1])
        standard_votes=sum(v['confidence']>=.55 for v in supporters)
        single_digit=(task.get('field')=='quantity' and re.fullmatch(r'\d',candidate or '') is not None)
        single_votes=sum(v['confidence']>=.18 for v in supporters) if single_digit else 0
        confidence_rule=('single_digit_consensus' if single_digit and single_votes>=2 else 'standard_0.55')
        agreed=bool(candidate is not None and len(supporters)>=2 and unopposed
                    and (standard_votes>=2 or single_votes>=2))
        result.append(dict(**task,views=views,agreed=agreed,supporting_votes=len(supporters),
                           confidence_rule=confidence_rule,candidate=candidate if agreed else None))
    return result


def _validate(receipt):
    return update_receipt(copy.deepcopy(receipt))['calculation']


def _clean_state(receipt):
    """Only count actual OCR check results, not catalog name-review status."""
    calculation = _validate(receipt)
    vat_status = calculation['vat']['status']
    return calculation, vat_status


def _apply_verified(original, proposals, target='tax_only'):
    """Promote OCR evidence only if VAT and printed total improve without regression."""
    receipt = copy.deepcopy(original)
    for proposal in proposals:
        if not proposal.get('agreed') or proposal['candidate'] is None:
            continue
        if proposal['field'] == 'total':
            # Never overwrite the first-pass OCR evidence.
            if receipt.get('expected_total_source') == 'user':
                continue
            receipt['expected_total'] = proposal['candidate']
            receipt['expected_total_source'] = 'numeric_recovery'
            receipt['expected_total_ocr'] = dict(views=proposal['views'], bounds=proposal['bounds'])
            continue
        row = next((r for r in receipt['rows'] if r['row'] == proposal['row']), None)
        if row is None:
            continue
        current_value=_as_numeric(row.get(proposal['field'],row.get(proposal['field']+'_raw','')))
        # The broad second pass must not replace an already valid product or
        # service number merely because another OCR preparation suggests a
        # different value. Missing fields are recoverable; a tax value may be
        # corrected because it has an independent configured-rate equation.
        if current_value is not None and row.get('kind') not in {'tax'}:
            continue
        row[proposal['field']] = proposal['candidate']
        row[proposal['field'] + '_source'] = 'numeric_recovery'
        row[proposal['field'] + '_ocr'] = dict(views=proposal['views'], bounds=proposal['bounds'])
    return receipt


def recover_numeric_receipt(receipt, image, layout, aligned_lines, recognize, *, ocr_lock=None, settings=None):
    """Run two mathematical gates, returning receipt and inspectable JSON diagnostics.

    Stage 1: a VAT-only second pass on mismatch (both numeric cells of VAT row).
    Stage 2: if grand total mismatch/missing math, rescan ALL numeric cells, VAT, printed total.
    Original OCR is never silently overwritten without evidence and calculation checks.
    """
    cfg = settings if settings is not None else load_numeric_settings()
    current = copy.deepcopy(receipt)
    initial_calc, vat = _clean_state(current)
    audit = dict(version=VERSION, vat_rate_percent=cfg['vat_rate_percent'],
                 vat_rate_source=cfg['vat_rate_source'],
                 initial_vat_status=vat,
                 initial_total_status=initial_calc['status'],
                 vat_attempts=[], all_numeric_attempts=[],
                 vat_triggered=False, total_triggered=False, applied=[], warnings=[])
    # Isolated recovery of absent prices before the existing fallback. Do not
    # replace valid prices or derive a number from the receipt total.
    from App.ocr.selective_price_ocr import recover_missing_prices
    current, price_report = recover_missing_prices(
        current, image, layout, recognize, ocr_lock=ocr_lock,
        max_images=cfg['max_images'])
    audit['selective_price'] = price_report
    audit['applied'].extend(price_report['applied'])
    _, vat = _clean_state(current)

    def task_for(row, field):
        bounds = _crop_cell(image, layout, row, field)
        return dict(row=row['row'], kind=row['kind'], field=field, bounds=list(bounds)) if bounds is not None else None

    tax_rows = [r for r in current['rows'] if r.get('kind') == 'tax']
    # Skip round 1 if amount(s) are not meaningful or there is no VAT row.
    if vat == 'mismatch' and tax_rows:
        audit['vat_triggered'] = True
        tasks = [t for row in tax_rows for fld in ('quantity', 'price') if (t := task_for(row, fld))]
        proposals = _recognition_pairs(image, tasks, recognize, ocr_lock, cfg['max_images'])
        audit['vat_attempts'] = proposals
        trial = _apply_verified(current, proposals)
        after, after_vat = _clean_state(trial)
        if after_vat == 'match' and (initial_calc['status'] != 'match' or after['status'] == 'match'):
            changed = [(r, fld) for r in trial['rows'] for fld in ('quantity', 'price')
                       if _as_numeric(r.get(fld, r.get(fld + '_raw', ''))) != _as_numeric(
                           next(old for old in current['rows'] if old['row'] == r['row']).get(
                               fld, next(old for old in current['rows'] if old['row'] == r['row']).get(fld + '_raw', '')))]
            if changed:
                current = trial
                audit['applied'].extend([dict(row=r['row'], field=fld, reason='vat_and_total_mathematics') for r, fld in changed])
        else:
            audit['warnings'].append('Повторное чтение НДС не подтверждено арифметикой, исходные значения сохранены')

    # Numeric ensemble: bounded, independently verified candidate corrections.
    # On failure or when disabled, keep the established numeric fallback below.
    middle, _ = _clean_state(current)
    if middle['status'] in {'mismatch', 'incomplete'} or middle['vat']['status'] == 'mismatch':
        try:
            from pathlib import Path
            import os
            import shutil
            from App.ocr.numeric_fallback_ensemble import (
                recover_mismatched_numbers, load_ensemble_options, optional_easyocr_reader,
            )
            options = load_ensemble_options()
            if options['enabled']:
                executable = options['tesseract_exe']
                if executable == 'tesseract' and not shutil.which(executable):
                    program_files = os.environ.get('ProgramFiles')
                    candidate = Path(program_files) / 'Tesseract-OCR' / 'tesseract.exe' if program_files else None
                    if candidate and candidate.is_file():
                        executable = str(candidate)
                reader = optional_easyocr_reader(Path(__file__).resolve().parents[2], enabled=options['enable_easyocr'])
                current, ensemble = recover_mismatched_numbers(
                    current, image, layout, recognize, ocr_lock=ocr_lock,
                    max_cells=options['max_cells'], max_views=options['max_views'],
                    use_tesseract=options['use_tesseract'], easyocr_reader=reader,
                    require_two_engines=options['require_two_engines'], tesseract_exe=executable,
                )
                audit['ensemble'] = ensemble
                audit['applied'].extend(ensemble['applied'])
                audit['warnings'].extend(ensemble.get('warnings', []))
        except Exception as error:
            audit['ensemble'] = dict(status='error', error=str(error))
            audit['warnings'].append('Дополнительный числовой OCR недоступен; используется штатная проверка')
    middle, _ = _clean_state(current)
    # No grand total printed -> missing_original_total: do not fabricate one.
    if middle['status'] in {'mismatch', 'incomplete', 'missing_original_total'}:
        audit['total_triggered'] = True
        tasks = [t for row in current['rows'] if row.get('kind') in NUMERIC_KINDS
                 for fld in ('quantity', 'price') if (t := task_for(row, fld))]
        total_bounds = _printed_total_crop(image, aligned_lines, layout)
        if total_bounds:
            tasks.append(dict(row=None, kind='total', field='total', bounds=list(total_bounds)))
        proposals = _recognition_pairs(image, tasks, recognize, ocr_lock, cfg['max_images'])
        audit['all_numeric_attempts'] = proposals
        # Do not accept a candidate merely because it could be forced to balance;
        # both different OCR preparations must independently read that exact value.
        trial = _apply_verified(current, proposals)
        check, check_vat = _clean_state(trial)
        if (check['status'] == 'match' and check_vat in {'match', 'not_applicable'}
                and (middle['status'] != 'match' or middle['vat']['status'] != 'match')):
            updated = []
            for proposal in proposals:
                if not proposal['agreed']:
                    continue
                if proposal['field'] == 'total':
                    before = current.get('expected_total', current.get('total_raw', ''))
                    after = trial.get('expected_total', trial.get('total_raw', ''))
                else:
                    old = next(r for r in current['rows'] if r['row'] == proposal['row'])
                    new = next(r for r in trial['rows'] if r['row'] == proposal['row'])
                    field = proposal['field']
                    before, after = old.get(field, old.get(field + '_raw', '')), new.get(field, new.get(field + '_raw', ''))
                if _as_numeric(before) != _as_numeric(after):
                    updated.append(dict(row=proposal['row'], field=proposal['field'],
                                        before=str(before), after=str(after), reason='ocr_consensus_and_full_math'))
            if updated:
                audit['applied'].extend(updated)
                current = trial
        else:
            audit['warnings'].append('Повторное чтение всех цифр не подтвердило итог; значения не изменены')
    final, _ = _clean_state(current)
    audit['final_vat_status'] = final['vat']['status']
    audit['final_total_status'] = final['status']
    audit['applied_count'] = len(audit['applied'])
    return current, audit
