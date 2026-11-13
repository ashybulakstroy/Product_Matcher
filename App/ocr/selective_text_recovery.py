"""Selective name repair from existing OCR blocks; then local RapidOCR/Tesseract.

No change to name_raw, raw_lines, quantity, price, VAT or row geometry.
Tesseract is a local optional executable, not downloaded by the web server.
"""
from __future__ import annotations

from contextlib import nullcontext
from difflib import SequenceMatcher
from pathlib import Path
from io import StringIO
import copy
import csv
import json
import math
import os
import re
import shutil
import subprocess

import cv2
import numpy as np

VERSION = 'selective-text-recovery-1.0'


def _normal(value):
    return re.sub(r'\s+', ' ', str(value or '')).strip()


def _key(value):
    return _normal(value).casefold().replace('ё', 'е')


def _word_tokens(value):
    return re.findall(r'[а-яёa-z0-9]+(?:[-/][а-яёa-z0-9]+)*', _key(value))


def _confidence(value):
    try:
        f = float(value)
        return max(0.0, min(1.0, f)) if math.isfinite(f) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _shape(raw):
    try:
        box = np.asarray(raw, dtype=np.float32)
        if box.shape == (4, 2) and np.all(np.isfinite(box)):
            return box
    except (TypeError, ValueError):
        pass
    return None


def _line_from_ocr(item):
    box = _shape(item.get('eval_box', item.get('box')))
    if box is None:
        return None
    return {'text': _normal(item.get('text')), 'box': box,
            'x': float(np.mean(box[:, 0])), 'y': float(np.mean(box[:, 1])),
            'confidence': _confidence(item.get('confidence', 0.0))}


def _line_groups(items):
    """Join fragments left-right within a printed line, then top-to-bottom."""
    groups = []
    for item in sorted(items, key=lambda x: (x['y'], x['x'])):
        box = item['box']
        h = max(1.0, float(np.ptp(box[:, 1])))
        match = None
        for group in groups:
            if abs(group['y']-item['y']) <= min(9.0, max(2.0, .43 * (group['h']+h))):
                match = group
                break
        if match is None:
            groups.append({'y': item['y'], 'h': h, 'items': [item]})
        else:
            match['items'].append(item)
            match['y'] = sum(v['y'] for v in match['items']) / len(match['items'])
            match['h'] = max(match['h'], h)
    groups.sort(key=lambda g: g['y'])
    return _normal(' '.join(' '.join(v['text'] for v in sorted(g['items'], key=lambda x: x['x']))
                             for g in groups))


def _existing_name_blocks(row, aligned_lines, layout):
    y_range = row.get('y_range')
    if not isinstance(y_range, (list, tuple)) or len(y_range) != 2:
        return []
    try:
        top, bottom = float(y_range[0]), float(y_range[1])
        left, right = float(layout['name_left']), float(layout['name_right'])
    except (TypeError, ValueError, KeyError):
        return []
    if bottom <= top or right <= left:
        return []
    out = []
    for original in aligned_lines:
        item = _line_from_ocr(original)
        if item is None or not item['text']:
            continue
        box = item['box']
        bx0, bx1 = float(box[:, 0].min()), float(box[:, 0].max())
        by0, by1 = float(box[:, 1].min()), float(box[:, 1].max())
        # Prevent accidental transfer of a neighbor row's words or numbers.
        overlap_x = max(0., min(bx1, right-2) - max(bx0, left+2))
        overlap_y = max(0., min(by1, bottom-2) - max(by0, top+2))
        x_fraction = overlap_x / max(1., bx1-bx0)
        y_fraction = overlap_y / max(1., by1-by0)
        if (top <= item['y'] < bottom and left+2 <= item['x'] < right-2
                and x_fraction >= .72 and y_fraction >= .65):
            out.append(item)
    return out


def _preserves_words(old, new):
    """Preserve informative old OCR tokens, including exact numeric model codes."""
    # Short numeric codes and single Latin designators are significant too.
    # The original implementation protected only tokens of length >= 4.
    old_numbers, old_designators, _ = _critical_readings(old)
    new_numbers, new_designators, _ = _critical_readings(new)
    for old_values, new_values in ((old_numbers, new_numbers), (old_designators, new_designators)):
        remaining = iter(new_values)
        if any(not any(value == candidate for candidate in remaining) for value in old_values):
            return False
    old_words = [x for x in _word_tokens(old) if len(x) >= 4]
    new_words = _word_tokens(new)
    if not old_words:
        return True
    matched = 0
    for tok in old_words:
        if any(tok == other or (not any(c.isdigit() for c in tok) and
                                SequenceMatcher(None, tok, other).ratio() >= .85)
               for other in new_words):
            matched += 1
        elif any(c.isdigit() for c in tok):
            return False
    return matched / len(old_words) >= .8


def _damaged(text, conf):
    words = _word_tokens(text)
    isolated = 0
    for index, token in enumerate(words):
        if len(token) != 1 or not token.isalpha():
            continue
        if token in {'в', 'с', 'к', 'у', 'и', 'о', 'а'}:
            continue  # Ordinary Russian one-letter words, not broken glyphs.
        if token in {'м', 'л', 'г', 'т', 'm', 'l', 'g', 't'} and index and any(
                character.isdigit() for character in words[index-1]):
            continue  # E.g. "100 м в бухте" is a clean product description.
        isolated += 1
    reasons = []
    if not _normal(text) or len(_normal(text)) < 7:
        reasons.append('empty_or_very_short_name')
    if isolated >= 2:
        reasons.append('fragmented_single_letters')
    if conf < .48:
        reasons.append('low_original_confidence')
    return reasons


def _text_better(old, new):
    if not new or _key(new) == _key(old):
        return False
    if not _preserves_words(old, new):
        return False
    if len(_normal(new)) < max(4, int(len(_normal(old)) * .85)):
        return False
    old_reasons = _damaged(old, 1.)
    new_reasons = _damaged(new, 1.)
    if 'fragmented_single_letters' in old_reasons and 'fragmented_single_letters' not in new_reasons:
        return True
    if 'empty_or_very_short_name' in old_reasons and len(_normal(new)) > len(_normal(old)) + 4:
        return True
    return len(_normal(new)) >= len(_normal(old)) + 6 and \
        SequenceMatcher(None, _key(old), _key(new)).ratio() >= .62


def _crop_name(image, row, layout):
    band = row.get('y_range')
    if not isinstance(band, (list, tuple)) or len(band) != 2:
        return None, None
    h, w = image.shape[:2]
    try:
        left, right = float(layout['name_left']), float(layout['name_right'])
        top, bottom = map(float, band)
    except (TypeError, ValueError, KeyError):
        return None, None
    x0, x1 = max(0, math.floor(left)+3), min(w, math.ceil(right)-3)
    y0, y1 = max(0, math.floor(top)+3), min(h, math.ceil(bottom)-3)
    if x1-x0 < 24 or y1-y0 < 12:
        return None, None
    return image[y0:y1, x0:x1].copy(), [x0, y0, x1, y1]


def _views(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    h, w = gray.shape[:2]
    # Two alternatives maximum; don't enlarge the entire photographed receipt.
    scale = min(3., max(1.5, 48. / max(1., h)))
    scaled = cv2.resize(gray, (round(w*scale), round(h*scale)), interpolation=cv2.INTER_CUBIC)
    normal = cv2.copyMakeBorder(scaled, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
    clahe = cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8)).apply(scaled)
    contrast = cv2.copyMakeBorder(clahe, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
    return [cv2.cvtColor(v, cv2.COLOR_GRAY2BGR) for v in (normal, contrast)]


def _rapid_read(img, recognize, ocr_lock):
    if recognize is None:
        return {'status': 'unavailable', 'engine': 'rapidocr'}
    try:
        with (ocr_lock if ocr_lock is not None else nullcontext()):
            output = recognize(img)
        # Compatible with provided callback returning only RapidOCR's lines,
        # and with RapidOCR's (lines, phase_times) API.
        if isinstance(output, tuple) and len(output) == 2:
            output = output[0]
        if not output:
            return {'status': 'empty', 'engine': 'rapidocr', 'text': '', 'confidence': 0.}
        values = []
        for entry in output:
            if not isinstance(entry, (tuple, list)) or len(entry) < 3:
                continue
            box, text, conf = entry[:3]
            parsed = _line_from_ocr({'box': box, 'text': text, 'confidence': conf})
            if parsed is not None:
                values.append(parsed)
        if not values:
            return {'status': 'empty', 'engine': 'rapidocr', 'text': '', 'confidence': 0.}
        return {'status': 'ok', 'engine': 'rapidocr', 'text': _line_groups(values),
                'confidence': round(sum(v['confidence'] for v in values)/len(values), 4)}
    except Exception as exc:
        return {'status': 'error', 'engine': 'rapidocr', 'detail': str(exc)[:280]}


def _tesseract_bin(explicit=None):
    if explicit:
        return str(explicit) if Path(str(explicit)).is_file() else shutil.which(str(explicit))
    directories = [os.environ.get(key) for key in ('ProgramFiles', 'ProgramFiles(x86)')]
    return shutil.which('tesseract') or next((str(Path(directory) / 'Tesseract-OCR/tesseract.exe')
        for directory in directories if directory and
        (Path(directory) / 'Tesseract-OCR/tesseract.exe').is_file()), None)


def _tesseract_read(img, executable=None, languages='rus+eng', timeout=12):
    exe = _tesseract_bin(executable)
    if exe is None:
        return {'engine': 'tesseract', 'status': 'binary_missing'}
    ok, encoded = cv2.imencode('.png', img)
    if not ok:
        return {'engine': 'tesseract', 'status': 'encode_error'}
    try:
        command = [exe, 'stdin', 'stdout', '-l', languages, '--psm', '6', '-c', 'tessedit_create_tsv=1']
        models = Path(__file__).resolve().parents[2] / 'models/tesseract'
        if all((models / (lang + '.traineddata')).is_file() for lang in languages.split('+')):
            command.extend(['--tessdata-dir', str(models)])
        run = subprocess.run(command,
                             input=encoded.tobytes(), stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, timeout=timeout, check=False)
        if run.returncode != 0:
            return {'engine': 'tesseract', 'status': 'error',
                    'detail': run.stderr.decode('utf-8', errors='replace')[-300:]}
        data = run.stdout.decode('utf-8-sig', errors='replace')
        blocks = {}
        confidences = []
        for rec in csv.DictReader(StringIO(data), delimiter='\t'):
            word = _normal(rec.get('text', ''))
            try:
                conf = float(rec.get('conf', '-1'))
            except (ValueError, TypeError):
                conf = -1.
            if not word or conf < 0:
                continue
            key = (rec.get('block_num'), rec.get('par_num'), rec.get('line_num'))
            blocks.setdefault(key, []).append(word)
            confidences.append(conf/100.)
        text = _normal(' '.join(' '.join(tokens) for tokens in blocks.values()))
        return {'engine': 'tesseract', 'status': 'ok' if text else 'empty', 'text': text,
                'confidence': round(sum(confidences)/len(confidences), 4) if confidences else 0.}
    except subprocess.TimeoutExpired:
        return {'engine': 'tesseract', 'status': 'timeout'}
    except OSError as exc:
        return {'engine': 'tesseract', 'status': 'error', 'detail': str(exc)[:280]}


def _critical_readings(text):
    """Model numbers and single Latin designators must match between engines.

    Similar text with changed 12 -> 13 or N -> M is NOT a safe consensus.
    """
    text = _normal(text)
    numeric = re.findall(r'\d+(?:[.,]\d+)*', text)
    latin_designators = re.findall(r'(?<![A-Za-z])\b[A-Z]\b(?![A-Za-z])', text)
    mixed_codes = re.findall(r'(?<!\w)[\w/-]*\d[\w/-]*(?!\w)', text)
    return numeric, latin_designators, [v.casefold() for v in mixed_codes]


def _best_consensus(original, rapid_reads, tess_reads, min_conf):
    """Promote only independently corroborated text, never catalog-guess it."""
    options = []
    for rapid in rapid_reads:
        if rapid.get('status') != 'ok' or rapid.get('confidence', 0.) < min_conf:
            continue
        for tess in tess_reads:
            if tess.get('status') != 'ok' or tess.get('confidence', 0.) < min_conf:
                continue
            rtext, ttext = _normal(rapid.get('text')), _normal(tess.get('text'))
            if not rtext or not ttext:
                continue
            agreement = SequenceMatcher(None, _key(rtext), _key(ttext)).ratio()
            if agreement < .84 or _critical_readings(rtext) != _critical_readings(ttext):
                continue
            selected = rtext if len(rtext) >= len(ttext) else ttext
            if not _text_better(original, selected):
                continue
            if SequenceMatcher(None, _key(original), _key(selected)).ratio() < .55 and _normal(original):
                continue
            score = agreement + min(rapid['confidence'], tess['confidence']) * .3
            options.append((score, selected, round(agreement, 4)))
    if not options:
        return None
    options.sort(reverse=True)
    # Two materially different, equally supported readings are ambiguous.
    if len(options) > 1 and options[0][0] - options[1][0] < .03 and \
            SequenceMatcher(None, _key(options[0][1]), _key(options[1][1])).ratio() < .94:
        return None
    return {'text': options[0][1], 'agreement': options[0][2]}


def _read_settings(root):
    defaults = {'enabled': True, 'max_rows': 3, 'min_confidence': .60,
                'languages': 'rus+eng', 'tesseract_exe': None}
    try:
        from dotenv import dotenv_values
        values = dotenv_values(Path(root) / '.env', encoding='utf-8-sig')
    except ImportError:
        return defaults
    def flag(key, default):
        return default if values.get(key) in (None, '') else str(values[key]).strip() not in ('0', 'false', 'False')
    defaults['enabled'] = flag('OCR_TEXT_RECOVERY_ENABLED', True)
    try:
        defaults['max_rows'] = min(10, max(0, int(values.get('OCR_TEXT_RECOVERY_MAX_ROWS') or 3)))
        defaults['min_confidence'] = min(1., max(0., float(values.get('OCR_TEXT_RECOVERY_MIN_CONFIDENCE') or .60)))
    except ValueError as exc:
        raise ValueError('Invalid OCR_TEXT_RECOVERY_* setting in root .env') from exc
    defaults['languages'] = str(values.get('OCR_TEXT_RECOVERY_LANG') or 'rus+eng')
    defaults['tesseract_exe'] = values.get('OCR_TEXT_TESSERACT_EXE') or None
    return defaults


def recover_text_names(rows, aligned_lines, image, layout, *, rapidocr=None,
                       ocr_lock=None, settings=None, output_dir=None, only_rows=None):
    """Return (deep-copied rows, diagnostic report), no source OCR is overwritten.

    Stage 1: inspect original blocks for every product row; no OCR calls.
    Stage 2: strictly bounded retry for suspicious still-damaged rows only.
    """
    cfg = dict(settings or _read_settings(Path(__file__).resolve().parents[2]))
    result = copy.deepcopy(rows)
    report = {'version': VERSION, 'inspected': 0, 'stage1_applied': [],
              'stage2_attempts': [], 'stage2_applied': [], 'skipped': [],
              'rapidocr_calls': 0, 'tesseract_calls': 0}
    output = Path(output_dir) if output_dir is not None else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
    if not cfg.get('enabled', True):
        report['status'] = 'disabled'
        if output is not None:
            (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        return result, report
    pending = []
    chosen_rows = set(map(int, only_rows)) if only_rows is not None else None
    for row in result:
        if row.get('kind') != 'product' or (chosen_rows is not None and row.get('row') not in chosen_rows):
            continue
        report['inspected'] += 1
        if row.get('name_confirmed') or row.get('name_origin') in {'user', 'manual'}:
            report['skipped'].append({'row': row['row'], 'reason': 'confirmed_name'})
            continue
        old = _normal(row.get('name_raw'))
        blocks = _existing_name_blocks(row, aligned_lines, layout)
        rebuilt = _line_groups(blocks)
        if rebuilt and _text_better(old, rebuilt):
            row.update(name_recovered=rebuilt, name_recovery_source='original_ocr_blocks',
                       name_recovery_status='accepted')
            report['stage1_applied'].append({'row': row['row'], 'original': old, 'recovered': rebuilt,
                                             'block_count': len(blocks)})
            continue  # No second OCR: the original blocks contained the words.
        reasons = _damaged(old, _confidence(row.get('ocr_confidence')))
        if reasons:
            pending.append((row, reasons))
        else:
            report['skipped'].append({'row': row['row'], 'reason': 'clean_original'})
    if not cfg.get('enabled', True):
        report['status'] = 'disabled'
    else:
        limit = cfg.get('max_rows', 3)
        pending.sort(key=lambda r: (-len(r[1]), _confidence(r[0].get('ocr_confidence', 0))))
        for row, reasons in pending[:limit]:
            original = _normal(row.get('name_raw'))
            crop, bounds = _crop_name(image, row, layout)
            item = {'row': row['row'], 'reasons': reasons, 'original': original, 'bounds': bounds,
                    'rapidocr': [], 'tesseract': [], 'applied': False}
            if crop is None:
                item['status'] = 'invalid_crop'
                report['stage2_attempts'].append(item)
                continue
            variants = _views(crop)
            if output is not None:
                for i, variant in enumerate(variants, 1):
                    cv2.imwrite(str(output / f'row_{row["row"]:03d}_name_view{i}.png'), variant)
            for variant in variants:
                r = _rapid_read(variant, rapidocr, ocr_lock)
                item['rapidocr'].append(r)
                report['rapidocr_calls'] += int(rapidocr is not None)
            for variant in variants:
                t = _tesseract_read(variant, cfg.get('tesseract_exe'),
                                    cfg.get('languages', 'rus+eng'))
                item['tesseract'].append(t)
                report['tesseract_calls'] += int(t.get('status') not in {'binary_missing'})
            accepted = _best_consensus(original, item['rapidocr'], item['tesseract'],
                                       cfg.get('min_confidence', .60))
            if accepted is not None:
                row.update(name_recovered=accepted['text'], name_recovery_source='rapidocr+tesseract',
                           name_recovery_status='accepted', name_recovery_confidence=accepted['agreement'])
                item.update(status='accepted', recovered=accepted['text'], agreement=accepted['agreement'], applied=True)
                report['stage2_applied'].append({'row': row['row'], 'original': original,
                                                  'recovered': accepted['text']})
            else:
                item['status'] = 'unconfirmed'
                row['name_recovery_status'] = 'needs_review'
            report['stage2_attempts'].append(item)
        for row, reasons in pending[limit:]:
            report['skipped'].append({'row': row['row'], 'reason': 'retry_limit', 'reasons': reasons})
        report['status'] = 'ok'
    if output is not None:
        (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return result, report
