"""Grid-first reconstruction of printed Torgsoft product rows.

Use visible numeric price anchors on the *rectified* image to recover physical
row bands before mapping already-recognized OCR blocks. No expected row count,
price or receipt total is read or inferred from any golden fixture.

This is a conservative fallback: only when the first-pass parser has apparently
merged rows, and only if distinct printed price lines substantiate more rows.
"""
from __future__ import annotations

import copy
import re
from statistics import median
from typing import Any, Callable

import cv2
import numpy as np

VERSION = "grid-first-rows-1.0"


def preserve_source_columns(image, polygon, rectified, map_x, map_y, metadata):
    """Keep an observed source grid when contour warping loses a divider.

    No OCR retry or guessed corner: compare independently detected columns in
    the original bounding crop and the contour result before the first OCR.
    """
    original = (rectified, map_x, map_y, metadata)
    report = dict(status='not_needed')
    if metadata.get('method') != 'contour_coons_patch':
        return original, report
    from App.ocr.table_verification import verify_table_candidate
    def verify(region, margin=16):
        h,w=region.shape[:2]
        return verify_table_candidate(region, dict(polygon=np.array(
            [[margin,margin],[w-margin,margin],[w-margin,h-margin],[margin,h-margin]],np.float32)),None)
    after=verify(rectified)
    if after.get('geometry',{}).get('four_columns'):
        return original,report
    points=np.asarray(polygon,dtype=float)
    lo=np.maximum(0,np.floor(points.min(axis=0)).astype(int)-16)
    hi=np.minimum([image.shape[1],image.shape[0]],np.ceil(points.max(axis=0)).astype(int)+17)
    crop=image[lo[1]:hi[1],lo[0]:hi[0]]
    before=verify(crop)
    geometry=before.get('geometry',{})
    fractions=geometry.get('column_fractions',[])
    if not geometry.get('four_columns') or len(fractions)!=5 or geometry.get('dense_bars'):
        return original,dict(status='not_applied',reason='source_full_grid_not_confirmed')
    corners=np.asarray(metadata.get('source_body_corners',[]),np.float32)
    if corners.shape != (4,2) or not np.isfinite(corners).all():
        return original,dict(status='not_applied',reason='missing_observed_body_corners')
    # A single perspective sampling keeps the complete observed four-corner
    # grid; the contour map can otherwise pull an internal divider to its edge.
    width=float(max(np.linalg.norm(corners[1]-corners[0]),np.linalg.norm(corners[2]-corners[3])))
    height=float(max(np.linalg.norm(corners[3]-corners[0]),np.linalg.norm(corners[2]-corners[1])))
    scale=min(2.2,max(1.,540./width))
    bw,bh=round(width*scale),round(height*scale)
    margin=16
    footer=max(32,round(width*.20*scale))
    destination=np.array([[margin,margin],[margin+bw,margin],
                          [margin+bw,margin+bh],[margin,margin+bh]],np.float32)
    backward=cv2.getPerspectiveTransform(destination,corners)
    yy,xx=np.indices((bh+2*margin+footer,bw+2*margin),dtype=np.float32)
    coordinates=cv2.perspectiveTransform(np.stack([xx,yy],axis=-1),backward)
    mx,my=coordinates[:,:,0],coordinates[:,:,1]
    sampled=cv2.remap(image,mx,my,cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255))
    corrected=verify(sampled)
    if not corrected.get('geometry',{}).get('four_columns'):
        return original,dict(status='not_applied',reason='perspective_grid_not_confirmed')
    revised=dict(method='source_grid_crop',algorithm_version=VERSION,margin=margin,
                 body_size=[bw,bh],warnings=[],source_polygon=points.tolist())
    report=dict(status='applied',reason='source_five_columns_lost_by_contour',before=before,after=after)
    return (sampled,mx,my,revised),report


def _kind(text):
    key = re.sub(r'\s+', '', str(text or '').casefold())
    if re.match(r'^(?:[нh][дd][сc]|налог)(?:\d|%|$)',key):
        return 'tax'
    if 'доставка' in key:
        return 'service'
    if 'итого' in key or 'корыт' in key or 'қорыт' in key:
        return 'total'
    return 'product'


def _number(text: Any):
    value = str(text or '').strip().replace("₸", "")
    value = re.sub(r"(?i)(?:KZT|ТГ|ТЕНГЕ|[ТT])$", "", value).strip()
    # Spaces are legal only as thousands groups. Joining arbitrary OCR words
    # would dangerously convert "1 10 4" into a fabricated price 1104.
    if re.search(r"\s", value) and not re.fullmatch(r"\d{1,3}(?:\s\d{3})+(?:[.,]\d+)?",value):
        return None
    value = re.sub(r"\s+", "", value)
    return value if re.fullmatch(r"\d+(?:[.,]\d+)?", value) else None


def _box(line):
    try:
        a = np.asarray(line.get("eval_box", line.get("box")), dtype=float)
        return a if a.shape == (4, 2) and np.isfinite(a).all() else None
    except (ValueError, TypeError):
        return None


def _center(line):
    box = _box(line)
    return box.mean(axis=0) if box is not None else None


def _group_positions(items, gap):
    groups = []
    for item in sorted(items, key=lambda x: x[0]):
        if not groups or item[0] - np.mean([x[0] for x in groups[-1]]) > gap:
            groups.append([item])
        else:
            groups[-1].append(item)
    return groups


def find_price_anchors(image: np.ndarray, layout: dict, aligned_lines=()) -> tuple[list[float], dict]:
    """Find individual printed price text baselines using OpenCV only.

    Excludes likely printed total using its own OCR/text-position evidence or
    bottom-of-grid spacing, never by knowing the printed total amount.
    """
    h, w = image.shape[:2]
    try:
        q_right, right = float(layout["quantity_right"]), float(layout["right"])
        top = float(layout["body_top"])
        bottom = float(layout["body_bottom"])
    except (KeyError, ValueError, TypeError):
        return [], {"status": "invalid_layout"}
    price_width = right - q_right
    x0 = max(0, min(w, int(q_right + max(4, price_width * .11))))
    x1 = max(0, min(w, int(right - max(5, price_width * .15))))
    y0 = max(0, min(h, int(top + max(7, h * .003))))
    y1 = max(0, min(h, int(bottom - max(7, h * .003))))
    if x1 - x0 < 16 or y1 - y0 < 45:
        return [], {"status": "price_column_too_small"}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    region = gray[y0:y1, x0:x1]
    block = min(31, max(15, ((min(region.shape) // 3) * 2 + 1)))
    dark = cv2.adaptiveThreshold(region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, block, 8)
    count, _, stats, cents = cv2.connectedComponentsWithStats(dark, 8)
    glyphs = []
    min_h = max(6, int(h * .004))
    max_h = max(20, int(h * .027))
    for (x, y, ww, hh, area), (_, cy) in zip(stats[1:], cents[1:]):
        if (2 <= ww <= max(25, int(price_width * .4)) and min_h <= hh <= max_h
                and 7 <= area <= max(250, int(h * .2)) and x > 0
                and x + ww < region.shape[1] - 1):
            glyphs.append((float(cy + y0), int(ww), int(hh)))
    if not glyphs:
        return [], {"status": "no_price_glyphs", "column": [x0, x1]}
    typical_height = float(median(g[2] for g in glyphs))
    groups = _group_positions(glyphs, gap=max(8., typical_height * .95))
    anchors = [float(median(g[0] for g in group)) for group in groups]
    # Total normally lies underneath the last product and has a smaller gap.
    # Prefer explicit OCR geometry when available (not an arithmetic guess).
    total_ys = []
    for line in aligned_lines:
        text = str(line.get("text", ""))
        center = _center(line)
        if center is None:
            continue
        if ("корыт" in text.lower() or "қорыт" in text.lower() or "итого" in text.lower()
                or (center[0] >= q_right and re.search(
                    r"\d+(?:\s\d{3})*[.,]\d{2}\s*(?:[₸ТT])?$", text.strip()))):
            total_ys.append(float(center[1]))
    skipped = []
    # Header glyphs are not product prices, including on short receipts.
    from App.ocr.table_verification import header_role
    for line in aligned_lines:
        center = _center(line)
        if (center is not None and center[0] >= q_right and
                center[1] < top + .4*(bottom-top) and header_role(line.get('text','')) == 'price'):
            near=[a for a in anchors if abs(a-center[1]) <= max(8.,typical_height*1.5)]
            for anchor in near:
                anchors.remove(anchor)
                skipped.append(dict(y=anchor,reason='recognized_price_header'))
    for yc in total_ys:
        near = [a for a in anchors if abs(a - yc) <= max(typical_height * 2, h * .025)]
        if near and yc > top + .70 * (bottom - top):
            v = min(near, key=lambda a: abs(a - yc))
            anchors.remove(v)
            skipped.append(dict(y=v, reason="printed_total_evidence"))
    if len(anchors) >= 5:
        # The table header ("Бағасы") can produce one glyph group above the
        # first product. This is image geometry, not a hard-coded row count.
        early_gap = anchors[1] - anchors[0]
        following_gaps = np.diff(anchors[1:min(len(anchors), 7)])
        typical = float(median(following_gaps)) if len(following_gaps) else early_gap
        if (anchors[0] < top + .07 * (bottom - top) and
                early_gap < typical * .72):
            skipped.append(dict(y=anchors.pop(0), reason="table_header_glyphs"))
    if len(anchors) >= 5:
        gaps = np.diff(anchors)
        previous = float(median(gaps[:-1])) if len(gaps) > 1 else 0
        last = float(gaps[-1])
        if (previous > 0 and last < previous * .65
                and anchors[-1] > bottom - .06 * (bottom - top)):
            skipped.append(dict(y=anchors.pop(), reason="footer_short_gap_and_bottom"))
    return anchors, dict(status="ok", glyphs=len(glyphs), anchors=len(anchors),
                         typical_glyph_height=typical_height, column=[x0, x1],
                         skipped=skipped)


def _printed_boundary(gray, left, right, guess, span, low, high):
    """Refine a between-price midpoint with a dark horizontal number-column rule."""
    h = gray.shape[0]
    x0 = max(0, int(left) + 3)
    x1 = min(gray.shape[1], int(right) - 3)
    if x1 - x0 < 8:
        return guess, False
    radius = max(7, int(span * .49))
    lo = max(int(low + 5), int(guess - radius), 0)
    hi = min(int(high - 5), int(guess + radius) + 1, h)
    if hi <= lo:
        return guess, False
    area = gray[lo:hi, x0:x1].astype(np.float32)
    bg = np.median(gray[:, x0:x1].astype(np.float32), axis=0)
    # A printed horizontal rule spans most of the blank row-number column.
    coverage = np.mean(area < bg[None, :] - 8., axis=1)
    where = int(np.argmax(coverage))
    strength = float(coverage[where])
    candidate = lo + where
    return (float(candidate), True) if strength >= .48 else (float(guess), False)


def price_bands(image, layout, anchors):
    if len(anchors) < 2:
        return [], {"status": "too_few_anchors"}
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    first_gap = anchors[1] - anchors[0]
    last_gap = anchors[-1] - anchors[-2]
    top = max(float(layout["body_top"]), float(anchors[0]) - max(14, first_gap * .35))
    bottom = min(float(layout["body_bottom"]), float(anchors[-1]) + max(14, last_gap * .47))
    boundaries = [top]
    confirmed = 0
    for left, right in zip(anchors, anchors[1:]):
        guess = (left + right) / 2
        candidate, ok = _printed_boundary(gray, layout.get("body_left", 0),
                                          layout["name_left"], guess, right-left, left, right)
        if ok:
            confirmed += 1
        boundaries.append(candidate)
    boundaries.append(bottom)
    if any(b-a < 10 for a,b in zip(boundaries,boundaries[1:])):
        return [], {"status": "invalid_bands"}
    return boundaries, dict(status="ok", horizontal_rule_supported=confirmed,
                            n_internal=len(anchors)-1)


def _read_name(crop, callback):
    if callback is None:
        return "", []
    result = callback(crop)
    lines = []
    for record in result or []:
        if len(record) != 3:
            continue
        box, text, confidence = record
        if str(text).strip() and float(confidence) >= .28:
            bb = np.asarray(box, dtype=float)
            y = float(bb[:,1].mean()) if bb.shape == (4,2) else 0.
            x = float(bb[:,0].mean()) if bb.shape == (4,2) else 0.
            lines.append((y,x,str(text)))
    return " ".join(item[2] for item in sorted(lines)), lines


def _numeric_views(crop):
    if crop.size == 0:
        return []
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    enlarged = cv2.resize(gray, None, fx=2., fy=2., interpolation=cv2.INTER_CUBIC)
    equal = cv2.createCLAHE(clipLimit=2., tileGridSize=(4, 4)).apply(enlarged)
    return [cv2.cvtColor(z, cv2.COLOR_GRAY2BGR) for z in (enlarged, equal)]


def _read_numeric(crop, callback):
    if callback is None:
        return None, []
    views = _numeric_views(crop)
    readings = []
    for view in views:
        values = callback([view])
        if len(values) != 1:
            return None, []
        text, confidence = values[0]
        readings.append((str(text), float(confidence)))
    valid = [_number(v[0]) for v in readings]
    if len(valid) == 2 and valid[0] is not None and valid[0] == valid[1] and min(v[1] for v in readings) >= .35:
        return valid[0], readings
    return None, readings


def _evidence_rows(aligned, boundaries, layout):
    """Assign whole OCR blocks by overlap, never split a text string by guessing."""
    slots = [[] for _ in range(len(boundaries)-1)]
    crossing = [[] for _ in slots]
    for line in aligned:
        box = _box(line)
        if box is None or not str(line.get("text", "")).strip():
            continue
        cx, cy = box.mean(axis=0)
        if not (layout.get("body_left", 0)-4 <= cx < layout["right"]+4):
            continue
        candidates = []
        for i,(top,bottom) in enumerate(zip(boundaries,boundaries[1:])):
            overlap = max(0., min(float(np.max(box[:,1])), bottom) - max(float(np.min(box[:,1])), top))
            if overlap > 0:
                candidates.append((i,overlap))
        if not candidates:
            continue
        main, strongest = max(candidates,key=lambda v:v[1])
        other = sum(v for i,v in candidates if i!=main)
        if other > max(3., strongest*.3):
            for i,_ in candidates:
                crossing[i].append(str(line['text']))
            continue
        if boundaries[main] <= cy <= boundaries[main+1]:
            slots[main].append(line)
        else:
            crossing[main].append(str(line['text']))
    return slots,crossing


def reconstruct_rows(image, layout, aligned, initial_rows, *, read_name=None, read_numeric=None,
                     max_name_crops=6, max_numeric_crops=44, prefer_printed_bands=False):
    """Return (rows, report). Fallback is gated by extra *visible* price lines.

    Does not require sum matching to preserve physical row boundaries. `*_raw`
    always represents first-pass OCR; independent cell readings are in the
    corresponding `name_recovered`, `quantity`, and `price` fields.
    """
    before = copy.deepcopy(initial_rows)
    report = dict(version=VERSION,status="not_needed",initial_rows=len(before),
                  price_anchors=[],bounds=[],name_calls=0,numeric_calls=0,
                  ambiguous_blocks=[],recovered_cells=[],warnings=[])
    anchors, detected = find_price_anchors(image,layout,aligned)
    report['price_detector'] = detected
    report['price_anchors'] = [round(x,1) for x in anchors]
    initial_count=sum(r.get('kind') in ('product','service','tax') for r in before)
    equal_count_refinement=prefer_printed_bands and len(anchors)==initial_count
    if len(anchors) < 3 or (len(anchors) <= initial_count and not equal_count_refinement):
        return before,report
    # Do not replace many rows based on random noise. At least three separated
    # price lines plus near-continuous printed vertical column evidence required.
    if len(anchors) > 120:
        report['status']='rejected_too_many_anchors';return before,report
    boundaries, band_report=price_bands(image,layout,anchors)
    report['band_detection']=band_report
    if not boundaries:
        report['status']='rejected_no_valid_bands';return before,report
    if equal_count_refinement and band_report['horizontal_rule_supported'] < max(2,int(np.ceil((len(anchors)-1)*.8))):
        report['status']='not_needed';return before,report
    # Require some independently seen horizontal separators for long grids.
    if len(anchors) >= 6 and band_report['horizontal_rule_supported'] < 2:
        report['status']='rejected_insufficient_grid_support';return before,report
    slots,crossings=_evidence_rows(aligned,boundaries,layout)
    rows=[]
    for k,(lo,hi) in enumerate(zip(boundaries,boundaries[1:])):
        members=sorted(slots[k], key=lambda l:(float(_center(l)[1]),float(_center(l)[0])))
        columns={key:[] for key in ('name','quantity','price','number')}
        for line in members:
            cx=float(_center(line)[0]);value=str(line['text'])
            if cx < layout['name_left']-2: columns['number'].append(value)
            elif cx < layout['name_right']: columns['name'].append(value)
            elif cx < layout['quantity_right']: columns['quantity'].append(value)
            elif cx < layout['right']+4: columns['price'].append(value)
        def raw(field): return ' '.join(columns[field]).strip()
        entry=dict(row=k+1,source_row=k+1,y_range=[round(lo,2),round(hi,2)],
                   item_number_raw=raw('number'),name_raw=raw('name'),
                   quantity_raw=raw('quantity'),price_raw=raw('price'),
                   kind=_kind(raw('name')),ocr_confidence=float(np.mean([float(l.get('confidence',0)) for l in members if float(_center(l)[0]) >= layout['name_left'] and float(_center(l)[0]) < layout['name_right']])) if columns['name'] else 0.,
                   raw_lines=[{name:copy.deepcopy(l.get(name)) for name in ('text','confidence','box','box_original')}
                              for l in members],grid_first_source='printed_price_anchors')
        if crossings[k]:
            report['ambiguous_blocks'].append(dict(row=k+1,values=crossings[k]))
        rows.append(entry)
    report['bounds']=[round(y,2) for y in boundaries]
    report['status']='segmented_pending_numeric_check'
    # Read only missing/ambiguous physical cells; the original OCR may have a
    # giant name block spanning several rows, so never copy it into every row.
    for row in rows:
        a,b=row['y_range']; top=max(0,int(a+3)); bottom=min(image.shape[0],int(b-3))
        if bottom-top<10:continue
        cx0=int(layout['name_left'])+3;cx1=int(layout['name_right'])-3
        needs_name=(not row['name_raw'].strip() or bool(crossings[row['row']-1]))
        if needs_name and read_name and report['name_calls']<max_name_crops and cx1-cx0>=15:
            crop=image[top:bottom,cx0:cx1]
            text,_=_read_name(crop,read_name);report['name_calls']+=1
            if text:
                row['name_recovered']=text;row['name_recovery_source']='grid_first_cell_ocr'
                report['recovered_cells'].append(dict(row=row['row'],field='name',text=text,source='cell_ocr'))
        for fld,low,high in [('quantity',layout['name_right'],layout['quantity_right']),
                             ('price',layout['quantity_right'],layout['right'])]:
            if _number(row[fld+'_raw']) is not None:continue
            if read_numeric is None or report['numeric_calls']+2>max_numeric_crops:continue
            x0=int(low)+4;x1=int(high)-4
            if x1-x0<12:continue
            value,readings=_read_numeric(image[top:bottom,x0:x1],read_numeric)
            report['numeric_calls']+=2
            if value is not None:
                row[fld]=value;row[fld+'_source']='grid_first_cell_ocr'
                row[fld+'_ocr']={'readings':readings,'bounds':[x0,top,x1,bottom]}
                report['recovered_cells'].append(dict(row=row['row'],field=fld,text=value,source='cell_ocr'))
    return rows, report
