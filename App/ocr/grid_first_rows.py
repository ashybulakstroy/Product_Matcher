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

VERSION = "grid-first-rows-1.1"


def recover_printed_boundary(image, previous, verification, recognize_header=None):
    """Replace an unverified paper outline with an independently observed grid."""
    report=dict(status='not_needed',ocr_calls=0)
    if (verification.get('status')!='uncertain' or
            verification.get('geometry',{}).get('four_columns') is not False or
            verification.get('header',{}).get('roles') or previous.get('polygon') is None):
        return previous,report
    from App.ocr.receipt_roi_recovery import find_paper_roi
    from App.ocr.full_grid_boundary import detect_full_grid_boundary
    from App.ocr.table_verification import verify_table_candidate
    points=np.asarray(previous['polygon'],dtype=float)
    lo,hi=points.min(axis=0),points.max(axis=0)
    if hi[1]-lo[1] < 1.6*(hi[0]-lo[0]):
        return previous,report
    box,paper=find_paper_roi(image)
    report.update(status='not_recovered',paper=paper)
    if box is None:return previous,report
    x0,y0,x1,y1=box
    crop=image[y0:y1,x0:x1]
    if max(crop.shape[:2])*2>2400:return previous,dict(report,reason='paper_roi_too_large')
    sample=cv2.resize(crop,None,fx=2.,fy=2.,interpolation=cv2.INTER_CUBIC)
    polygon,evidence=detect_full_grid_boundary(sample)
    report['grid_evidence']=evidence
    if polygon is None:return previous,report
    polygon=polygon/2+np.array([x0,y0],np.float32)
    a,b=polygon.min(axis=0),polygon.max(axis=0)
    if (b[1]-a[1] >= .8*(hi[1]-lo[1]) or b[0]-a[0] < .65*(hi[0]-lo[0]) or
            np.any(a<lo-8) or np.any(b>hi+8)):
        return previous,dict(report,reason='not_an_inner_printed_grid')
    candidate=dict(polygon=polygon,status='detected',warnings=['paper_outline_replaced_by_observed_grid'])
    checked=verify_table_candidate(image,candidate,None)
    if checked.get('geometry',{}).get('four_columns'):
        candidate['selected']=dict(support=[evidence['rule_count'],5],source='paper_roi_observed_grid')
        checked=verify_table_candidate(image,candidate,None)
    if (checked.get('status') == 'uncertain' and checked.get('geometry',{}).get('four_columns')
            and recognize_header is not None):
        checked=verify_table_candidate(image,candidate,recognize_header)
        report['ocr_calls']=1
    report['verification']=checked
    if checked.get('status')!='confirmed':return previous,report
    report.update(status='recovered',reason='paper_outline_contains_confirmed_printed_grid')
    return candidate,report


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


def _wrapped_numeric(lines, left, right):
    """Read a two-line number that physically wraps at the column's right edge.

    Join observed characters only, never amounts inferred from receipt math.
    Ordinary separated numbers and unrelated OCR fragments are not eligible.
    """
    if len(lines) != 2:
        return None
    ordered=sorted(lines,key=lambda l:float(_center(l)[1]))
    boxes=[_box(line) for line in ordered]
    if any(box is None for box in boxes):
        return None
    first,last=[str(line['text']).strip() for line in ordered]
    if not re.fullmatch(r'\d[\d .,]*',first) or not re.fullmatch(r'\d{1,2}',last):
        return None
    a,b=boxes
    gap=float(b[:,1].min()-a[:,1].max())
    height=float(np.median([np.ptp(box[:,1]) for box in boxes]))
    if (not -2 <= gap <= height*.7 or
            abs(float(a[:,0].max()-b[:,0].max())) > 3 or
            right-float(a[:,0].max()) > 5 or
            float(np.ptp(a[:,0])) < (right-left)*.65):
        return None
    combined=first+last
    return combined if _number(combined) is not None else None




def find_printed_grid_bands(image: np.ndarray, layout: dict, aligned_lines=(), *, header_verified=False):
    """Detect independent horizontal rules in number AND price columns.

    A short, sharply printed grid is much stronger evidence than OCR price
    strings. Only use it when the header is actually seen in original OCR
    blocks; never assume that the first band is a heading without evidence.
    """
    try:
        left = float(layout.get('body_left', 0))
        n_end = float(layout['name_left'])
        p_start, p_end = float(layout['quantity_right']), float(layout['right'])
        body_top, body_bottom = float(layout['body_top']), float(layout['body_bottom'])
    except (KeyError, ValueError, TypeError):
        return [], dict(status='invalid_layout')
    h, w = image.shape[:2]
    if h < 100 or n_end-left < 16 or p_end-p_start < 22:
        return [], dict(status='too_small')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    # Keep narrow spans away from vertical table dividers.
    spans = [(int(left+max(3,(n_end-left)*.09)), int(n_end-max(3,(n_end-left)*.09))),
             (int(p_start+max(4,(p_end-p_start)*.09)), int(p_end-max(4,(p_end-p_start)*.09)))]
    scores=[]
    for x0,x1 in spans:
        x0=max(0,min(w,x0));x1=max(0,min(w,x1))
        if x1-x0 < 10:
            return [], dict(status='invalid_column_width')
        roi=gray[:,x0:x1]
        block=min(31, max(15, ((min(roi.shape)//3)*2+1)))
        inverse=cv2.adaptiveThreshold(roi,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV,block,5)
        horizontal=cv2.morphologyEx(inverse,cv2.MORPH_CLOSE,np.ones((1,max(5,(x1-x0)//14)),np.uint8))
        horizontal=cv2.morphologyEx(horizontal,cv2.MORPH_OPEN,
                                   np.ones((1,max(12,int((x1-x0)*.27))),np.uint8))
        scores.append(np.count_nonzero(horizontal,axis=1)/float(x1-x0))
    support=(scores[0] >= .72)&(scores[1] >= .72)
    mask=np.flatnonzero(support & (np.arange(h) >= max(0,body_top-4))
                       & (np.arange(h) <= min(h-1,body_bottom+4)))
    clusters=[]
    for y in mask:
        if not clusters or y-clusters[-1][-1] > max(7,round(h*.008)):
            clusters.append([int(y)])
        else: clusters[-1].append(int(y))
    rules=[float(np.median(c)) for c in clusters]
    rules=[v for v in rules if body_top-4 <= v <= body_bottom+4]
    report=dict(status='insufficient_printed_rules', rules=[round(v,2) for v in rules],
                strip_ranges=spans, supports='number_and_price')
    if not 5 <= len(rules) <= 60:
        return [], report
    # Find header words strictly in original OCR geometry. If the heading is
    # unreadable, leave this strategy off and try the old price-anchor fallback.
    header_words=('тауар', 'саны', 'багас', 'бағас', 'цена', 'наимен', 'кол-во')
    header_ys=[]
    for line in aligned_lines:
        center=_center(line)
        if center is not None and any(word in str(line.get('text','')).casefold() for word in header_words):
            header_ys.append(float(center[1]))
    if not header_ys and not header_verified:
        report['status']='no_header_evidence';return [],report
    # A header may be broken across OCR blocks. Identify a physical interval
    # with the heading text, then start after the separator beneath it.
    header_idx=None
    for idx,(a,b) in enumerate(zip(rules,rules[1:])):
        if any(a+3 <= y <= b-3 for y in header_ys) and idx <= 2:
            header_idx=idx;break
    if header_idx is None and header_verified:
        # Upstream App/ocr/table_verification has independently confirmed the
        # first printed header; no additional OCR call is necessary.
        if rules[0] <= body_top + max(10.,(body_bottom-body_top)*.06):
            header_idx=0
    if header_idx is None:
        report['status']='header_outside_grid';return [],report
    bounds=rules[header_idx+1:]
    if not 4 <= len(bounds)-1 <= 50:
        report['status']='too_few_product_bands';return [],report
    if any(b-a < 12 for a,b in zip(bounds,bounds[1:])):
        report['status']='narrow_band';return [],report
    # Reject very wide gaps which imply a missing printed separator.
    gaps=np.diff(bounds)
    if gaps.size>=4 and max(gaps) > max(200.,float(np.median(gaps))*3.1):
        report['status']='missing_printed_separator';return [],report
    report.update(status='ok',header_band=[rules[header_idx],rules[header_idx+1]],
                  row_bands=len(bounds)-1,all_rules=len(rules))
    return bounds,report


def _batch_numeric_read(tasks,callback,report):
    """One batched OCR recognizer call for missing numeric cell views."""
    images=[]
    for task in tasks:
        views=_numeric_views(task['crop'])
        if len(views)==2:
            task['offset']=len(images)
            images.extend(views)
    if not images or callback is None:return
    report['numeric_calls'] += 1
    report['numeric_images'] += len(images)
    try:
        readings=callback(images)
    except Exception as error:
        report['warnings'].append('numeric_batch_failed:'+type(error).__name__)
        return
    if len(readings)!=len(images):
        report['warnings'].append('numeric_batch_output_count_mismatch')
        return
    for task in tasks:
        if 'offset' not in task:continue
        start=task['offset']
        pair=[]
        for result in readings[start:start+2]:
            try:
                text,confidence=result
                pair.append((str(text),float(confidence)))
            except (TypeError,ValueError):
                pair=[];break
        valid=[_number(item[0]) for item in pair]
        if len(valid)==2 and valid[0] is not None and valid[0]==valid[1] and min(x[1] for x in pair)>=.35:
            row=task['row'];field=task['field']
            row[field]=valid[0]
            row[field+'_source']='grid_first_cell_ocr'
            row[field+'_ocr']={'readings':pair,'bounds':task['bounds']}
            report['recovered_cells'].append(dict(row=row['row'],field=field,text=valid[0],source='cell_ocr'))
        else:
            report['uncertain_cells'].append(dict(row=task['row']['row'],field=task['field'],readings=pair))


def reconstruct_rows(image, layout, aligned, initial_rows, *, read_name=None, read_numeric=None,
                     max_name_crops=12, max_numeric_crops=44, header_verified=False, prefer_printed_bands=False):
    """Recover physical rows only on independent geometric evidence.

    Printed grid first (short receipts), price glyphs second (long receipts).
    No expected total, VAT amount, or expected row count is consulted.
    """
    before=copy.deepcopy(initial_rows)
    report=dict(version=VERSION,status='not_needed',initial_rows=len(before),
                strategy=None,price_anchors=[],bounds=[],name_calls=0,numeric_calls=0,
                numeric_images=0,ambiguous_blocks=[],recovered_cells=[],uncertain_cells=[],warnings=[])
    payable=sum(r.get('kind') in ('product','service','tax') for r in before)
    bounds,grid=find_printed_grid_bands(image,layout,aligned,header_verified=header_verified)
    report['printed_grid_detector']=grid
    payable_rows=[r for r in before if r.get('kind') in ('product','service','tax')]
    equal_printed_count = bool(bounds) and len(bounds)-1 == payable
    needs_printed_refinement = equal_printed_count and (
        any(_number(r.get(f, r.get(f+'_raw'))) is None
            for r in payable_rows for f in ('quantity','price')) or
        any(len(r.get('y_range', [])) != 2 or
            max(abs(float(a)-float(b)) for a,b in zip(r['y_range'], band)) > 3
            for r,band in zip(payable_rows,zip(bounds,bounds[1:]))))
    if bounds and (len(bounds)-1 > payable or needs_printed_refinement):
        strategy='printed_grid'
        report['strategy']=strategy
        report['price_detector']={'status':'not_needed'}
    else:
        anchors,det=find_price_anchors(image,layout,aligned)
        report['price_detector']=det
        report['price_anchors']=[round(x,1) for x in anchors]
        equal_count_refinement=prefer_printed_bands and len(anchors)==payable
        if len(anchors)<3 or (len(anchors)<=payable and not equal_count_refinement) or len(anchors)>120:
            return before,report
        bounds,band_report=price_bands(image,layout,anchors)
        report['band_detection']=band_report
        if not bounds:
            report['status']='rejected_no_valid_bands';return before,report
        if len(anchors)>=6 and band_report['horizontal_rule_supported']<2:
            report['status']='rejected_insufficient_grid_support';return before,report
        if equal_count_refinement and band_report['horizontal_rule_supported'] < max(2,int(np.ceil((len(anchors)-1)*.8))):
            report['status']='not_needed';return before,report
        strategy='price_anchors'
        report['strategy']=strategy
    slots,crossings=_evidence_rows(aligned,bounds,layout)
    rows=[]
    for k,(lo,hi) in enumerate(zip(bounds,bounds[1:])):
        members=sorted(slots[k],key=lambda l:(float(_center(l)[1]),float(_center(l)[0])))
        columns={key:[] for key in ('name','quantity','price','number')}
        for line in members:
            cx=float(_center(line)[0]);value=str(line['text'])
            if cx<layout['name_left']-2:columns['number'].append(value)
            elif cx<layout['name_right']:columns['name'].append(value)
            elif cx<layout['quantity_right']:columns['quantity'].append(value)
            elif cx<layout['right']+4:columns['price'].append(value)
        def raw(field):return ' '.join(columns[field]).strip()
        name_confidence=[float(l.get('confidence',0)) for l in members
                         if layout['name_left']<=float(_center(l)[0])<layout['name_right']]
        row=dict(row=k+1,source_row=k+1,y_range=[round(lo,2),round(hi,2)],
                 item_number_raw=raw('number'),name_raw=raw('name'),
                 quantity_raw=raw('quantity'),price_raw=raw('price'),
                 kind=_kind(raw('name')),ocr_confidence=float(np.mean(name_confidence)) if name_confidence else 0.,
                 raw_lines=[{name:copy.deepcopy(l.get(name)) for name in ('text','confidence','box','box_original')}
                            for l in members],grid_first_source=strategy)
        if crossings[k]:report['ambiguous_blocks'].append(dict(row=k+1,values=crossings[k]))
        for field,left,right in [('quantity',layout['name_right'],layout['quantity_right']),
                                 ('price',layout['quantity_right'],layout['right'])]:
            if _number(row[field+'_raw']) is not None:
                continue
            blocks=[line for line in members if left <= float(_center(line)[0]) < right+1]
            wrapped=_wrapped_numeric(blocks,left,right)
            if wrapped is not None:
                row[field]=wrapped
                report['recovered_cells'].append(dict(row=row['row'],field=field,text=wrapped,
                    source='wrapped_ocr_blocks', evidence=[dict(text=l['text'],box=_box(l).tolist()) for l in blocks]))
        rows.append(row)
    report['bounds']=[round(y,2) for y in bounds]
    report['status']='segmented_pending_numeric_check'
    if strategy == 'printed_grid':
        # The interface and later cell crops must use the same observed grid
        # as reconstruction, including its independently verified header band.
        layout['row_boundaries']=[float(grid['header_band'][0]), *bounds]
    pending=[]
    for i,row in enumerate(rows):
        a,b=row['y_range'];top=max(0,int(a+3));bottom=min(image.shape[0],int(b-3))
        if bottom-top<10:continue
        name_left,name_right=int(layout['name_left'])+3,int(layout['name_right'])-3
        needs_name=(not row['name_raw'].strip() or bool(crossings[i]))
        if needs_name and read_name and report['name_calls']<max_name_crops and name_right-name_left>=15:
            crop=image[top:bottom,name_left:name_right]
            try:
                text,_=_read_name(crop,read_name)
                report['name_calls']+=1
            except Exception as error:
                text=''
                report['warnings'].append('name_read_failed:'+type(error).__name__)
            if text:
                row['name_recovered']=text;row['name_recovery_source']='grid_first_cell_ocr'
                if _kind(text)=='tax':row['kind']='tax'
                elif _kind(text)=='service':row['kind']='service'
                report['recovered_cells'].append(dict(row=row['row'],field='name',text=text,source='cell_ocr'))
        for field,left,right in [('quantity',layout['name_right'],layout['quantity_right']),
                                 ('price',layout['quantity_right'],layout['right'])]:
            if _number(row.get(field,row[field+'_raw'])) is not None:continue
            if read_numeric is None or 2*(len(pending)+1)>max_numeric_crops:continue
            x0=max(0,int(left)+4);x1=min(image.shape[1],int(right)-4)
            if x1-x0<12:continue
            pending.append(dict(row=row,field=field,bounds=[x0,top,x1,bottom],
                                crop=image[top:bottom,x0:x1].copy()))
    _batch_numeric_read(pending,read_numeric,report)
    report['unresolved_numeric']=[dict(row=r['row'],field=f)
        for r in rows for f in ('quantity','price')
        if _number(r.get(f,r[f+'_raw'])) is None]
    return rows,report
