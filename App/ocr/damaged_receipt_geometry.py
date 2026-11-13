"""One-shot OpenCV fallback for broken, short Torgsoft grids.

Runs only after established detectors/verification fail.  All accepted polygon
corners are anchored to observed printed horizontal and vertical segments;
never infer a price, text or a table from white paper alone.
"""
from __future__ import annotations
import cv2
import numpy as np

VERSION = 'damaged-receipt-grid-1.0'


def _segments(gray):
    clahe = cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8)).apply(gray)
    binary = cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 12)
    threshold = max(25, int(gray.shape[1] * .038))
    minimum = max(45, int(gray.shape[1] * .07))
    raw = cv2.HoughLinesP(binary, 1, np.pi / 1440, threshold=threshold,
                          minLineLength=minimum, maxLineGap=28)
    return raw.reshape(-1, 4).astype(float) if raw is not None else np.empty((0, 4), float)


def _cluster_vertical(segments, height, width):
    mid_y = height * .47
    candidates = []
    for x0, y0, x1, y1 in segments:
        dx, dy = x1-x0, y1-y0
        if (abs(dy) < max(85, height*.075) or abs(dx) > abs(dy) * .24
                or min(y0,y1) < height*.29 or max(y0,y1) > height*.635
                or min(y0,y1) > height*.385 or max(y0,y1) < height*.53):
            continue
        at_mid = x0 + dx * (mid_y-y0)/dy
        if .04*width <= at_mid <= .93*width:
            candidates.append({'x': at_mid, 'points': (x0, y0, x1, y1), 'span': abs(dy)})
    groups = []
    for item in sorted(candidates, key=lambda v:v['x']):
        if not groups or item['x'] - np.median([g['x'] for g in groups[-1]]) > width*.033:
            groups.append([item])
        else:
            groups[-1].append(item)
    found = []
    for group in groups:
        ys = [y for g in group for y in (g['points'][1],g['points'][3])]
        if max(ys)-min(ys) < height*.18:
            continue
        found.append(dict(x=float(np.median([g['x'] for g in group])),
                          y_min=float(min(ys)), y_max=float(max(ys)), count=len(group),
                          span=float(max(ys)-min(ys))))
    return found


def _horizontals(segments, width, height, left, right):
    out=[]
    for x0,y0,x1,y1 in segments:
        dx,dy=x1-x0,y1-y0
        if abs(dx) < (right-left)*.42 or abs(dy) > abs(dx)*.21:
            continue
        if not (height*.29 < min(y0,y1) < height*.64 and height*.32 < max(y0,y1) < height*.68):
            continue
        # Rule must run through a substantial part of the observed grid, not heading text.
        if max(x0,x1) < left+(right-left)*.35 or min(x0,x1) > right-(right-left)*.35:
            continue
        slope=dy/dx
        at_mid=y0+slope*(.5*(left+right)-x0)
        out.append(dict(y=float(at_mid), slope=float(slope), length=abs(dx),
                        ends=[x0,y0,x1,y1]))
    # Repeated Hough segments for the same printed rule count only once.
    levels=[]
    for item in sorted(out,key=lambda v:v['y']):
        if not levels or item['y']-np.mean([a['y'] for a in levels[-1]]) > max(8.,height*.013):
            levels.append([item])
        else:levels[-1].append(item)
    return levels


def _choose_five(vertical, width, height):
    if len(vertical)<5:return None
    if len(vertical)>18:return None  # Bound the combinatorial search on clutter.
    # Avoid barcode/paper contours: five long parallel dividers with plausible
    # proportions and substantial vertical overlap.
    import itertools
    winner=None
    for candidate in itertools.combinations(vertical,5):
        v=sorted(candidate,key=lambda a:a['x']); xx=np.array([a['x'] for a in v])
        gaps=np.diff(xx)
        if (xx[-1]-xx[0] < width*.57 or xx[-1]-xx[0] > width*.92 or
             gaps.min() < width*.055 or gaps.max() > width*.49 or
             xx[0] > width*.135 or xx[-1]<width*.68):continue
        # Receipt columns: narrow row number, wide name, narrow quantity, price.
        frac=gaps/(xx[-1]-xx[0])
        if not (.065<frac[0]<.16 and .35<frac[1]<.65 and .08<frac[2]<.30 and .12<frac[3]<.35):continue
        if min(a['span'] for a in v)<height*.14:continue
        score=sum(min(a['span'],height*.29) for a in v) + sum(min(a['count'],4)*12 for a in v)
        if winner is None or score>winner[0]:winner=(score,v)
    return winner[1] if winner else None


def detect_damaged_grid(image):
    """Return (boundary, report). No OCR or expected totals are used."""
    empty=dict(polygon=None,status='needs_review',warnings=['damaged_grid_not_confirmed'],
               algorithm_version=VERSION,candidates=[])
    report=dict(version=VERSION,status='not_recovered',reason='insufficient_observed_grid',ocr_calls=0)
    if image is None or image.ndim not in (2,3) or min(image.shape[:2])<240:
        report['reason']='invalid_image';return empty,report
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    h,w=gray.shape
    # Avoid upscaling the whole photo: Hough/CLAHE are local, CPU-only.
    segments=_segments(gray)
    vertical=_cluster_vertical(segments,h,w)
    selected=_choose_five(vertical,w,h)
    report.update(line_segments=len(segments),vertical_groups=len(vertical),
                  selected_columns=[round(a['x'],1) for a in selected] if selected else [])
    if selected is None:return empty,report
    left,right=selected[0]['x'],selected[-1]['x']
    levels=_horizontals(segments,w,h,left,right)
    report['horizontal_levels']=len(levels)
    if len(levels)<3 or levels[-1][0]['y']-levels[0][0]['y']<h*.14:
        report['reason']='insufficient_horizontal_rules';return empty,report
    # Strong top rule: must span most of the complete table and lie close to
    # the top of the observed left/right printed vertical borders.
    top_rules=[i for group in levels for i in group
               if i['length'] >= (right-left)*.73]
    top_rules.sort(key=lambda i:i['y'])
    if not top_rules:
        report['reason']='full_top_rule_missing';return empty,report
    top=top_rules[0]
    # Keep independent full-width printed separators for segmentation. Torn
    # name-column rules can vanish from a simple horizontal morphology pass.
    rules = [max(group, key=lambda item:item['length']) for group in levels
             if max(item['length'] for item in group) >= (right-left)*.73
             and np.mean([item['y'] for item in group]) >= top['y']-8]
    report['horizontal_rules'] = [dict(ends=list(map(float,r['ends'])),
                                       slope=float(r['slope'])) for r in rules]
    y_at_left=top['ends'][1]+top['slope']*(left-top['ends'][0]);y_at_right=top['ends'][1]+top['slope']*(right-top['ends'][0])
    if abs(y_at_left-selected[0]['y_min'])>h*.055 or abs(y_at_right-selected[-1]['y_min'])>h*.055:
        report['reason']='top_rule_not_aligned_to_grid';return empty,report
    # Printed bottom is stepped (total box exists only in last columns).
    # Both lower endpoints must be observed, not projected from one another.
    bottom_left=selected[0]['y_max'];bottom_right=selected[-1]['y_max']
    if not (bottom_left-y_at_left>h*.14 and bottom_right-y_at_right>h*.14
            and abs((bottom_left-y_at_left)-(bottom_right-y_at_right))<h*.12):
        report['reason']='bottom_dividers_inconsistent';return empty,report
    # Keep four real side intersections as a conservative working hypothesis.
    polygon=np.asarray([[left,y_at_left],[right,y_at_right],[right,bottom_right],[left,bottom_left]],np.float32)
    if not np.isfinite(polygon).all() or np.any(polygon[:,0]<0) or np.any(polygon[:,0]>=w) or np.any(polygon[:,1]<0) or np.any(polygon[:,1]>=h):
        report['reason']='out_of_bounds';return empty,report
    selected_info=dict(source='damaged_observed_grid',support=[len(levels),5],
                       bounds=[float(left),float(y_at_left),float(right-left),float(bottom_right-y_at_left)],
                       vertical_dividers=[round(a['x'],1) for a in selected],
                       horizontal_rule_count=len(levels))
    boundary=dict(polygon=polygon,status='needs_review',warnings=['damaged_broken_grid_fallback'],
                  algorithm_version=VERSION,selected=selected_info,candidates=[])
    report.update(status='recovered',reason='observed_dividers_and_horizontal_rules',
                  polygon=polygon.tolist(),horizontal_levels=len(levels),
                  selected_columns=selected_info['vertical_dividers'])
    return boundary,report


def recover_damaged_boundary(image, previous, verifier=None, previous_verification=None):
    """Never change a verified pre-existing table; only recover failed geometry."""
    verification = previous_verification or {}
    fragment = (previous.get('polygon') is not None and
                verification.get('status') == 'uncertain' and
                verification.get('geometry', {}).get('four_columns') is False and
                verification.get('header', {}).get('state') != 'confirmed')
    if previous.get('polygon') is not None and not fragment:
        return previous,dict(version=VERSION,status='not_needed',ocr_calls=0)
    candidate,report=detect_damaged_grid(image)
    if candidate['polygon'] is None:return previous,report
    if fragment:
        # A partial right-hand grid is not a missing polygon. Require a full
        # observed grid surrounding it, plus independent physical tear evidence.
        from App.ocr.damaged_receipt_text import detect_damage_stripe
        old = np.asarray(previous['polygon'], dtype=float)
        new = np.asarray(candidate['polygon'], dtype=float)
        lo, hi = old.min(axis=0), old.max(axis=0)
        full_lo, full_hi = new.min(axis=0), new.max(axis=0)
        width = full_hi[0] - full_lo[0]
        height = full_hi[1] - full_lo[1]
        contained_right_fragment = (
            hi[0]-lo[0] < width*.60 and lo[0] > full_lo[0]+width*.45 and
            abs(hi[0]-full_hi[0]) < width*.08 and
            lo[1] >= full_lo[1]-height*.10 and hi[1] <= full_hi[1]+height*.10 and
            hi[1]-lo[1] >= height*.60)
        stripe = detect_damage_stripe(image, candidate['polygon'], report['selected_columns'])
        report['fragment_check'] = dict(contained_right_fragment=bool(contained_right_fragment),
                                       damage_stripe=stripe)
        if not contained_right_fragment or not stripe.get('forced_ocr'):
            report.update(status='not_recovered', reason='existing_boundary_not_proven_fragment')
            return previous, report
    # Optional production verifier is used WITHOUT a second full OCR pass.
    if verifier is not None:
        verification=verifier(image,candidate,None)
        report['verification']=verification
        if verification.get('status')=='rejected':
            report.update(status='not_recovered',reason='candidate_rejected_by_project_verifier')
            return previous,report
        if fragment and verification.get('status') != 'confirmed':
            report.update(status='not_recovered', reason='full_grid_not_confirmed')
            return previous, report
    elif fragment:
        report.update(status='not_recovered', reason='fragment_requires_project_verifier')
        return previous, report
    report['replaced_fragment'] = fragment
    return candidate,report


def observed_row_layout(layout, map_x, map_y, report):
    """Map measured full-width rules into the working crop, without OCR.

    The stepped, short total box is deliberately outside the product grid.
    Refuse an incomplete/ambiguous set rather than invent a missing divider.
    """
    rules = report.get('horizontal_rules', [])
    audit = dict(status='not_applied', reason='insufficient_printed_separators')
    if not 4 <= len(rules) <= 6:
        return layout, audit
    height, width = map_y.shape
    columns = np.linspace(layout['name_left']+8,layout['right']-8,5).astype(int)
    columns = np.clip(columns,0,width-1)
    levels = []
    for rule in rules:
        x0,y0,_,_ = rule['ends']
        distance = map_y[:,columns]-(y0+rule['slope']*(map_x[:,columns]-x0))
        indices = np.argmin(np.abs(distance),axis=0)
        if np.max(np.abs(distance[indices,np.arange(len(columns))])) > 2:
            return layout, dict(audit, reason='separator_outside_crop')
        levels.append(float(np.median(indices)))
    gaps = np.diff(levels)
    if (min(gaps) < 15 or levels[0] < 0 or levels[-1] >= height or
            not .04*width < gaps[0] < .30*width or
            levels[-1]-levels[0] > .70*width):
        return layout, dict(audit, reason='implausible_printed_bands')
    revised = dict(layout,row_boundaries=levels,body_top=levels[0],body_bottom=levels[-1],
                   grid_confirmed=True)
    return revised, dict(status='applied',reason='observed_full_width_rules',levels=levels)
