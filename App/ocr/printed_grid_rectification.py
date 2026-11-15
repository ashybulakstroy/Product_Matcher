"""Experimental printed-grid 2D rectification. Never infer receipt values.

Input: EXIF-oriented source BGR image. Output: a rectified image and backward
pixel maps (destination -> source), or a rejected report with no changes.
No OCR, catalog, totals, network or dependencies beyond numpy/OpenCV.
"""
from __future__ import annotations

from itertools import combinations
from math import isfinite

import cv2
import numpy as np

VERSION = 'printed-grid-rectification-v3.0'


def _threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (3, 3), 0), 255,
                                 cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 12)


def vertical_candidates(image):
    """Locate long, printed vertical strokes using morphology + Hough.

    All candidates are observations, not the five dividers themselves. Paper
    edges may also appear; they are rejected by the four-column geometry gate.
    """
    h, w = image.shape[:2]
    if h < 250 or w < 260:
        return [], {'reason': 'too_small'}
    x0, x1 = round(w * .08), round(w * .85)
    y0, y1 = round(h * .12), round(h * .76)
    mask = _threshold(image)[y0:y1, x0:x1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(11, round(.012 * h))))
    long_vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    segments = cv2.HoughLinesP(long_vertical, 1, np.pi / 720,
                               threshold=max(23, round(.03 * w)),
                               minLineLength=max(65, round(.085 * h)), maxLineGap=35)
    if segments is None:
        return [], {'reason': 'no_vertical_lines'}
    reference_y = (y0 + y1) / 2
    parts = []
    for ax, ay, bx, by in segments.reshape(-1, 4):
        dy, dx = float(by - ay), float(bx - ax)
        if abs(dy) < .085 * h or abs(dx) > .22 * abs(dy):
            continue
        ax, bx, ay, by = ax + x0, bx + x0, ay + y0, by + y0
        ref_x = float(ax + (bx - ax) * (reference_y - ay) / (by - ay))
        parts.append((ref_x, float(ax), float(ay), float(bx), float(by)))
    parts.sort()
    groups = []
    for part in parts:
        if not groups or part[0] - np.mean([item[0] for item in groups[-1]]) > max(10, .014 * w):
            groups.append([part])
        else:
            groups[-1].append(part)
    result = []
    for group in groups:
        support = sum(abs(item[4] - item[2]) for item in group)
        high = min(min(item[2], item[4]) for item in group)
        low = max(max(item[2], item[4]) for item in group)
        if support < h * .3 or low - high < h * .25:
            continue
        yy = np.array([value for p in group for value in (p[2], p[4])], dtype=np.float64)
        xx = np.array([value for p in group for value in (p[1], p[3])], dtype=np.float64)
        weights = np.array([abs(p[4] - p[2]) ** .4 for p in group for _ in range(2)])
        design = np.column_stack((yy - reference_y, np.ones_like(yy)))
        slope, position = np.linalg.lstsq(design * weights[:, None], xx * weights, rcond=None)[0]
        if not np.isfinite([slope, position]).all() or abs(slope) > .22:
            continue
        result.append(dict(x=float(position), slope=float(slope), top=float(high),
                           bottom=float(low), support=float(support)))
    return result, dict(reason='ok', reference_y=reference_y, number_of_candidates=len(result))


def select_five_dividers(candidates, shape):
    """Pick physical columns only if 5 supports and Torgsoft ratios are present."""
    h, w = shape[:2]
    if len(candidates) < 5 or len(candidates) > 20:
        return None, {'status': 'rejected', 'reason': 'five_printed_columns_not_supported'}
    ranked = sorted(candidates, key=lambda c: c['x'])
    best = None
    for option in combinations(ranked, 5):
        xs = np.array([p['x'] for p in option])
        width = xs[-1] - xs[0]
        if not w * .25 < width < w * .85:
            continue
        shares = np.diff(xs) / width
        # Column order: item number | name | quantity | price.
        if not (.048 < shares[0] < .20 and .38 < shares[1] < .69 and
                .09 < shares[2] < .27 and .12 < shares[3] < .32):
            continue
        shape_penalty = np.sum(np.abs(shares - np.array([.10, .52, .16, .22])))
        support = sum(min(1., c['support'] / (h * 1.2)) for c in option)
        score = -shape_penalty * 12 + support * .2
        if best is None or score > best[0]:
            best = score, option, shares
    if best is None:
        return None, {'status': 'rejected', 'reason': 'column_geometry_not_confirmed'}
    return list(best[1]), {'status': 'found', 'score': round(float(best[0]), 4),
                           'column_width_shares': [round(float(v), 4) for v in best[2]]}


def _vertical_warp(image, five, reference_y):
    h, w = image.shape[:2]
    first_y = round(float(np.median([line['top'] for line in five])))
    last_y = round(float(np.median([line['bottom'] for line in five])))
    if first_y < h * .04 or last_y > h * .90 or last_y - first_y < h * .32:
        raise ValueError('vertical_line_spans_not_trustworthy')
    first_y = max(0, first_y)
    last_y = min(h - 1, last_y)

    def source_columns(src_y):
        return np.array([c['x'] + c['slope'] * (src_y - reference_y) for c in five], dtype=float)

    center = (first_y + last_y) / 2
    widths = np.diff(source_columns(center))
    if np.min(widths) <= 14:
        raise ValueError('columns_intersect')
    margin = 16
    columns = np.r_[0., np.cumsum(widths)]
    columns = np.round(columns).astype(int) + margin
    output_width = int(columns[-1] + margin)
    footer = max(36, round(output_width * .10))
    output_height = int(last_y - first_y + 2 * margin + footer)
    if output_width > w * 2 or output_height > h * 2:
        raise ValueError('overlarge_output')

    map_x = np.empty((output_height, output_width), np.float32)
    src_y = np.arange(output_height, dtype=np.float32) + first_y - margin
    dest_x = np.arange(output_width, dtype=np.float32)
    for row, yy in enumerate(src_y):
        source_x = source_columns(float(yy))
        if np.min(np.diff(source_x)) <= 10:
            raise ValueError('crossed_printed_verticals')
        interpolated = np.interp(dest_x, columns, source_x)
        interpolated[:columns[0]] = source_x[0] + dest_x[:columns[0]] - columns[0]
        interpolated[columns[-1]:] = source_x[-1] + dest_x[columns[-1]:] - columns[-1]
        map_x[row] = interpolated
    map_y = np.broadcast_to(src_y[:, None], map_x.shape).copy()
    return map_x, map_y, [int(c) for c in columns], margin, int(last_y - first_y)


def horizontal_rules(image):
    """Measure printed horizontal separators after vertical correction."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    binary = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (3, 3), 0), 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (max(16, int(.04*w)), 1)))
    segments = cv2.HoughLinesP(horizontal, 1, np.pi / 720, threshold=max(20, round(.065*w)),
                               minLineLength=max(60, round(.15*w)), maxLineGap=max(12, int(.05*w)))
    if segments is None:
        return []
    lines = []
    for x1, y1, x2, y2 in segments.reshape(-1, 4):
        dx, dy = float(x2 - x1), float(y2 - y1)
        if abs(dx) < .15*w or abs(dy) > .16*abs(dx):
            continue
        slope = dy / dx
        lines.append((float(y1 + slope * (w/2 - x1)), slope, abs(dx)))
    lines.sort()
    groups = []
    for line in lines:
        if not groups or line[0] - np.mean([r[0] for r in groups[-1]]) > max(4, h * .008):
            groups.append([line])
        else:
            groups[-1].append(line)
    result = []
    for group in groups:
        support = sum(r[2] for r in group)
        if support < .2 * w:
            continue
        level, slope = np.average(np.array([r[:2] for r in group]), axis=0,
                                  weights=np.array([r[2] for r in group]))
        result.append((float(level), float(slope), float(support)))
    return result


def layout_rules(image):
    """Measure printed horizontal separators after vertical correction."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    binary = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (3, 3), 0), 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 3)
    # Faint slanted rules otherwise vanish under a one-pixel opening.
    binary = cv2.dilate(binary, np.ones((3, 1), np.uint8))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (max(16, int(.13*w)), 1)))
    segments = cv2.HoughLinesP(horizontal, 1, np.pi / 720, threshold=max(20, round(.065*w)),
                               minLineLength=max(60, round(.30*w)), maxLineGap=max(12, int(.12*w)))
    if segments is None:
        return []
    lines = []
    for x1, y1, x2, y2 in segments.reshape(-1, 4):
        dx, dy = float(x2 - x1), float(y2 - y1)
        if abs(dx) < .15*w or abs(dy) > .16*abs(dx):
            continue
        slope = dy / dx
        lines.append((float(y1 + slope * (w/2 - x1)), slope, abs(dx)))
    lines.sort()
    groups = []
    for line in lines:
        if not groups or line[0] - np.mean([r[0] for r in groups[-1]]) > max(4, h * .008):
            groups.append([line])
        else:
            groups[-1].append(line)
    result = []
    for group in groups:
        support = sum(r[2] for r in group)
        if support < .2 * w:
            continue
        level, slope = np.average(np.array([r[:2] for r in group]), axis=0,
                                  weights=np.array([r[2] for r in group]))
        result.append((float(level), float(slope), float(support)))
    return result


def refine_printed_row_boundaries(image, boundaries, layout, lines):
    """Keep nearby observed separators outside original OCR name rectangles.

    Residual slopes and padded OCR boxes can overlap by a few pixels. Only
    refine an existing boundary near a real long rule, without moving any
    text centre between rows, adding rows or editing source OCR evidence.
    """
    bounds = list(map(float, boundaries))
    rules = [r[0] for r in layout_rules(image) if r[2] >= image.shape[1] * .3]
    if len(bounds) < 3 or len(rules) < 5:
        return bounds
    boxes=[]
    for line in lines:
        pts=np.asarray(line.get('eval_box',line.get('box')),dtype=float)
        if pts.shape != (4,2) or not np.isfinite(pts).all():
            continue
        if layout['name_left']+3 < pts[:,0].mean() < layout['name_right']-3:
            boxes.append((float(pts[:,1].min()),float(pts[:,1].max()),float(pts[:,1].mean())))
    updated=bounds.copy()
    for i in range(1,len(bounds)-1):
        limit=min(14.,(bounds[i]-bounds[i-1])*.35,(bounds[i+1]-bounds[i])*.35)
        near=[r for r in rules if abs(r-bounds[i]) <= limit]
        if not near:
            continue
        target=min(near,key=lambda r:abs(r-bounds[i]))
        lower,upper=bounds[i]-limit,bounds[i]+limit
        for top,bottom,center in boxes:
            if center < bounds[i]: lower=max(lower,center+.01)
            else: upper=min(upper,center-.01)
        candidates=[float(np.clip(target,lower,upper)),bounds[i]]
        candidates.extend(v for top,bottom,_ in boxes for v in (top+2,bottom-2) if lower<=v<=upper)
        safe=[v for v in candidates if lower<=v<=upper and
              all(not top+2 < v < bottom-2 for top,bottom,_ in boxes)]
        if safe:
            updated[i]=min(safe,key=lambda v:abs(v-target))
    if any(b-a < 8 for a,b in zip(updated,updated[1:])):
        return bounds
    return updated


def _horizontal_warp(original, original_x, original_y):
    """Straighten slopes using separator anchors while sampling source once."""
    current = cv2.remap(original, original_x, original_y, cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    rules = horizontal_rules(current)
    report = {'observed_rules': len(rules), 'horizontal_adjusted': False}
    if len(rules) < 5:
        return current, original_x, original_y, report
    h, w = current.shape[:2]
    levels = np.array([v[0] for v in rules], dtype=float)
    slopes = np.array([v[1] for v in rules], dtype=float)
    if np.min(np.diff(levels)) < 12:
        return current, original_x, original_y, report
    output_x = np.broadcast_to(np.arange(w, dtype=np.float32)[None, :], (h, w)).copy()
    output_y = np.empty((h, w), dtype=np.float32)
    dst_rows = np.arange(h, dtype=float)
    for x in range(w):
        source_rows = levels + slopes * (x - w/2)
        mapped = np.interp(dst_rows, levels, source_rows)
        before, after = dst_rows < levels[0], dst_rows > levels[-1]
        mapped[before] = dst_rows[before] + source_rows[0] - levels[0]
        mapped[after] = dst_rows[after] + source_rows[-1] - levels[-1]
        output_y[:, x] = mapped
    jacobian = np.gradient(output_y, axis=0)
    if not np.isfinite(jacobian).all() or not (.7 <= jacobian.min() <= jacobian.max() <= 1.3):
        report['reason'] = 'horizontal_jacobian_unsafe'
        return current, original_x, original_y, report
    composed_x = cv2.remap(original_x, output_x, output_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
    composed_y = cv2.remap(original_y, output_x, output_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
    output = cv2.remap(original, composed_x, composed_y, cv2.INTER_CUBIC,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    report.update(horizontal_adjusted=True, min_jacobian=float(jacobian.min()),
                  max_jacobian=float(jacobian.max()))
    return output, composed_x, composed_y, report


def rectify_by_printed_grid(image):
    """Return (output, backward_x, backward_y, metadata) or (None, report).

    Reject rather than invent missing borders. This candidate is not an OCR
    success: the consumer must separately validate rows, names, numbers and VAT.
    """
    if image is None or image.ndim != 3:
        return None, {'version': VERSION, 'status': 'rejected', 'reason': 'invalid_image'}
    try:
        candidates, geometry = vertical_candidates(image)
        selected, verification = select_five_dividers(candidates, image.shape)
        report = {'version': VERSION, 'status': 'rejected', 'vertical_detector': geometry,
                  'five_columns': verification}
        if selected is None:
            report['reason'] = verification['reason']
            return None, report
        mx, my, columns, margin, body_height = _vertical_warp(image, selected, geometry['reference_y'])
        output, mx, my, horizontal = _horizontal_warp(image, mx, my)
        if output.shape[:2] != mx.shape or not np.isfinite(mx).all() or not np.isfinite(my).all():
            raise ValueError('coordinate_map_invalid')
        meta = {'method': 'printed_grid_rectified', 'algorithm_version': VERSION,
                'margin': margin, 'body_size': [columns[-1] - columns[0], body_height],
                'source_polygon': None, 'warnings': [],
                'printed_grid_columns': columns, 'printed_grid_report': horizontal}
        report.update(status='candidate_ready', reason='five_printed_columns_and_nonfolded_warp',
                      columns=columns, horizontal=horizontal,
                      source_vertical_slopes=[round(v['slope'], 4) for v in selected])
        return (output, mx, my, meta), report
    except (ValueError, IndexError, cv2.error, np.linalg.LinAlgError) as error:
        return None, {'version': VERSION, 'status': 'rejected', 'reason': f'{type(error).__name__}: {error}'}


def project_cell_boundary(backward_x, backward_y, points, samples=40):
    """Curved original-photo polylines for a rectangular cell on rectified grid."""
    points = np.asarray(points, dtype=float)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError('invalid_cell_polygon')
    if backward_x.shape != backward_y.shape or backward_x.ndim != 2:
        raise ValueError('invalid_backward_maps')
    out = []
    height, width = backward_x.shape
    for start, end in zip(points, np.roll(points, -1, axis=0)):
        destination = np.linspace(start, end, samples, dtype=np.float32)
        xx = np.clip(destination[:, 0], 0, width-1).reshape(1, -1)
        yy = np.clip(destination[:, 1], 0, height-1).reshape(1, -1)
        px = cv2.remap(backward_x, xx, yy, cv2.INTER_LINEAR).ravel()
        py = cv2.remap(backward_y, xx, yy, cv2.INTER_LINEAR).ravel()
        out.append(np.stack([px, py], axis=-1).round(2).tolist())
    return out
