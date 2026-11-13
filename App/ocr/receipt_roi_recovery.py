# -*- coding: utf-8 -*-
"""Conservative paper-ROI fallback for Torgsoft printed-grid detection.

Production entry point: recover_after_boundary_failure(image, detector, previous).
Runs OpenCV only, with no OCR, models, settings or network. Does not change
previous successful detections. The returned polygon is always in original
EXIF-normalized image coordinates.
"""
from __future__ import annotations

from typing import Any, Callable

import cv2
import numpy as np

VERSION = "long-receipt-roi-1.0"


def recover_compact_table(image, detector, previous, recognize_header=None):
    """Additional fallback for short tables occupying a small part of the paper.

    Keep the established long-table recovery unchanged. A compact replacement
    requires early verification of header words or a complete header grid.
    """
    from App.ocr.table_verification import verify_table_candidate

    box, paper = find_paper_roi(image)
    report = dict(status='not_recovered', paper=paper, attempts=[])
    if box is None:
        return previous, report
    x0, y0, x1, y1 = box
    crop = image[y0:y1, x0:x1]
    for scale in (1.5, 2.0):
        if max(crop.shape[:2]) * scale > 2400:
            break
        sample = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        candidate = detector(sample)
        polygon = candidate.get('polygon')
        if polygon is None:
            report['attempts'].append(dict(scale=scale, reason='no_boundary'))
            continue
        points = np.asarray(polygon, dtype=np.float32).copy()
        lo, hi = points.min(axis=0), points.max(axis=0)
        ch, cw = sample.shape[:2]
        if not (np.isfinite(points).all() and hi[0]-lo[0] >= cw*.57 and
                hi[1]-lo[1] >= (hi[0]-lo[0])*.18 and
                lo[0] >= 3 and lo[1] >= ch*.04 and hi[0] <= cw-3 and hi[1] <= ch*.94):
            report['attempts'].append(dict(scale=scale, reason='implausible_compact_grid'))
            continue
        points[:, 0] = points[:, 0]/scale + x0
        points[:, 1] = points[:, 1]/scale + y0
        candidate = dict(candidate, polygon=points)
        verification = verify_table_candidate(image, candidate, recognize_header)
        report['attempts'].append(dict(scale=scale, verification=verification))
        if verification['status'] != 'confirmed':
            continue
        candidate['warnings'] = list(candidate.get('warnings', [])) + ['compact_paper_roi_verified']
        report.update(status='recovered', scale=scale, verification=verification,
                      table_polygon=points.tolist())
        return candidate, report
    return previous, report


def recover_table_boundary(image, detector, previous, full_grid_detector=None):
    """Recheck small background contours using independently observed tall grids.

    A bright paper component can merge with pale background. Narrow overlapping
    windows change the grid scale without guessing paper corners or using OCR.
    A replacement needs eight horizontal and three vertical observed rules,
    must span most of the photo height and must not touch the window edges.
    """
    height, width = image.shape[:2]
    polygon = previous.get('polygon')
    if polygon is not None:
        pts = np.asarray(polygon).reshape(-1, 2)
        if np.ptp(pts[:, 1]) >= height * .20 and np.ptp(pts[:, 0]) >= width * .16:
            return previous, dict(version=VERSION, status='not_needed', ocr_calls=0)
        report = dict(version=VERSION, status='not_recovered',
                      trigger='small_boundary_candidate', attempts=[], ocr_calls=0)
    else:
        boundary, report = recover_after_boundary_failure(image, detector, previous, full_grid_detector)
        if boundary.get('polygon') is not None:
            return boundary, report
    # Landscape orientation is handled by the existing orientation detector.
    if height < width * 1.3:
        return previous, report
    candidates = []
    for center in (.35, .50, .65):
        left, right = round(width * (center - .25)), round(width * (center + .25))
        sample = image[:, left:right]
        candidate = detector(sample)
        valid, reason = _check_candidate(candidate, sample.shape[:2], min_support=8)
        if valid:
            pts = np.asarray(candidate['polygon'], dtype=np.float32)
            lo, hi = pts.min(axis=0), pts.max(axis=0)
            valid = bool(hi[1] - lo[1] >= height * .55 and
                         (hi[1] - lo[1]) / max(1, hi[0] - lo[0]) >= 1.5 and
                         lo[0] >= 5 and hi[0] <= sample.shape[1] - 6)
            if not valid:
                reason = 'short_or_clipped_grid'
        report['attempts'].append(dict(source='vertical_window', box=[left, 0, right, height],
                                       valid=valid, reason=reason))
        if valid:
            pts = pts.copy()
            if full_grid_detector is not None:
                # Broken upper separators can attach words to the contour.
                # Use independently fitted printed-rule endpoints for the sides,
                # retaining the observed full contour's top and bottom extent.
                grid, evidence = full_grid_detector(sample)
                if grid is not None and int(evidence.get('rule_count', 0)) >= 8:
                    tl, tr, br, bl = np.asarray(grid, dtype=np.float32)
                    top, bottom = float(pts[:, 1].min()), float(pts[:, 1].max())
                    grid_top, grid_bottom = max(tl[1], tr[1]), min(bl[1], br[1])
                    if (grid_bottom - grid_top >= height * .55 and
                            0 <= grid_top - top <= height * .25 and
                            0 <= bottom - grid_bottom <= height * .10):
                        def edge_x(a, b, y):
                            return float(a[0] + (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]))
                        envelope = np.array([[edge_x(tl, bl, top), top],
                                             [edge_x(tr, br, top), top],
                                             [edge_x(tr, br, bottom), bottom],
                                             [edge_x(tl, bl, bottom), bottom]], np.float32)
                        if (envelope[:, 0].min() >= 5 and
                                envelope[:, 0].max() <= sample.shape[1] - 6):
                            candidate = dict(candidate, observed_contour=pts.tolist(),
                                             grid_envelope=evidence, status='needs_review')
                            pts = envelope
            pts[:, 0] += left
            recovered = dict(candidate, polygon=pts,
                             warnings=list(candidate.get('warnings', [])) + ['vertical_window_grid_recovery'])
            candidates.append(recovered)
    if not candidates:
        return previous, report
    boundary = max(candidates, key=lambda b: float(np.ptp(b['polygon'][:, 1])))
    report.update(status='recovered', source='vertical_window_grid',
                  table_polygon=boundary['polygon'].tolist())
    return boundary, report


def _json_number(value: Any) -> float | int:
    if isinstance(value, (float, np.floating)):
        return float(value)
    return int(value)


def find_paper_roi(image: np.ndarray) -> tuple[tuple[int, int, int, int] | None, dict]:
    """Find a tall, nearly solid bright sheet surrounded by darker background.

    The method is used ONLY after the primary printed-table detector failed.
    It must not fall back to invented page corners or infer a table polygon
    from the paper rectangle.
    """
    if image is None or image.size == 0 or image.ndim not in (2, 3):
        return None, {"status": "invalid_image"}
    h, w = image.shape[:2]
    if min(h, w) < 160:
        return None, {"status": "image_too_small"}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    smooth = cv2.GaussianBlur(gray, (9, 9), 0)
    threshold, bright = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Require a significant bright/dark separation; white sheets on white
    # backgrounds must NOT be mistaken for a crop that supports a printed table.
    low = float(np.percentile(smooth, 15))
    high = float(np.percentile(smooth, 85))
    if high - low < 35:
        return None, {"status": "insufficient_contrast", "contrast_span": round(high - low, 2)}
    kernel_size = max(5, min(19, int(round(min(w, h) * 0.018)) | 1))
    mask = cv2.morphologyEx(bright, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    scored: list[tuple[float, tuple[int, int, int, int], dict]] = []
    for i in range(1, n):
        x, y, cw, ch, area = map(int, stats[i])
        coverage = area / max(1, cw * ch)
        if not (.17 <= cw / w <= .88 and ch / h >= .36
                and 1.4 <= ch / max(cw, 1) <= 7.0
                and coverage >= .55 and area / (w * h) >= .10):
            continue
        center_offset = abs(x + cw * .5 - w * .5) / w
        if center_offset > .34:
            continue
        # Require dark surroundings beside the bright sheet to exclude
        # page-filling photos with arbitrary global threshold components.
        left_side = gray[max(0, y):min(h, y + ch), max(0, x - max(8, cw // 10)):x]
        right_side = gray[max(0, y):min(h, y + ch), x + cw:min(w, x + cw + max(8, cw // 10))]
        surround = [float(np.median(s)) for s in (left_side, right_side) if s.size > 100]
        if not surround or float(np.median(surround)) >= threshold + 16:
            continue
        metric = area * coverage * (1 - center_offset * .65)
        scored.append((metric, (x, y, cw, ch),
                       {"coverage": round(coverage, 3), "center_offset": round(center_offset, 3)}))
    if not scored:
        return None, {"status": "paper_not_detected", "otsu": round(float(threshold), 2)}
    _, (x, y, cw, ch), meta = max(scored, key=lambda item: item[0])
    px = max(5, round(cw * .025))
    py = max(5, round(cw * .025))
    box = (max(0, x - px), max(0, y - py), min(w, x + cw + px), min(h, y + ch + py))
    return box, {"status": "paper_found", "box": list(box),
                 "raw_component": [x, y, cw, ch], "otsu": round(float(threshold), 2), **meta}


def _check_candidate(boundary: dict, shape: tuple[int, int], min_support: int = 5) -> tuple[bool, str]:
    polygon = boundary.get("polygon")
    if polygon is None:
        return False, "table_not_found_in_roi"
    poly = np.asarray(polygon, dtype=np.float32).reshape((-1, 2))
    if len(poly) < 4 or not np.isfinite(poly).all():
        return False, "invalid_polygon"
    h, w = shape
    left, top = poly.min(axis=0)
    right, bottom = poly.max(axis=0)
    width, height = float(right-left), float(bottom-top)
    if (width < w * .57 or height < h * .32 or width > w * 1.01 or height > h * 1.01
            or left < -2 or top < -2 or right > w + 2 or bottom > h + 2):
        return False, "implausible_table_size"
    if top < h * .04 and bottom > h * .91:
        return False, "paper_outline_not_printed_table"
    selected = boundary.get("selected") or {}
    support = selected.get("support") or [0, 0]
    # Need independently detected printed horizontal/vertical grid lines;
    # simply finding the edges of white paper does not count.
    if len(support) != 2 or int(support[0]) < min_support or int(support[1]) < 3:
        return False, "insufficient_grid_evidence"
    return True, "grid_confirmed"


def recover_after_boundary_failure(
    image: np.ndarray,
    detector: Callable[[np.ndarray], dict],
    previous: dict,
    full_grid_detector: Callable[[np.ndarray], tuple[np.ndarray | None, dict]] | None = None,
) -> tuple[dict, dict]:
    """Return (boundary, JSON-serializable report); respect successful baseline.

    detector is the project's unmodified detect_table_boundary function.
    Baseline should be computed by caller and never re-run here.
    """
    if previous.get("polygon") is not None:
        return previous, {"version": VERSION, "status": "not_needed", "ocr_calls": 0}
    box, paper = find_paper_roi(image)
    report = {"version": VERSION, "status": "not_recovered", "paper": paper,
              "attempts": [], "ocr_calls": 0}
    if box is None:
        return previous, report
    x0, y0, x1, y1 = box
    crop = image[y0:y1, x0:x1]
    # The project's independent full-grid rules must win over a large paper
    # contour. On narrow receipts the normal detector can find the sheet itself
    # after crop, rather than the printed table. This check prevents it.
    if full_grid_detector is not None:
        try:
            grid_polygon, grid_info = full_grid_detector(crop)
        except (ValueError, cv2.error, np.linalg.LinAlgError) as error:
            grid_polygon, grid_info = None, {"error": str(error)}
        ok_grid = False
        if grid_polygon is not None:
            pts = np.asarray(grid_polygon, dtype=np.float32).reshape(-1, 2)
            cx0, cy0 = pts.min(axis=0)
            cx1, cy1 = pts.max(axis=0)
            hh, ww = crop.shape[:2]
            rule_count = int(grid_info.get("rule_count", 0))
            residual = float(grid_info.get("residual_px", 1000))
            ok_grid = (len(pts) >= 4 and np.isfinite(pts).all() and
                       rule_count >= 8 and (cx1 - cx0) >= ww * .57 and
                       hh * .35 <= (cy1 - cy0) <= hh * .90 and
                       cy0 >= hh * .04 and cy1 <= hh * .94 and
                       residual <= max(20, ww * .07))
        report["attempts"].append({"source": "full_grid_rules", "valid": ok_grid,
                                   "reason": (grid_info or {}).get("reason", "no_full_grid"),
                                   "rule_count": (grid_info or {}).get("rule_count")})
        if ok_grid:
            pts = np.asarray(grid_polygon, dtype=np.float32).copy()
            pts[:, 0] += x0
            pts[:, 1] += y0
            selected = {"source": "paper_roi_full_grid_rules", "bounds":
                        [float(pts[:, 0].min()), float(pts[:, 1].min()),
                         float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1]))],
                        **{k: v for k, v in grid_info.items() if isinstance(v, (str, int, float, bool))}}
            boundary = dict(polygon=pts, status="needs_review",
                            warnings=["paper_roi_fallback_used", "paper_roi_full_grid_hypothesis"],
                            algorithm_version=VERSION, selected=selected, candidates=[])
            report.update(status="recovered", source="full_grid_rules", paper_roi=list(box),
                          table_polygon=pts.tolist(), table_bounds=[float(pts[:, 0].min()),
                          float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())])
            return boundary, report
    for scale in (1.0, 1.5):
        if scale > 1.0 and max(crop.shape[:2]) > 1600:
            break
        sample = crop if scale == 1.0 else cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        candidate = detector(sample)
        passed, reason = _check_candidate(candidate, sample.shape[:2])
        report["attempts"].append({"scale": scale, "status": candidate.get("status"),
                                   "valid": passed, "reason": reason,
                                   "support": (candidate.get("selected") or {}).get("support")})
        if not passed:
            continue
        polygon = np.asarray(candidate["polygon"], dtype=np.float32).copy()
        polygon[:, 0] = polygon[:, 0] / scale + x0
        polygon[:, 1] = polygon[:, 1] / scale + y0
        if not np.isfinite(polygon).all():
            continue
        recovered = dict(candidate)
        recovered["polygon"] = polygon
        recovered["warnings"] = list(candidate.get("warnings", [])) + ["paper_roi_fallback_used"]
        recovered["paper_roi"] = list(box)
        recovered["algorithm_version"] = str(candidate.get("algorithm_version", "")) + "+paper-roi-1.0"
        # An existing 'needs_review' flag stays unchanged; fallback must not
        # upgrade ambiguous geometry into confirmed geometry.
        report.update(status="recovered", paper_roi=list(box), table_polygon=polygon.tolist(),
                      table_bounds=[_json_number(v) for v in
                                    [polygon[:, 0].min(), polygon[:, 1].min(),
                                     polygon[:, 0].max(), polygon[:, 1].max()]],
                      scale=scale)
        return recovered, report
    return previous, report
