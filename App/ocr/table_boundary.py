"""Printed table boundaries, kept separate from perspective quadrilaterals.

Scores rank candidates within one image; they are not calibrated probabilities.
No camera model, convex hull or invented page corners are used here.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from App.ocr.full_grid_boundary import detect_full_grid_boundary

ALGORITHM_VERSION = "table-boundary-2.0"


@dataclass(frozen=True)
class BoundaryOptions:
    max_detection_side: int = 1600
    adaptive_block: int = 31
    adaptive_c: float = 3
    line_length: int = 19
    angle_range: int = 15
    angle_step: int = 3
    closing_sizes: tuple[int, ...] = (3, 5, 7)
    min_area_ratio: float = .004
    min_box_fill: float = .65
    min_segment_fraction: float = .3
    min_grid_lines: int = 3
    max_vertical_lines: int = 6  # Four columns; reject dense barcode stripes.
    polygon_tolerance_px: float = 1.5


def _line_masks(binary: np.ndarray, options: BoundaryOptions) -> list[np.ndarray]:
    masks = []
    n = options.line_length
    radius = (n - 1) / 2
    for base in (0, 90):
        mask = np.zeros_like(binary)
        for angle in range(base - options.angle_range, base + options.angle_range + 1, options.angle_step):
            delta = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))]) * radius
            kernel = np.zeros((n, n), np.uint8)
            cv2.line(kernel, tuple(np.rint(radius - delta).astype(int)),
                     tuple(np.rint(radius + delta).astype(int)), 1, 1)
            center = n // 2
            # Bridge small breaks along a line before suppressing letters.
            short = kernel[center - 3:center + 4, center - 3:center + 4]
            joined = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, short)
            mask |= cv2.morphologyEx(joined, cv2.MORPH_OPEN, kernel)
        masks.append(mask)
    return masks


def _grid_support(segments: list[np.ndarray], bounds: tuple, options: BoundaryOptions) -> list[int]:
    x, y, w, h = bounds
    counts = []
    for axis, segs in enumerate(segments):
        dx, dy = np.abs(segs[:, 2] - segs[:, 0]), np.abs(segs[:, 3] - segs[:, 1])
        select = ((segs[:, 0] >= x) & (segs[:, 2] >= x) & (segs[:, 0] < x + w)
                  & (segs[:, 2] < x + w) & (segs[:, 1] >= y) & (segs[:, 3] >= y)
                  & (segs[:, 1] < y + h) & (segs[:, 3] < y + h))
        fraction = options.min_segment_fraction
        select &= ((dx >= fraction * w) & (dy <= dx * .35)) if axis == 0 else ((dy >= fraction * h) & (dx <= dy * .35))
        positions = []
        for segment in segs[select]:
            pos = float(segment[1] + segment[3]) / 2 if axis == 0 else float(segment[0] + segment[2]) / 2
            if all(abs(pos - other) > max(5, (h if axis == 0 else w) * .06) for other in positions):
                positions.append(pos)
        counts.append(len(positions))
    return counts


def _internal_vertical_span(segments: np.ndarray, bounds: tuple) -> float:
    """How much of the height is supported by two internal column separators.

    A page enclosing a small table also encloses grid pixels; those pixels must
    span the candidate itself, rather than merely appear somewhere inside it.
    """
    x, y, w, h = bounds
    dx = np.abs(segments[:, 2] - segments[:, 0])
    dy = np.abs(segments[:, 3] - segments[:, 1])
    mid_x = (segments[:, 0] + segments[:, 2]) / 2
    valid = ((dx <= dy * .35) & (mid_x > x + .08*w) & (mid_x < x + .92*w)
             & (segments[:, 1] >= y) & (segments[:, 3] >= y)
             & (segments[:, 1] < y+h) & (segments[:, 3] < y+h))
    selected = np.flatnonzero(valid)
    positions, lengths = [], []
    for index in selected[np.argsort(-dy[selected])]:
        if all(abs(mid_x[index] - other) > max(5, w * .06) for other in positions):
            positions.append(mid_x[index])
            lengths.append(float(dy[index]) / h)
            if len(lengths) == 2:
                break
    return sum(lengths) / 2


def _dense_vertical_bars(binary: np.ndarray, bounds: tuple) -> bool:
    """Reject barcode fragments whose thick bars mimic a small printed grid."""
    x, y, w, h = bounds
    if h < w * .8:
        return False
    region = binary[y:y+h, x:x+w]
    persistent = np.mean(region > 0, axis=0) >= .65
    transitions = np.diff(np.r_[False, persistent, False].astype(np.int8))
    bars = int(np.count_nonzero(transitions == 1))
    return bars >= 4 and float(np.mean(persistent)) >= .32


def detect_table_boundary(image: np.ndarray, options: BoundaryOptions | None = None) -> dict:
    """Return an observed polygon or explicit needs_review, never a fabricated table.

    Angled directional morphology supplies grid evidence. Candidates from several
    gap sizes and edge/threshold masks compete using internal line evidence and
    boundary continuity. Area and rectangularity alone cannot select a table.
    """
    if image is None or image.size == 0:
        raise ValueError("Empty image passed to detect_table_boundary.")
    options = options or BoundaryOptions()
    source_h, source_w = image.shape[:2]
    scale = min(1., options.max_detection_side / max(source_h, source_w))
    work = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY) if work.ndim == 3 else work
    height, width = gray.shape
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, options.adaptive_block, options.adaptive_c)
    masks = _line_masks(binary, options)
    network = cv2.morphologyEx(masks[0] | masks[1], cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    segments = []
    for mask in masks:
        lines = cv2.HoughLinesP(mask, 1, np.pi / 360, 25, minLineLength=25, maxLineGap=12)
        segments.append(np.empty((0, 4), np.int32) if lines is None else lines.reshape(-1, 4))

    candidates = []
    outline_candidates = []
    for source_name, source in (("grid", network), ("adaptive", binary), ("edges", cv2.Canny(gray, 20, 80))):
        for gap in options.closing_sizes:
            closed = cv2.morphologyEx(source, cv2.MORPH_CLOSE, np.ones((gap, gap), np.uint8))
            contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                x, y, w, h = bounds = cv2.boundingRect(contour)
                if area < height * width * options.min_area_ratio or w < 50 or h < 30:
                    continue
                if x < 2 or y < 2 or x + w > width - 2 or y + h > height - 2:
                    continue
                if area / (w * h) < options.min_box_fill:
                    continue
                if _dense_vertical_bars(binary, bounds):
                    continue
                support = _grid_support(segments, bounds, options)
                local = contour - np.array([[[x, y]]], np.int32)
                fill = np.zeros((h, w), np.uint8)
                cv2.drawContours(fill, [local], -1, 255, -1)
                inside = cv2.erode(fill, np.ones((11, 11), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
                ink = sum(np.count_nonzero(mask[y:y+h, x:x+w][inside]) for mask in masks)
                continuity = max(1., cv2.arcLength(contour, True) / (2 * (w + h)))
                score = ink / np.sqrt(area) * min(support[0], 16) * min(support[1], 5) / continuity ** 4
                unweighted_score = score
                column_span = _internal_vertical_span(segments[1], bounds)
                score *= min(1., column_span / .9) ** 4
                candidate = dict(contour=contour, score=float(score), area=area, bounds=bounds,
                                 support=support, source=source_name, gap=gap, continuity=continuity,
                                 column_span=column_span, unweighted_score=unweighted_score)
                # A clean isolated outline is useful for scans and synthetic fixtures.
                # It remains explicitly unverified without an internal printed grid.
                if source_name == "edges" and continuity < 1.3:
                    outline_candidates.append(candidate)
                if min(support) >= options.min_grid_lines and support[1] <= options.max_vertical_lines:
                    candidates.append(candidate)

    warnings = []
    if not candidates and np.mean(gray > 245) > .9 and outline_candidates:
        candidates = [max(outline_candidates, key=lambda c: c["area"])]
        warnings.append("outline_without_grid")
    if not candidates:
        return dict(polygon=None, status="needs_review", warnings=["table_boundary_not_found"],
                    algorithm_version=ALGORITHM_VERSION, candidates=[])

    candidates.sort(key=lambda c: c["score"], reverse=True)
    best = candidates[0]
    # Short intact cells can outrank a complete grid with broken separators.
    # Prefer the enclosing multi-row hypothesis, but make that ambiguity explicit.
    if best["support"][0] <= 4:
        bx, by, bw, bh = best["bounds"]
        enclosing = []
        for candidate in candidates:
            cx, cy, cw, ch = candidate["bounds"]
            if (candidate["area"] > best["area"] * 4 and candidate["support"][0] >= 6
                    and cx <= bx and cy <= by and cx+cw >= bx+bw and cy+ch >= by+bh
                    and candidate["unweighted_score"] > best["unweighted_score"]):
                enclosing.append(candidate)
        if enclosing:
            best = max(enclosing, key=lambda c: (c["support"][0], c["unweighted_score"]))
            warnings.append("partial_and_full_table_candidates")
    x, y, w, h = best["bounds"]
    fill = np.zeros((h, w), np.uint8)
    cv2.drawContours(fill, [best["contour"] - np.array([[[x, y]]], np.int32)], -1, 255, -1)
    # Remove narrow connected text spurs without filling concavities or taking a hull.
    cleaned = cv2.morphologyEx(fill, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundary = max(contours, key=cv2.contourArea) + np.array([[[x, y]]], np.int32)
    polygon = cv2.approxPolyDP(boundary, max(options.polygon_tolerance_px, min(6., .008 * min(w, h))), True).reshape(-1, 2).astype(np.float32)
    # Actual per-axis resize factors avoid rounding errors on non-square images.
    polygon *= np.array([source_w / width, source_h / height], np.float32)
    if best["continuity"] > 1.15:
        warnings.append("boundary_has_gaps_or_attached_text")
    for other in candidates[1:]:
        if other["score"] < best["score"] * .85:
            break
        ox, oy, ow, oh = other["bounds"]
        intersection = max(0, min(x+w, ox+ow) - max(x, ox)) * max(0, min(y+h, oy+oh) - max(y, oy))
        union = w*h + other["bounds"][2]*other["bounds"][3] - intersection
        if intersection / max(union, 1) < .8:
            warnings.append("ambiguous_boundary_candidates")
            break

    def metadata(c):
        return {key: c[key] for key in ("score", "area", "bounds", "support", "source", "gap", "continuity", "column_span")}

    # A hand or a dark annotation can close a single row into a high-scoring
    # contour.  It must not become the OCR table.  Only for an already
    # suspicious *and* implausibly small candidate, look for an independently
    # supported full printed grid in the original-resolution image.
    partial = w / width < .45 or h / height < .20
    if warnings and partial:
        fallback_polygon, fallback = detect_full_grid_boundary(image)
        if fallback_polygon is not None:
            fallback_selected = dict(source="full_grid_rules", bounds=[float(fallback_polygon[:, 0].min()),
                                     float(fallback_polygon[:, 1].min()), float(np.ptp(fallback_polygon[:, 0])),
                                     float(np.ptp(fallback_polygon[:, 1]))], **fallback)
            return dict(polygon=fallback_polygon, status="needs_review",
                        warnings=warnings + ["partial_contour_replaced_by_full_grid_hypothesis"],
                        algorithm_version=ALGORITHM_VERSION, selected=fallback_selected,
                        candidates=[metadata(c) for c in candidates[:10]])
    return dict(polygon=polygon, status="needs_review" if warnings else "detected",
                warnings=warnings, algorithm_version=ALGORITHM_VERSION,
                selected=metadata(best), candidates=[metadata(c) for c in candidates[:10]])
