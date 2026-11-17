"""Experimental, conservative reconstruction and union of printed grid lines.

This module is intentionally outside the V3 detector and is not imported by the
working OCR.  It only groups the *already observed* candidate segments from the
parallel experiment, keeps their observed domains, and optionally checks each
added line against pixels before adding it to the current V3 lines.

The post-processor is shared by every observed line detector.  Geometry methods
(perspective / grid_mesh / tps) are model projections and are not mixed into the
observed-line union.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable
import math

import cv2
import numpy as np


@dataclass
class FullLine:
    axis: str
    points: np.ndarray
    domain: tuple[float, float]
    observed_length: float
    observed_coverage: float
    n_segments: int
    methods: tuple[str, ...] = ()
    evidence: float | None = None
    contrast: float | None = None
    sources: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "axis": self.axis,
            "points": self.points.round(3).tolist(),
            "domain": [round(float(self.domain[0]), 3), round(float(self.domain[1]), 3)],
            "observed_length": round(float(self.observed_length), 2),
            "observed_coverage": round(float(self.observed_coverage), 4),
            "n_segments": int(self.n_segments),
            "methods": list(self.methods),
            "evidence": None if self.evidence is None else round(float(self.evidence), 4),
            "contrast": None if self.contrast is None else round(float(self.contrast), 4),
            "sources": list(self.sources),
        }


def resample(points: Iterable, step: float = 3.0) -> np.ndarray:
    q = np.asarray(points, dtype=float)
    if q.ndim != 2 or q.shape[1] != 2 or len(q) < 2 or not np.isfinite(q).all():
        return np.empty((0, 2), float)
    # Remove repeated points so interpolation is well defined.
    keep = np.ones(len(q), bool)
    keep[1:] = np.linalg.norm(np.diff(q, axis=0), axis=1) > 1e-9
    q = q[keep]
    if len(q) < 2:
        return np.empty((0, 2), float)
    seg = np.linalg.norm(np.diff(q, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(seg)))
    total = float(cumulative[-1])
    if total < 1.0:
        return np.empty((0, 2), float)
    n = max(2, int(total / max(step, 1.0)) + 1)
    t = np.linspace(0.0, total, n)
    x = np.interp(t, cumulative, q[:, 0])
    y = np.interp(t, cumulative, q[:, 1])
    return np.column_stack((x, y))


def _axis_points(line: dict, shape) -> tuple[np.ndarray, str] | None:
    axis = line.get("axis")
    if axis not in ("h", "v"):
        return None
    pts = resample(line.get("points", []), 3.0)
    if len(pts) < 2:
        return None
    h, w = shape[:2]
    along = pts[:, 1] if axis == "v" else pts[:, 0]
    across = pts[:, 0] if axis == "v" else pts[:, 1]
    if np.ptp(along) < 8.0:
        return None
    # Reject diagonals mislabelled by a detector (they are not physical rules).
    if np.ptp(across) > 2.2 * np.ptp(along) + 12:
        return None
    return pts, axis


def _line_stats(pts: np.ndarray, axis: str, methods: tuple[str, ...], sources: tuple[str, ...]) -> FullLine:
    along = pts[:, 1] if axis == "v" else pts[:, 0]
    across = pts[:, 0] if axis == "v" else pts[:, 1]
    lo, hi = float(along.min()), float(along.max())
    span = max(hi - lo, 1.0)
    # Unique observed bins: do not reward a dense text blob as a long rule.
    bins = np.unique(np.rint(along / 3.0).astype(np.int64))
    observed_length = float(len(bins) * 3.0)
    coverage = min(1.0, observed_length / span)
    sort = np.argsort(along)
    a = along[sort]
    c = across[sort]
    degree = 1
    if span >= 0.20 * max(pts.shape[0] * 3.0, 200.0) and len(a) >= 24:
        degree = 2
    try:
        coeff = np.polyfit(a, c, degree)
    except (np.linalg.LinAlgError, ValueError):
        coeff = np.polyfit(a, c, 1)
        degree = 1
    grid = np.linspace(lo, hi, max(8, int(span / 3.0) + 1))
    across_fit = np.polyval(coeff, grid)
    if axis == "v":
        points = np.column_stack((across_fit, grid))
    else:
        points = np.column_stack((grid, across_fit))
    return FullLine(
        axis=axis,
        points=points,
        domain=(lo, hi),
        observed_length=observed_length,
        observed_coverage=coverage,
        n_segments=1,
        methods=methods,
        sources=sources,
    )


def _predict(line: FullLine, along_values: np.ndarray) -> np.ndarray:
    along = line.points[:, 1] if line.axis == "v" else line.points[:, 0]
    across = line.points[:, 0] if line.axis == "v" else line.points[:, 1]
    return np.interp(along_values, along, across, left=np.nan, right=np.nan)


def _compatible(a: FullLine, b: FullLine, extent: float, params: dict) -> bool:
    if a.axis != b.axis:
        return False
    near = params["near_px"](extent)
    gap_allow = params["gap_px"](extent)
    continuity = params["continuity_px"](extent)
    angle_tol = params["angle_rad"]
    a0, a1 = a.domain
    b0, b1 = b.domain
    overlap = min(a1, b1) - max(a0, b0)
    gap = max(a0, b0) - min(a1, b1)
    # Angle near the adjoining part.  Endpoint chords are stable enough for
    # short segments; long segments have already been resampled.
    def chord(line):
        p = line.points
        along = p[:, 1] if line.axis == "v" else p[:, 0]
        across = p[:, 0] if line.axis == "v" else p[:, 1]
        return math.atan2(float(across[-1] - across[0]), float(along[-1] - along[0]))
    if abs(chord(a) - chord(b)) > angle_tol:
        return False
    if overlap >= 0:
        probe = np.asarray([(max(a0, b0) + min(a1, b1)) * 0.5])
        da = _predict(a, probe)[0]
        db = _predict(b, probe)[0]
        return bool(np.isfinite(da) and np.isfinite(db) and abs(da - db) <= continuity)
    if gap > gap_allow:
        return False
    # Disjoint fragments: compare the two facing endpoints.
    if a1 < b0:
        da = _predict(a, np.asarray([a1]))[0]
        db = _predict(b, np.asarray([b0]))[0]
    else:
        da = _predict(b, np.asarray([b1]))[0]
        db = _predict(a, np.asarray([a0]))[0]
    return bool(np.isfinite(da) and np.isfinite(db) and abs(da - db) <= continuity)


class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def postprocess(lines, shape, method: str = "", params: dict | None = None):
    """Merge observed fragments into full, non-extrapolated lines.

    The parameters are fixed before evaluation and are deliberately tighter than
    typical printed row spacing.  Two long parallel rules are never merged.
    """
    h, w = shape[:2]
    params = params or {}
    params = {
        "near_px": params.get("near_px", lambda extent: max(5.0, 0.006 * extent)),
        "gap_px": params.get("gap_px", lambda extent: max(9.0, 0.022 * extent)),
        "continuity_px": params.get("continuity_px", lambda extent: max(4.0, 0.006 * extent)),
        "angle_rad": params.get("angle_rad", 0.14),
    }
    prepared = []
    for line in lines:
        got = _axis_points(line, shape)
        if got is None:
            continue
        pts, axis = got
        full = _line_stats(pts, axis, (method,), (line.get("source", "observed"),))
        prepared.append(full)
    # Union fragments per axis.
    by_axis = {"h": [], "v": []}
    for idx, line in enumerate(prepared):
        by_axis[line.axis].append(idx)
    uf = _UnionFind(len(prepared))
    for axis, indices in by_axis.items():
        extent = w if axis == "h" else h
        for ii in range(len(indices)):
            for jj in range(ii + 1, len(indices)):
                a = prepared[indices[ii]]
                b = prepared[indices[jj]]
                if _compatible(a, b, extent, params):
                    uf.union(indices[ii], indices[jj])
    groups = {}
    for idx in range(len(prepared)):
        groups.setdefault(uf.find(idx), []).append(prepared[idx])
    out = []
    for group in groups.values():
        all_points = np.vstack([g.points for g in group])
        axis = group[0].axis
        # Refit all observed points, but keep the observed min/max domain.
        along = all_points[:, 1] if axis == "v" else all_points[:, 0]
        across = all_points[:, 0] if axis == "v" else all_points[:, 1]
        span = float(np.ptp(along))
        degree = 2 if span >= 0.20 * (h if axis == "v" else w) and len(along) >= 24 else 1
        try:
            coeff = np.polyfit(along, across, degree)
        except (np.linalg.LinAlgError, ValueError):
            coeff = np.polyfit(along, across, 1)
        lo, hi = float(along.min()), float(along.max())
        grid = np.linspace(lo, hi, max(8, int((hi - lo) / 3.0) + 1))
        fit = np.polyval(coeff, grid)
        points = np.column_stack((fit, grid)) if axis == "v" else np.column_stack((grid, fit))
        unique_bins = np.unique(np.rint(along / 3.0).astype(np.int64))
        observed_length = float(len(unique_bins) * 3.0)
        coverage = min(1.0, observed_length / max(hi - lo, 1.0))
        methods = tuple(sorted({m for g in group for m in g.methods if m}))
        sources = tuple(sorted({s for g in group for s in g.sources if s}))
        out.append(FullLine(axis, points, (lo, hi), observed_length, coverage,
                            sum(g.n_segments for g in group), methods, sources=sources))
    # Drop fragments that are too short to be a printed table rule.
    filtered = []
    for line in out:
        extent = w if line.axis == "h" else h
        if line.domain[1] - line.domain[0] < max(30.0, 0.06 * extent):
            continue
        if line.observed_length < max(24.0, 0.04 * extent):
            continue
        if line.observed_coverage < 0.20 and line.n_segments < 3:
            continue
        filtered.append(line)
    return sorted(filtered, key=lambda q: (q.axis, q.domain[0]))


def ink_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 8)
    return binary


def line_evidence(image, points, ink=None, half_width: int = 2, offset: int = 7):
    """Fraction of line samples with nearby ink, minus the same at offsets.

    This is a physical-pixel sanity check, not a proof that the line is a table
    rule.  It rejects lines that lie only on blank paper or text-free texture.
    """
    if ink is None:
        ink = ink_mask(image)
    q = resample(points, 3.0)
    if len(q) < 3:
        return 0.0, 0.0
    h, w = ink.shape[:2]
    p = np.rint(q).astype(int)
    p[:, 0] = np.clip(p[:, 0], 0, w - 1)
    p[:, 1] = np.clip(p[:, 1], 0, h - 1)
    # Axis from the actual point spread.
    dx, dy = np.ptp(q[:, 0]), np.ptp(q[:, 1])
    horizontal = dx >= dy
    hits = []
    off_hits = []
    for x, y in p:
        if horizontal:
            band = ink[max(0, y - half_width):min(h, y + half_width + 1), x]
            hits.append(bool(band.size and band.max() > 0))
            up = ink[max(0, y - offset - half_width):max(0, y - offset + half_width + 1), x]
            dn = ink[min(h - 1, y + offset - half_width):min(h, y + offset + half_width + 1) + 1, x]
            off_hits.append(bool((up.size and up.max() > 0) or (dn.size and dn.max() > 0)))
        else:
            band = ink[y, max(0, x - half_width):min(w, x + half_width + 1)]
            hits.append(bool(band.size and band.max() > 0))
            lf = ink[y, max(0, x - offset - half_width):max(0, x - offset + half_width + 1)]
            rt = ink[y, min(w - 1, x + offset - half_width):min(w, x + offset + half_width + 1) + 1]
            off_hits.append(bool((lf.size and lf.max() > 0) or (rt.size and rt.max() > 0)))
    if not hits:
        return 0.0, 0.0
    evidence = float(np.mean(hits))
    contrast = evidence - float(np.mean(off_hits))
    return evidence, contrast


def _overlap_fraction(a: FullLine, b: FullLine) -> float:
    lo = max(a.domain[0], b.domain[0])
    hi = min(a.domain[1], b.domain[1])
    if hi <= lo:
        return 0.0
    return (hi - lo) / max(min(a.domain[1] - a.domain[0], b.domain[1] - b.domain[0]), 1.0)


def _duplicate(a: FullLine, b: FullLine, shape, dup_px: float) -> bool:
    if a.axis != b.axis:
        return False
    if _overlap_fraction(a, b) < 0.25:
        return False
    lo = max(a.domain[0], b.domain[0])
    hi = min(a.domain[1], b.domain[1])
    probes = np.linspace(lo, hi, max(3, int((hi - lo) / 12) + 1))
    pa = _predict(a, probes)
    pb = _predict(b, probes)
    valid = np.isfinite(pa) & np.isfinite(pb)
    if not valid.any():
        return False
    return float(np.median(np.abs(pa[valid] - pb[valid]))) <= dup_px


def _merge_duplicate(a: FullLine, b: FullLine) -> FullLine:
    # Keep the better observed support, merge domains and provenance.
    base = a if (a.observed_length, a.observed_coverage) >= (b.observed_length, b.observed_coverage) else b
    other = b if base is a else a
    lo = min(a.domain[0], b.domain[0])
    hi = max(a.domain[1], b.domain[1])
    grid = np.linspace(lo, hi, max(8, int((hi - lo) / 3.0) + 1))
    # Evaluate the base fit inside its own domain and extend only along the
    # straight tangent over the other line's observed domain.
    along = base.points[:, 1] if base.axis == "v" else base.points[:, 0]
    across = base.points[:, 0] if base.axis == "v" else base.points[:, 1]
    deg = 2 if len(along) >= 24 and np.ptp(along) >= 80 else 1
    try:
        coeff = np.polyfit(along, across, deg)
    except (np.linalg.LinAlgError, ValueError):
        coeff = np.polyfit(along, across, 1)
    fit = np.polyval(coeff, grid)
    points = np.column_stack((fit, grid)) if base.axis == "v" else np.column_stack((grid, fit))
    return FullLine(
        axis=base.axis, points=points, domain=(lo, hi),
        observed_length=max(base.observed_length, other.observed_length),
        observed_coverage=min(1.0, max(base.observed_coverage, other.observed_coverage)),
        n_segments=base.n_segments + other.n_segments,
        methods=tuple(sorted(set(base.methods) | set(other.methods))),
        evidence=max(x for x in (base.evidence, other.evidence, 0.0) if x is not None),
        contrast=max(x for x in (base.contrast, other.contrast, 0.0) if x is not None),
        sources=tuple(sorted(set(base.sources) | set(other.sources))),
    )


def as_full_line(line, shape, method: str = "") -> FullLine | None:
    """Normalise a detector/V3 line dict into the common FullLine form."""
    if isinstance(line, FullLine):
        return line
    got = _axis_points(line, shape)
    if got is None:
        return None
    pts, axis = got
    full = _line_stats(pts, axis, (method or line.get("method", ""),),
                       (line.get("source", "observed"),))
    return full


def combine(v3_lines, detector_lines, image, evidence_min=0.55, contrast_min=0.05,
            dup_px=6.0):
    """Union V3 lines with pixel-supported observed detector lines.

    No line is extended beyond the union of the observed domains.  A detector
    line must pass the physical ink check before it can add a separator.
    """
    h, w = image.shape[:2]
    ink = ink_mask(image)
    accepted = []
    for line in v3_lines:
        item = as_full_line(line, image.shape, "physical_grid_current")
        if item is None:
            continue
        item.sources = tuple(sorted(set(item.sources) | {"v3_current"}))
        accepted.append(item)
    rejected = []
    for line in detector_lines:
        if isinstance(line, dict):
            # Allow raw dicts for convenience.
            got = _axis_points(line, image.shape)
            if got is None:
                continue
            pts, axis = got
            line = _line_stats(pts, axis, (line.get("method", ""),),
                               (line.get("source", "observed"),))
        ev, contrast = line_evidence(image, line.points, ink=ink)
        line.evidence = ev
        line.contrast = contrast
        if ev < evidence_min or contrast < contrast_min:
            rejected.append(line)
            continue
        duplicate_index = next((i for i, x in enumerate(accepted)
                                if _duplicate(x, line, image.shape, dup_px)), None)
        if duplicate_index is not None:
            merged = _merge_duplicate(accepted[duplicate_index], line)
            merged.sources = tuple(sorted(set(merged.sources) | set(line.sources)))
            accepted[duplicate_index] = merged
        else:
            line.sources = tuple(sorted(set(line.sources) | {"detector"}))
            accepted.append(line)
    accepted.sort(key=lambda q: (q.axis, q.domain[0]))
    return accepted, rejected
