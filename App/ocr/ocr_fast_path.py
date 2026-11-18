"""Conservative opt-in geometry Fast Path for clean Torgsoft printed-grid photos.

This module never infers items, numeric values, VAT or total. It selects only
between the already-existing contour rectification and the full geometry stack.
"""
from __future__ import annotations

from pathlib import Path
import os

import cv2
import numpy as np

from App.ocr.contour_unwarp import build_contour_map
from App.ocr.dewarp_necessity_detector import assess_dewarp_necessity

VERSION = "ocr-fast-path-v1"


def fast_mode(root: Path) -> str:
    from dotenv import dotenv_values
    root = Path(root)
    config = dotenv_values(root / ".env", encoding="utf-8-sig") if (root / ".env").exists() else {}
    mode = str(os.environ.get("OCR_FAST_PATH_MODE", config.get("OCR_FAST_PATH_MODE") or "off")).lower().strip()
    if mode not in {"off", "select"}:
        raise ValueError("OCR_FAST_PATH_MODE должен быть off или select")
    return mode


def eligibility(verification, boundary, *, orientation_mode, paper_roi_report,
                grid_boundary_report, damage_grid_report, image_shape) -> list[str]:
    """Cheap structural gate: no pixel operations, no OCR calls."""
    reasons = []
    if orientation_mode != "normal" or image_shape[0] < image_shape[1]:
        reasons.append("orientation_not_normal_portrait")
    if verification.get("status") != "confirmed":
        reasons.append("table_not_confirmed")
    # A valid-looking four-column header is not enough; demand text evidence.
    if verification.get("header", {}).get("state") != "confirmed":
        reasons.append("text_header_not_confirmed")
    if verification.get("geometry", {}).get("dense_bars"):
        reasons.append("barcode_like_header")
    if boundary.get("status") != "detected" or boundary.get("warnings"):
        reasons.append("nonstandard_table_boundary")
    if boundary.get("polygon") is None:
        reasons.append("no_polygon")
    if paper_roi_report.get("status") != "not_needed":
        reasons.append("paper_roi_recovery_used")
    if grid_boundary_report.get("status") == "recovered":
        reasons.append("printed_boundary_recovery_used")
    if damage_grid_report.get("status") == "recovered":
        reasons.append("damaged_grid_recovery_used")
    return reasons


def geometry_is_straight(report) -> bool:
    if report.get("decision") != "SKIP":
        return False
    h, v = report.get("horizontal") or {}, report.get("vertical") or {}
    # Five printed vertical dividers, at least three independent horizontal rules.
    # Be stricter than the standalone Dewarp detector (which tolerates uncertainty).
    if h.get("qualified_rules", 0) < 3 or v.get("qualified_rules", 0) < 5:
        return False
    if h.get("curved_rules", 0) or v.get("curved_rules", 0):
        return False
    if any(x.get("coverage", 0) < .78 or x.get("fit_error_px", 999) > 1.2
           for x in h.get("rules", []) + v.get("rules", [])):
        return False
    return True


def try_fast_geometry(image, polygon, *, mode, verification, boundary,
                      orientation_mode, paper_roi_report, grid_boundary_report,
                      damage_grid_report):
    """Return (rectified tuple | None, JSON report), full stack on any uncertainty.

    Main OCR and all row/numeric recovery are deliberately unchanged. The
    caller must additionally validate the final receipt and retry full geometry
    if numerical checks fail.
    """
    report = {"version": VERSION, "mode": mode, "selected": False, "status": "disabled"}
    if mode != "select":
        return None, report
    reasons = eligibility(
        verification, boundary, orientation_mode=orientation_mode,
        paper_roi_report=paper_roi_report, grid_boundary_report=grid_boundary_report,
        damage_grid_report=damage_grid_report, image_shape=image.shape,
    )
    if reasons:
        report.update(status="ineligible", reasons=reasons)
        return None, report
    try:
        mx, my, meta = build_contour_map(polygon, trust_top=True)
        if mx.shape != my.shape or mx.size > 6_000_000:
            report.update(status="rejected", reason="map_too_large_or_invalid")
            return None, report
        if meta.get("top_right_refinement", {}).get("applied"):
            report.update(status="rejected", reason="corrected_top_right_corner")
            return None, report
        # Non-folded and not grossly distorted. Boundary is still validated
        # by the existing independent header+grid check.
        if meta.get("min_jacobian", 0) < .20 or meta.get("max_jacobian", 999) > 4:
            report.update(status="rejected", reason="unsafe_geometric_jacobian")
            return None, report
        rectified = cv2.remap(image, mx, my, cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        necessity = assess_dewarp_necessity(rectified)
        report["dewarp_necessity"] = necessity
        if not geometry_is_straight(necessity):
            report.update(status="rejected", reason="printed_grid_not_straight_and_complete")
            return None, report
        meta = dict(meta, warnings=[], dewarp_necessity=necessity,
                    wave_assessment={"status": "skipped_by_verified_fast_path", "selected": False},
                    fast_path={"selected": True, "version": VERSION})
        report.update(status="selected", selected=True, reason="header_and_straight_printed_grid_verified")
        return (rectified, mx, my, meta), report
    except (ValueError, TypeError, KeyError, IndexError, cv2.error,
            np.linalg.LinAlgError) as exc:
        report.update(status="rejected", reason=f"{type(exc).__name__}: {str(exc)[:200]}")
        return None, report


def math_validates_fast_receipt(result: dict) -> bool:
    """Require real printed-total agreement; never accept unknown math."""
    from App.ocr.receipt_editing import parse_number

    calc = result.get("calculation") or {}
    if calc.get("status") != "match" or calc.get("validation_status") != "match":
        return False
    if (calc.get("vat") or {}).get("status") not in {"match", "not_applicable"}:
        return False
    if parse_number(calc.get("expected_total")) is None:
        return False
    if calc.get("missing_rows") or calc.get("missing_original_total"):
        return False
    products = [r for r in result.get("rows", []) if r.get("kind") == "product"]
    if not products:
        return False
    for row in products:
        if not str(row.get("name_raw", "")).strip() or row.get("invalid_fields"):
            return False
        # Numeric agreement must not conceal a broken multi-line product name.
        if float(row.get("ocr_confidence") or 0) < .60:
            return False
        if parse_number(row.get("quantity")) is None or parse_number(row.get("price")) is None:
            return False
    return True
