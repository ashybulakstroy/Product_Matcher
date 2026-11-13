"""Choose validated wave-grid rectification or the established contour fallback."""
from __future__ import annotations

import os

import cv2
import numpy as np

from App.ocr.contour_unwarp import build_contour_map
from App.ocr.grid_curvature_v2 import (VERSION as GRID_V2_VERSION,
                                       diagnose as grid_v2_diagnose,
                                       rectify_candidate as grid_v2_rectify)
from App.ocr.ocr_catalog_eval import table_layout
from App.ocr.wave_dewarp import find_rules, rectify
from App.ocr.dewarp_necessity_detector import assess_dewarp_necessity


def grid_v2_mode():
    """Return ``off``, ``shadow`` or ``select`` for the V2 experiment."""
    value = str(os.getenv("OCR_GRID_CURVATURE_V2", "select")).strip().lower()
    if value in {"0", "off", "false", "no", "disabled"}:
        return "off"
    if value in {"shadow", "report", "diagnose"}:
        return "shadow"
    if value in {"1", "on", "true", "yes", "select"}:
        return "select"
    return "off"


def _grid_v2_candidate(image, polygon, baseline, baseline_map_x, metadata):
    """Build a V2 candidate without raising; keep the established result safe."""
    margin = int(metadata.get("margin", 16))
    polygon = np.asarray(polygon, np.float32)
    lo = np.maximum(0, np.floor(polygon.min(axis=0)).astype(int) - margin)
    hi = np.minimum([image.shape[1], image.shape[0]],
                    np.ceil(polygon.max(axis=0)).astype(int) + margin + 1)
    crop = image[lo[1]:hi[1], lo[0]:hi[0]]
    result = {"mode": grid_v2_mode(), "status": "not_started"}
    try:
        result["before"] = grid_v2_diagnose(crop)
        if result["before"].get("decision") != "DEWARP":
            result["status"] = "not_needed"
            return result
        layout = table_layout(baseline, metadata)
        layout.setdefault("body_left", margin)
        output, map_x, map_y, grid, report = grid_v2_rectify(
            image, polygon, layout, baseline_map_x, margin=margin)
        result.update(status="candidate_ready", output=output, map_x=map_x,
                      map_y=map_y, grid=grid, after=report)
    except (ValueError, KeyError, IndexError, TypeError, cv2.error,
            np.linalg.LinAlgError) as error:
        result.update(status="rejected", reason=str(error))
    return result


def _v2_quality(report):
    """Jacobian range of the applied warp (2D when available)."""
    quality = report.get("quality_2d") or {}
    return (quality.get("min_jacobian", report.get("min_jacobian")),
            quality.get("max_jacobian", report.get("max_jacobian")))


def _v2_metadata(metadata, candidate, polygon, warnings):
    """Return a JSON-safe selected V2 metadata block."""
    margin = int(metadata.get("margin", 16))
    output = candidate["output"]
    details = candidate["after"]
    min_jacobian, max_jacobian = _v2_quality(details)
    result = dict(metadata)
    result.update(
        method="grid_curvature_v2",
        algorithm_version=GRID_V2_VERSION,
        margin=margin,
        body_size=[int(output.shape[1] - 2 * margin),
                   int(output.shape[0] - 2 * margin)],
        grid_layout=candidate["grid"],
        grid_curvature=candidate["before"],
        grid_curvature_candidate=details,
        grid_curvature_assessment=dict(
            selected=True,
            status="selected",
            reason="V2 подтвердил согласованный изгиб и безопасное выпрямление",
            min_jacobian=min_jacobian,
            max_jacobian=max_jacobian,
            vertical_status=details.get("vertical_status"),
            method=details.get("method"),
        ),
        source_polygon=np.asarray(polygon).tolist(),
        warnings=list(warnings),
    )
    return result


def _attach_v2_report(metadata, candidate):
    if not candidate or "before" not in candidate:
        return
    metadata["grid_curvature"] = candidate["before"]
    if candidate.get("status") == "candidate_ready":
        details = candidate["after"]
        min_jacobian, max_jacobian = _v2_quality(details)
        metadata["grid_curvature_candidate"] = details
        metadata["grid_curvature_assessment"] = dict(
            selected=False,
            status="candidate_ready",
            reason="Кандидат V2 рассчитан, но сохранён прежний результат",
            min_jacobian=min_jacobian,
            max_jacobian=max_jacobian,
            vertical_status=details.get("vertical_status"),
        )
    elif candidate.get("status") == "not_needed":
        metadata["grid_curvature_assessment"] = dict(
            selected=False, status="not_needed",
            reason=candidate["before"].get("reason", "V2 не подтвердил изгиб исходной таблицы"),
        )
    elif candidate.get("status") == "rejected":
        metadata["grid_curvature_assessment"] = dict(
            selected=False, status="rejected",
            reason=candidate.get("reason", "Кандидат V2 отклонён"),
        )


def rectify_table(image, polygon, *, trust_top=True):
    polygon = np.asarray(polygon, np.float32)
    warnings = []
    contour_valid = True
    try:
        mx, my, metadata = build_contour_map(polygon, trust_top=trust_top)
        baseline = cv2.remap(image, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    except ValueError:
        contour_valid = False
        lo = np.maximum(0, np.floor(polygon.min(axis=0)).astype(int) - 16)
        hi = np.minimum([image.shape[1], image.shape[0]], np.ceil(polygon.max(axis=0)).astype(int) + 17)
        baseline = image[lo[1]:hi[1], lo[0]:hi[0]].copy()
        h, w = baseline.shape[:2]
        yy, xx = np.indices((h, w), dtype=np.float32)
        mx, my = xx + float(lo[0]), yy + float(lo[1])
        metadata = dict(method="original_crop_fallback", margin=16, body_size=[max(16, w - 32), max(16, h - 32)])
        warnings.append("Выпрямление не удалось проверить: OCR выполнен по исходной области таблицы.")

    mode = grid_v2_mode()
    candidate = None
    if contour_valid and mode != "off":
        candidate = _grid_v2_candidate(image, polygon, baseline, mx, metadata)

    assessment = dict(status="not_selected", reason="Сохранён проверенный результат по контуру")
    try:
        necessity = assess_dewarp_necessity(baseline)
        metadata["dewarp_necessity"] = necessity
        # UNCERTAIN and contour fallback preserve the established validator.
        if contour_valid and necessity["decision"] == "SKIP":
            assessment = dict(
                status="skipped_not_needed",
                selected=False,
                reason=necessity["reason"],
            )
            metadata.update(
                source_polygon=polygon.tolist(),
                wave_assessment=assessment,
                warnings=warnings,
            )
            _attach_v2_report(metadata, candidate)
            return baseline, mx.astype(np.float32), my.astype(np.float32), metadata
    except (ValueError, np.linalg.LinAlgError, cv2.error) as error:
        warnings.append(f"Не удалось проверить необходимость Dewarp: {error}")

    try:
        output, wx, wy, wave, _ = rectify(image, polygon)
        assessment = dict(wave, selected=False)
        if wave["status"] == "rectified" and len(wave.get("vertical_levels", [])) == 5:
            horizontal, _ = find_rules(baseline)
            vertical, _ = find_rules(baseline, vertical=True)
            previous_h = max([rule.curvature for rule in horizontal], default=float("inf"))
            previous_v = max([rule.curvature for rule in vertical], default=float("inf"))
            current_h = wave["straightness_check"]["horizontal"]["after_px"]
            current_v = wave["straightness_check"]["vertical"]["after_px"]
            better = metadata["method"] == "original_crop_fallback" or (previous_h > 2 and current_h < previous_h * .75 and previous_h - current_h > 1)
            safe_columns = current_v <= max(previous_v + .7, previous_v * 1.12)
            assessment["baseline_comparison"] = dict(horizontal_before_px=previous_h if np.isfinite(previous_h) else None,
                vertical_before_px=previous_v if np.isfinite(previous_v) else None,
                horizontal_after_px=current_h, vertical_after_px=current_v)
            if better and safe_columns:
                assessment["selected"] = True
                xs, ys = wave["vertical_levels"], wave["horizontal_levels"]
                layout = dict(body_left=xs[0], name_left=xs[1], name_right=xs[2], quantity_right=xs[3], right=xs[4],
                              body_top=ys[0], body_bottom=ys[-1], row_boundaries=ys, grid_confirmed=True)
                result = dict(method="wave_grid", algorithm_version=wave["version"], margin=int(round(ys[0])),
                    body_size=[int(round(xs[-1] - xs[0])), int(round(ys[-1] - ys[0]))],
                    grid_layout=layout, wave_assessment=assessment, source_polygon=polygon.tolist(), warnings=[])
                _attach_v2_report(result, candidate)
                return output, wx, wy, result
            assessment["reason"] = "Результат по контуру уже достаточно ровный или новый вариант не улучшает его безопасно"
        elif wave["status"] == "rectified":
            assessment["reason"] = "Не подтверждены пять границ четырёх столбцов чека"
    except (ValueError, np.linalg.LinAlgError, cv2.error) as error:
        assessment = dict(status="rejected", reason=str(error))
    metadata.update(source_polygon=polygon.tolist(), wave_assessment=assessment, warnings=warnings)
    _attach_v2_report(metadata, candidate)
    if mode == "select" and candidate and candidate.get("status") == "candidate_ready":
        return (candidate["output"], candidate["map_x"], candidate["map_y"],
                _v2_metadata(metadata, candidate, polygon, warnings))
    return baseline, mx.astype(np.float32), my.astype(np.float32), metadata
