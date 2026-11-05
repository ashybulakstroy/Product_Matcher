"""Choose validated wave-grid rectification or the established contour fallback."""
from __future__ import annotations

import cv2
import numpy as np

from App.contour_unwarp import build_contour_map
from App.wave_dewarp import find_rules, rectify
from App.dewarp_necessity_detector import assess_dewarp_necessity


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
                return output, wx, wy, result
            assessment["reason"] = "Результат по контуру уже достаточно ровный или новый вариант не улучшает его безопасно"
        elif wave["status"] == "rectified":
            assessment["reason"] = "Не подтверждены пять границ четырёх столбцов чека"
    except (ValueError, np.linalg.LinAlgError, cv2.error) as error:
        assessment = dict(status="rejected", reason=str(error))
    metadata.update(source_polygon=polygon.tolist(), wave_assessment=assessment, warnings=warnings)
    return baseline, mx.astype(np.float32), my.astype(np.float32), metadata
