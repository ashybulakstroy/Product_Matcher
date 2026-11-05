from __future__ import annotations

import os
from typing import Iterable

import cv2
import numpy as np


DEFAULT_DEBUG_DIR = os.path.join(os.getcwd(), "debug_outputs")


def clamp_corners_to_image(corners: np.ndarray, image_shape: tuple[int, int] | None) -> np.ndarray:
    """Clip corner coordinates to the image frame so debug overlays do not extend outside it."""
    pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    if image_shape is None:
        return pts

    h, w = image_shape[:2]
    clipped = pts.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, max(w - 1, 0))
    clipped[:, 1] = np.clip(clipped[:, 1], 0, max(h - 1, 0))
    return clipped


def order_corners(corners: np.ndarray) -> np.ndarray:
    """Return corners in consistent order: top-left, top-right, bottom-right, bottom-left.

    The top edge is the receipt boundary closest to the image top. Sorting by angle around the
    centroid can mis-order these points on trapezoid-like receipt contours, so we choose the
    topmost two corners and bottommost two corners explicitly and then arrange them left-to-right.
    """
    pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 4:
        raise ValueError("Expected at least 4 corner points.")

    unique_pts = []
    for pt in pts:
        if not any(np.linalg.norm(pt - other) < 1e-3 for other in unique_pts):
            unique_pts.append(pt)

    if len(unique_pts) < 4:
        min_x = float(np.min(pts[:, 0]))
        min_y = float(np.min(pts[:, 1]))
        max_x = float(np.max(pts[:, 0]))
        max_y = float(np.max(pts[:, 1]))
        return np.array([
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y],
        ], dtype=np.float32)

    pts = np.asarray(unique_pts, dtype=np.float32)
    ordered_by_y = pts[np.argsort(pts[:, 1], kind="mergesort")]
    top_points = ordered_by_y[: min(2, ordered_by_y.shape[0])]
    bottom_points = ordered_by_y[-min(2, ordered_by_y.shape[0]):]

    top_left = top_points[np.argmin(top_points[:, 0])]
    top_right = top_points[np.argmax(top_points[:, 0])]
    bottom_left = bottom_points[np.argmin(bottom_points[:, 0])]
    bottom_right = bottom_points[np.argmax(bottom_points[:, 0])]

    return np.array([
        top_left,
        top_right,
        bottom_right,
        bottom_left,
    ], dtype=np.float32)


def detect_table_polygon(image: np.ndarray) -> np.ndarray:
    """Compatibility entry point; a frame fallback is not a detected table."""
    from App.table_boundary import detect_table_boundary

    result = detect_table_boundary(image)
    if result["polygon"] is not None:
        return order_corners(result["polygon"]) if len(result["polygon"]) == 4 else result["polygon"]
    h, w = image.shape[:2]
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)


def detect_table_corners(image: np.ndarray) -> np.ndarray:
    """Legacy name: returns the same complete polygon as detect_table_polygon."""
    return detect_table_polygon(image)


def perspective_warp(image: np.ndarray, corners: np.ndarray, width: int | None = None, height: int | None = None) -> np.ndarray:
    """Apply a perspective warp only for a real quadrilateral; complex bent receipt polygons are kept as-is."""
    pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    if pts.shape != (4, 2) or not np.isfinite(pts).all() or not cv2.isContourConvex(pts):
        return image.copy()

    corners = order_corners(pts)
    src = corners

    tl, tr, br, bl = src
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(round(width_a)), int(round(width_b)), 1)

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(round(height_a)), int(round(height_b)), 1)

    if width is None:
        width = max_width
    if height is None:
        height = max_height

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_LINEAR)
    return warped


def draw_debug_overlay(image: np.ndarray, corners: np.ndarray, title: str = "detected") -> np.ndarray:
    overlay = image.copy()
    pts = clamp_corners_to_image(np.asarray(corners, dtype=np.float32), image.shape[:2]).astype(np.int32)
    if len(pts):
        cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)
    for i, pt in enumerate(pts if len(pts) == 4 else []):
        cv2.circle(overlay, tuple(pt), 8, (0, 0, 255), -1)
        cv2.putText(overlay, str(i), tuple(pt), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(overlay, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    return overlay


def rectify_receipt(image_bytes: bytes, *, mode: str = "auto", manual_corners: Iterable[tuple[int, int]] | None = None, options: dict | None = None) -> dict:
    """Detect an observed boundary; complex or uncertain shapes keep the original."""
    from App.table_boundary import BoundaryOptions, detect_table_boundary
    from App.batch_geometry_eval import should_generate_warp

    if mode not in {"auto", "original", "none", "perspective", "manual"}:
        raise ValueError(f"Unknown geometry mode: {mode}")
    if mode == "manual" and manual_corners is None:
        raise ValueError("manual mode requires manual_corners")
    options = options or {}
    image_array = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image_array is None:
        raise ValueError("Unable to decode input image bytes.")

    manual = manual_corners is not None
    if manual:
        polygon = np.asarray(list(manual_corners), dtype=np.float32)
        if polygon.shape != (4, 2) or not np.isfinite(polygon).all():
            raise ValueError("manual_corners must contain exactly four finite (x, y) points.")
        if len(np.unique(polygon, axis=0)) != 4:
            raise ValueError("manual_corners must contain four distinct points")
        polygon = order_corners(polygon)
        if not cv2.isContourConvex(polygon) or cv2.contourArea(polygon) <= 1:
            raise ValueError("manual_corners must form a non-degenerate convex quadrilateral")
        boundary = {"status": "detected", "warnings": [], "algorithm_version": "manual"}
    else:
        boundary = detect_table_boundary(image_array, BoundaryOptions(**options.get("boundary", {})))
        polygon = boundary["polygon"]
        if polygon is not None and len(polygon) == 4:
            polygon = order_corners(polygon)

    quadrilateral = polygon is not None and polygon.shape == (4, 2)
    warnings = list(boundary["warnings"])
    output, method, status = image_array, "none", "original_fallback"
    if mode not in {"original", "none"}:
        if quadrilateral and (manual or (boundary["status"] == "detected" and should_generate_warp(polygon, image_array.shape[:2]))):
            output = perspective_warp(image_array, polygon)
            method, status = "perspective", "manual_rectified" if manual else "rectified"
        else:
            status = "needs_review"
            if polygon is not None and not quadrilateral:
                warnings.append("complex_boundary_requires_review")
            elif quadrilateral and boundary["status"] == "detected":
                warnings.append("unsafe_perspective_geometry")

    return {
        "image": output,
        "original_image": image_array,
        "status": status,
        "method": method,
        "source_polygon": None if polygon is None else polygon.tolist(),
        "source_corners": polygon.tolist() if quadrilateral else None,
        "algorithm_version": boundary["algorithm_version"],
        "original_size": list(image_array.shape[:2]),
        "output_size": list(output.shape[:2]),
        "quality_metrics": {
            "manual_override": manual,
            "detected_four_corners": quadrilateral,
            "boundary_status": boundary["status"],
        },
        "warnings": warnings,
    }


def save_debug_outputs(image_path: str, out_dir: str = DEFAULT_DEBUG_DIR) -> dict:
    """Detect corners, draw visual overlay and save the transformed result for comparison."""
    os.makedirs(out_dir, exist_ok=True)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    from pathlib import Path
    result = rectify_receipt(Path(image_path).read_bytes())
    polygon = np.asarray(result["source_polygon"] or [], dtype=np.float32).reshape(-1, 2)
    overlay = draw_debug_overlay(image, polygon, title=result["status"])
    corners = polygon
    warped = result["image"]

    stem = os.path.splitext(os.path.basename(image_path))[0]
    overlay_path = os.path.join(out_dir, f"{stem}_overlay.png")
    warped_path = os.path.join(out_dir, f"{stem}_warped.png")

    cv2.imwrite(overlay_path, overlay)
    cv2.imwrite(warped_path, warped)

    return {
        "overlay": overlay_path,
        "warped": warped_path,
        "corners": corners.tolist(),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect receipt/table corners and save debug outputs.")
    parser.add_argument("image_path", help="Path to the input receipt photo.")
    parser.add_argument("--out-dir", default=DEFAULT_DEBUG_DIR, help="Directory for debug artifacts.")
    args = parser.parse_args()

    result = save_debug_outputs(args.image_path, out_dir=args.out_dir)
    print(result)
