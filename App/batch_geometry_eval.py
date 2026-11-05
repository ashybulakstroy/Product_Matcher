from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from App.ocr_geometry import draw_debug_overlay, order_corners, perspective_warp
from App.table_boundary import detect_table_boundary


def empty_rectangularity_metrics() -> dict:
    return {
        "side_lengths": [],
        "mean_side_length": 0.0,
        "width_ratio": 0.0,
        "angle_dev_deg": 0.0,
        "rectangularity_score": 0.0,
    }

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "in_data"
OUT_DIR = Path(__file__).resolve().parent.parent / "eval_outputs"


def compute_rectangularity(corners: np.ndarray) -> dict:
    pts = np.asarray(corners, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError(f"Expected 4 corners, got {pts.shape}")

    edges = [
        (pts[0], pts[1]),
        (pts[1], pts[2]),
        (pts[2], pts[3]),
        (pts[3], pts[0]),
    ]
    lengths = np.asarray([np.linalg.norm(b - a) for a, b in edges], dtype=np.float32)
    mean_len = float(np.mean(lengths))
    width_ratio = float(np.min(lengths) / (np.max(lengths) + 1e-8))

    angles = []
    for i in range(4):
        a, b = edges[i]
        c, d = edges[(i + 1) % 4]
        v1 = b - a
        v2 = d - c
        angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8), -1.0, 1.0)))
        angles.append(angle)

    angle_dev = float(np.mean(np.abs(np.asarray(angles) - 90.0)))
    score = 1.0 - min(1.0, (angle_dev / 25.0) * 0.6 + (1.0 - width_ratio) * 0.4)
    score = max(0.0, min(1.0, score))

    return {
        "side_lengths": [round(float(x), 2) for x in lengths],
        "mean_side_length": round(mean_len, 2),
        "width_ratio": round(float(width_ratio), 3),
        "angle_dev_deg": round(angle_dev, 2),
        "rectangularity_score": round(score, 3),
    }


def should_generate_warp(corners: np.ndarray, image_shape: tuple[int, int] | None = None) -> bool:
    pts = np.asarray(corners, dtype=np.float32)
    if pts.shape != (4, 2):
        return False
    if not np.isfinite(pts).all():
        return False
    if not cv2.isContourConvex(pts):
        return False

    edges = [
        (pts[0], pts[1]),
        (pts[1], pts[2]),
        (pts[2], pts[3]),
        (pts[3], pts[0]),
    ]
    lengths = np.asarray([np.linalg.norm(b - a) for a, b in edges], dtype=np.float32)
    if np.any(lengths <= 5.0):
        return False

    width_ratio = float(np.min(lengths) / (np.max(lengths) + 1e-8))
    if width_ratio < 0.15:
        return False

    metrics = compute_rectangularity(pts)
    if metrics["rectangularity_score"] < 0.75 or metrics["angle_dev_deg"] > 10.0:
        return False

    if image_shape is not None:
        h, w = image_shape[:2]
        if np.any(pts[:, 0] < -5) or np.any(pts[:, 0] > w + 5) or np.any(pts[:, 1] < -5) or np.any(pts[:, 1] > h + 5):
            return False

    return True


def recommend_method(rect_score: float) -> str:
    if rect_score >= 0.85:
        return "original_or_deskew"
    if rect_score >= 0.70:
        return "deskew"
    if rect_score >= 0.55:
        return "perspective"
    return "needs_review"


def evaluate_image(path: Path | str) -> dict:
    path = Path(path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)

    boundary = detect_table_boundary(image)
    polygon = boundary["polygon"]
    corners = np.asarray(polygon if polygon is not None else [], dtype=np.float32).reshape(-1, 2)
    if corners.shape == (4, 2):
        corners = order_corners(corners)
    if corners.shape != (4, 2) or boundary["status"] != "detected":
        metrics = empty_rectangularity_metrics()
        method = "needs_review"
        safe_to_warp = False
        overlay = draw_debug_overlay(image, corners, title=f"{method}")
        warped = image.copy()

        out_dir = OUT_DIR / path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / "overlay.png"), overlay)
        if (out_dir / "warped.png").exists() and not (out_dir / "rectification.json").exists():
            (out_dir / "warped.png").unlink()

        result = {
            "file": path.name,
            "size": [int(image.shape[1]), int(image.shape[0])],
            "corners": [[int(x), int(y)] for x, y in corners],
            "method": method,
            "metrics": metrics,
            "safe_to_warp": safe_to_warp,
            "boundary_status": boundary["status"],
            "warnings": boundary["warnings"],
            "algorithm_version": boundary["algorithm_version"],
        }
        (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    metrics = compute_rectangularity(corners)
    method = recommend_method(metrics["rectangularity_score"])

    safe_to_warp = should_generate_warp(corners, image.shape[:2])
    if not safe_to_warp:
        method = "needs_review"

    overlay = draw_debug_overlay(image, corners, title=f"{method}")
    warped = image.copy() if not safe_to_warp else perspective_warp(image, corners)

    out_dir = OUT_DIR / path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "overlay.png"), overlay)
    if safe_to_warp and not (out_dir / "rectification.json").exists():
        cv2.imwrite(str(out_dir / "warped.png"), warped)
    elif (out_dir / "warped.png").exists() and not (out_dir / "rectification.json").exists():
        (out_dir / "warped.png").unlink()

    result = {
        "file": path.name,
        "size": [int(image.shape[1]), int(image.shape[0])],
        "corners": [[int(x), int(y)] for x, y in corners],
        "method": method,
        "metrics": metrics,
        "safe_to_warp": safe_to_warp,
        "boundary_status": boundary["status"],
        "warnings": boundary["warnings"],
        "algorithm_version": boundary["algorithm_version"],
    }
    (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate document geometry across a set of photo receipts.")
    parser.add_argument("--workers", type=int, default=6, help="Number of worker processes to use; default is 6.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in DATA_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})

    workers = min(args.workers, len(files) if files else 1, os.cpu_count() or 1)
    workers = max(1, workers)

    if len(files) <= 1 or workers == 1:
        summary = [evaluate_image(path) for path in files]
    else:
        with mp.Pool(processes=workers) as pool:
            summary = pool.map(evaluate_image, [str(path) for path in files])

    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps([
        {"file": item["file"], "method": item["method"], "score": item["metrics"]["rectangularity_score"]}
        for item in summary
    ], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
