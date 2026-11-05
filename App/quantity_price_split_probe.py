"""Experimental quantity/price separator probe.

This module is intentionally isolated from the production pipeline. It tests a
safer alternative to grid-line thickening: instead of modifying grid strokes, it
looks for a local x-profile minimum between the quantity and price columns, which
should correspond to the real white gap between them.

Decision rule:
- If the feature shows no absolute regressions and real gains on the full dataset,
  it may be considered for integration.
- Otherwise it stays experimental.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/cyrillic_PP-OCRv5_rec_mobile.onnx"


def smooth_profile(values, window=7):
    if values.size == 0:
        return values
    kernel = np.ones(window, dtype=np.float32) / float(window)
    pad = window // 2
    padded = np.pad(values.astype(np.float32), (pad, window - 1 - pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def detect_quantity_price_gap(image, threshold=200, x_start_ratio=0.28, x_end_ratio=0.88):
    """Return the x coordinate of the real quantity/price gap if one exists.

    The split is only accepted when the x-profile shows a sufficiently deep local
    minimum in the right region. When no real whitespace gap is present, this
    returns None so the caller can skip the preprocessing entirely.
    """
    if image is None:
        raise ValueError("Image is required.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.astype(np.uint8)
    dark = gray < threshold
    profile = dark.sum(axis=0).astype(np.float32)
    if profile.size == 0:
        return None

    smoothed = smooth_profile(profile, window=7)
    width = smoothed.shape[0]
    lo = max(0, int(width * x_start_ratio))
    hi = min(width, int(width * x_end_ratio))
    if hi <= lo:
        return None

    segment = smoothed[lo:hi]
    if segment.size < 5:
        return None

    best_local = int(np.argmin(segment))
    if best_local < 2 or best_local > segment.size - 3:
        return None

    best = best_local + lo
    local_min = float(segment[best_local])
    local_mean = float(segment.mean())
    if local_mean <= 0:
        return None

    gap_depth = (local_mean - local_min) / max(1.0, local_mean)
    if gap_depth < 0.18:
        return None

    left_start = max(0, best_local - 4)
    right_end = min(segment.size, best_local + 5)
    local_window = segment[left_start:right_end]
    if local_window.size < 3:
        return None

    left_vals = segment[max(0, best_local - 4):best_local]
    right_vals = segment[best_local + 1:min(segment.size, best_local + 5)]
    if left_vals.size == 0 or right_vals.size == 0:
        return None

    left_mean = float(left_vals.mean())
    right_mean = float(right_vals.mean())
    valley_strength = (left_mean + right_mean - 2.0 * local_min) / max(1.0, left_mean + right_mean)
    if valley_strength < 0.10:
        return None
    if local_min > min(left_mean, right_mean) * 0.8:
        return None

    left = max(lo, best - 3)
    right = min(hi, best + 3)
    local_profile = smoothed[left:right]
    if local_profile.size == 0:
        return None
    if local_min > float(np.min(local_profile)) + 1e-6:
        return None

    return int(best)


def find_quantity_price_gap(image, threshold=200, x_start_ratio=0.28, x_end_ratio=0.88):
    """Backward-compatible alias for the guarded split detector."""
    return detect_quantity_price_gap(image, threshold=threshold, x_start_ratio=x_start_ratio, x_end_ratio=x_end_ratio)


def should_apply_quantity_price_split(image, threshold=200, x_start_ratio=0.28, x_end_ratio=0.88):
    return detect_quantity_price_gap(image, threshold=threshold, x_start_ratio=x_start_ratio, x_end_ratio=x_end_ratio) is not None


def apply_gap_highlight(image, split_x=None, gap_width=4):
    if split_x is None:
        return image.copy()
    if not 0 <= int(split_x) < image.shape[1]:
        return image.copy()
    processed = image.copy()
    x0 = max(0, int(split_x) - gap_width)
    x1 = min(processed.shape[1], int(split_x) + gap_width + 1)
    processed[:, x0:x1] = 255
    return processed


def run_ocr(engine, image):
    temp = Path("__ocr_probe__.png")
    cv2.imwrite(str(temp), image)
    try:
        result, _ = engine(str(temp))
    finally:
        if temp.exists():
            temp.unlink(missing_ok=True)
    return result or []


def crop_table_region(image, polygon):
    polygon = np.asarray(polygon, dtype=np.float32)
    lo = np.maximum(0, np.floor(polygon.min(axis=0)).astype(int) - 16)
    hi = np.minimum([image.shape[1], image.shape[0]], np.ceil(polygon.max(axis=0)).astype(int) + 17)
    return image[lo[1]:hi[1], lo[0]:hi[0]]


def collect_photo_files():
    in_data = ROOT / "data/in_data"
    return sorted(p for p in in_data.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})


def evaluate_dataset(limit=None, output_path=ROOT / "debug_outputs/quantity_price_split_probe_report.json"):
    engine = RapidOCR(rec_model_path=str(MODEL), rec_img_shape=[3, 48, 320], text_score=0.3)
    records = []
    summary = json.loads((ROOT / "eval_outputs/rectification_summary.json").read_text(encoding="utf-8")) if (ROOT / "eval_outputs/rectification_summary.json").exists() else []
    registry = {entry["file"]: entry for entry in summary if "file" in entry}
    files = collect_photo_files()
    if limit is not None:
        files = files[:limit]

    for index, file in enumerate(files, start=1):
        image = cv2.imread(str(file), cv2.IMREAD_COLOR)
        if image is None:
            records.append({"file": file.name, "status": "skip", "reason": "Unreadable"})
            continue

        entry = registry.get(file.name)
        crop = crop_table_region(image, entry["source_polygon"]) if entry is not None else image
        split_x = detect_quantity_price_gap(crop)
        applied = split_x is not None
        enhanced = apply_gap_highlight(crop, split_x) if applied else crop.copy()

        baseline = run_ocr(engine, crop)
        variant = run_ocr(engine, enhanced)

        baseline_count = len(baseline)
        variant_count = len(variant)
        baseline_conf = float(np.mean([float(item[2]) for item in baseline])) if baseline else 0.0
        variant_conf = float(np.mean([float(item[2]) for item in variant])) if variant else 0.0

        records.append(dict(
            file=file.name,
            split_x=split_x,
            applied=applied,
            status="ok",
            baseline_lines=baseline_count,
            variant_lines=variant_count,
            delta_lines=variant_count - baseline_count,
            baseline_mean_confidence=baseline_conf,
            variant_mean_confidence=variant_conf,
            delta_confidence=variant_conf - baseline_conf,
        ))
        print(f"{index}/{len(files)} {file.name}: split={split_x} baseline={baseline_count} variant={variant_count} avg_conf={baseline_conf:.3f}->{variant_conf:.3f}")

    summary_payload = {
        "files_tested": len([record for record in records if record["status"] == "ok"]),
        "worse": sum(1 for record in records if record["status"] == "ok" and record["delta_lines"] < 0),
        "better": sum(1 for record in records if record["status"] == "ok" and record["delta_lines"] > 0),
        "unchanged": sum(1 for record in records if record["status"] == "ok" and record["delta_lines"] == 0),
        "mean_delta_lines": float(np.mean([record["delta_lines"] for record in records if record["status"] == "ok"])),
        "mean_delta_confidence": float(np.mean([record["delta_confidence"] for record in records if record["status"] == "ok"])),
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_payload


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a safer quantity/price gap heuristic without modifying the table grid.")
    parser.add_argument("--limit", type=int, default=None, help="Optional quick smoke-test limit.")
    args = parser.parse_args()
    summary = evaluate_dataset(limit=args.limit)
    print("\nSummary:")
    print(json.dumps({
        "files_tested": summary["files_tested"],
        "worse": summary["worse"],
        "better": summary["better"],
        "unchanged": summary["unchanged"],
        "mean_delta_lines": round(summary["mean_delta_lines"], 2),
        "mean_delta_confidence": round(summary["mean_delta_confidence"], 4),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
