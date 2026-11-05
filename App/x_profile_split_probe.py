"""Probe for local x-profile column separation instead of global grid thickening.

This module tries a safer alternative: find the valley between quantity and price
by measuring the dark-pixel profile along x and brightening only the local split
band. It is designed as a testable experiment to see whether the table can be
separated without modifying the actual text glyphs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/cyrillic_PP-OCRv5_rec_mobile.onnx"


def smooth_profile(values, window=5):
    if values.size == 0:
        return values
    kernel = np.ones(window, dtype=np.float32) / float(window)
    padded = np.pad(values.astype(np.float32), (window // 2, window - 1 - (window // 2)), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def find_quantity_price_split(image, threshold=200, x_start_ratio=0.30, x_end_ratio=0.85):
    """Return the x-coordinate of the most likely gap between quantity and price.

    A saltus is considered valid if it is a local minimum in the x-profile and its
    depth is strong enough relative to the local neighborhood.
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

    window = smoothed[lo:hi]
    candidate = int(np.argmin(window)) + lo
    if window.size < 5:
        return None

    local_mean = float(window.mean())
    local_min = float(window[candidate - lo])
    if local_mean <= 0:
        return None
    if (local_mean - local_min) / max(1.0, local_mean) < 0.12:
        return None

    left = max(lo, candidate - 3)
    right = min(hi, candidate + 3)
    local = smoothed[left:right]
    if local.size == 0:
        return None
    if local_min > np.min(local) + 1e-6:
        return None
    return int(candidate)


def apply_split_gap(image, split_x=None, gap_width=3, white_value=255):
    if split_x is None:
        return image.copy()
    processed = image.copy()
    x0 = max(0, int(split_x) - gap_width)
    x1 = min(processed.shape[1], int(split_x) + gap_width + 1)
    processed[:, x0:x1] = white_value
    return processed


def crop_table_region(image, polygon):
    polygon = np.asarray(polygon, dtype=np.float32)
    lo = np.maximum(0, np.floor(polygon.min(axis=0)).astype(int) - 16)
    hi = np.minimum([image.shape[1], image.shape[0]], np.ceil(polygon.max(axis=0)).astype(int) + 17)
    return image[lo[1]:hi[1], lo[0]:hi[0]]


def run_ocr(engine, image):
    temp = Path("__ocr_probe__.png")
    cv2.imwrite(str(temp), image)
    try:
        result, _ = engine(str(temp))
    finally:
        if temp.exists():
            temp.unlink(missing_ok=True)
    lines = []
    for box, text, confidence in result or []:
        lines.append(dict(text=str(text), confidence=float(confidence), box=np.asarray(box, dtype=np.float32).tolist()))
    return lines


def collect_photo_entries():
    summary = json.loads((ROOT / "eval_outputs/rectification_summary.json").read_text(encoding="utf-8"))
    registry = {entry["file"]: entry for entry in summary if "file" in entry}
    files = sorted((ROOT / "data/in_data").glob("*.jpg")) + sorted((ROOT / "data/in_data").glob("*.jpeg")) + sorted((ROOT / "data/in_data").glob("*.png"))
    return [(file, registry.get(file.name)) for file in files]


def evaluate_dataset(limit=None, output_path=ROOT / "debug_outputs/x_profile_split_report.json"):
    engine = RapidOCR(rec_model_path=str(MODEL), rec_img_shape=[3, 48, 320], text_score=0.3)
    entries = collect_photo_entries()
    if limit is not None:
        entries = entries[:limit]

    records = []
    for index, (file, entry) in enumerate(entries, start=1):
        image = cv2.imread(str(file), cv2.IMREAD_COLOR)
        if image is None:
            records.append({"file": file.name, "status": "skip", "reason": "Unreadable"})
            continue

        crop = crop_table_region(image, entry["source_polygon"]) if entry is not None else image
        split_x = find_quantity_price_split(crop)
        enhanced = apply_split_gap(crop, split_x)

        baseline = run_ocr(engine, crop)
        variant = run_ocr(engine, enhanced)

        baseline_count = len(baseline)
        variant_count = len(variant)
        baseline_conf = float(np.mean([line["confidence"] for line in baseline])) if baseline else 0.0
        variant_conf = float(np.mean([line["confidence"] for line in variant])) if variant else 0.0

        records.append(dict(
            file=file.name,
            split_x=split_x,
            status="ok",
            baseline_lines=baseline_count,
            variant_lines=variant_count,
            delta_lines=variant_count - baseline_count,
            baseline_mean_confidence=baseline_conf,
            variant_mean_confidence=variant_conf,
            delta_confidence=variant_conf - baseline_conf,
        ))
        print(f"{index}/{len(entries)} {file.name}: split={split_x} baseline={baseline_count} variant={variant_count} avg_conf={baseline_conf:.3f}->{variant_conf:.3f}")

    summary = {
        "files_tested": len([record for record in records if record["status"] == "ok"]),
        "worse": sum(1 for record in records if record["status"] == "ok" and record["delta_lines"] < 0),
        "better": sum(1 for record in records if record["status"] == "ok" and record["delta_lines"] > 0),
        "unchanged": sum(1 for record in records if record["status"] == "ok" and record["delta_lines"] == 0),
        "mean_delta_lines": float(np.mean([record["delta_lines"] for record in records if record["status"] == "ok"])),
        "mean_delta_confidence": float(np.mean([record["delta_confidence"] for record in records if record["status"] == "ok"])),
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate the local x-profile gap strategy as a safer alternative to grid thickening.")
    parser.add_argument("--limit", type=int, default=None, help="Optional quick smoke test limit.")
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
