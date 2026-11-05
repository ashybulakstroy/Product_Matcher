"""Experimental preprocessing for vertical table-grid enhancement before OCR.

This filter targets the "thickened vertical lines" idea from the grid-separation
hypothesis: it darkens and slightly widens near-vertical dark strokes while leaving
most text glyphs untouched. The goal is to reduce the chance of a grid boundary
being treated as text, not to aggressively transform the whole document.
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


def enhance_vertical_grid(image, line_width=2, threshold=185, strength=0.90, max_component_width=7, min_component_height_ratio=0.18, risk_fraction=0.08):
    """Apply a conservative vertical-grid enhancement only for thin, elongated dark rules.

    The filter is deliberately narrow: it keeps only connected components that look
    like a printed rule (very thin vertical strokes spanning a large fraction of the
    row height) and ignores broad or diagonal dark shapes that are more likely text.
    If the candidate mask is too large or too text-like, it bails out and returns the
    original image.
    """
    if image is None:
        raise ValueError("Image is required.")
    rgb = image.copy()
    if image.ndim == 2:
        gray = image.astype(np.uint8)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    dark = blur < threshold
    if not np.any(dark):
        return rgb

    # Keep only very thin dark structures with strong vertical elongation.
    vertical_strokes = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 1), np.uint8))
    vertical_strokes = cv2.morphologyEx(vertical_strokes, cv2.MORPH_OPEN, np.ones((1, max(1, line_width)), np.uint8))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(vertical_strokes, connectivity=8)
    mask = np.zeros_like(vertical_strokes, dtype=np.uint8)
    height = labels.shape[0]
    width = labels.shape[1]
    min_height = max(8, int(height * min_component_height_ratio))

    for label_index in range(1, num_labels):
        x, y, w, h, area = stats[label_index]
        if w <= 0 or h <= 0:
            continue
        if w > max_component_width:
            continue
        if h < min_height:
            continue
        if w / max(1, h) > 0.18:
            continue
        mask[y:y + h, x:x + w] |= (labels[y:y + h, x:x + w] == label_index)

    if not np.any(mask):
        return rgb

    if mask.mean() > risk_fraction:
        return rgb

    kernel = np.ones((1, max(1, line_width)), dtype=np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    if image.ndim == 2:
        processed = rgb.copy()
        processed[mask > 0] = np.clip(processed[mask > 0] * strength, 0, 255).astype(np.uint8)
        return processed

    masked = np.repeat(mask[:, :, None], 3, axis=2)
    processed = rgb.copy()
    processed[masked > 0] = np.clip(processed[masked > 0] * strength, 0, 255).astype(np.uint8)
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
    mapped = []
    for file in files:
        if file.name in registry:
            mapped.append((file, registry[file.name]))
        else:
            mapped.append((file, None))
    return mapped


def evaluate_dataset(limit=None, output_path=ROOT / "debug_outputs/grid_preprocessing_report.json"):
    engine = RapidOCR(rec_model_path=str(MODEL), rec_img_shape=[3, 48, 320], text_score=0.3)
    photo_entries = collect_photo_entries()
    if limit is not None:
        photo_entries = photo_entries[:limit]

    records = []
    for index, (file, entry) in enumerate(photo_entries, start=1):
        image = cv2.imread(str(file), cv2.IMREAD_COLOR)
        if image is None:
            records.append({"file": file.name, "status": "skip", "reason": "Unreadable"})
            continue

        if entry is not None:
            crop = crop_table_region(image, entry["source_polygon"])
        else:
            crop = image

        baseline = run_ocr(engine, crop)
        enhanced = enhance_vertical_grid(crop)
        variant = run_ocr(engine, enhanced)

        baseline_count = len(baseline)
        variant_count = len(variant)
        baseline_conf = float(np.mean([line["confidence"] for line in baseline])) if baseline else 0.0
        variant_conf = float(np.mean([line["confidence"] for line in variant])) if variant else 0.0

        records.append(dict(
            file=file.name,
            status="ok",
            baseline_lines=baseline_count,
            variant_lines=variant_count,
            delta_lines=variant_count - baseline_count,
            baseline_mean_confidence=baseline_conf,
            variant_mean_confidence=variant_conf,
            delta_confidence=variant_conf - baseline_conf,
        ))
        print(f"{index}/{len(photo_entries)} {file.name}: baseline={baseline_count} variant={variant_count} avg_conf={baseline_conf:.3f}->{variant_conf:.3f}")

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
    parser = argparse.ArgumentParser(description="Evaluate grid-line thickening as a preprocessing variant before OCR.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for a quick smoke test.")
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
