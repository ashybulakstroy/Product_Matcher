from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "in_data"
EVAL_DIR = ROOT / "eval_outputs"


def build_review_rows(summary_path: str | Path = EVAL_DIR / "summary.json") -> list[dict]:
    summary_path = Path(summary_path)
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for item in summary:
        file_name = str(item["file"])
        stem = Path(file_name).stem
        base_dir = EVAL_DIR / stem
        image_path = DATA_DIR / file_name
        overlay_path = base_dir / "overlay.png"
        warped_path = base_dir / "warped.png"
        rows.append(
            {
                "file": file_name,
                "method": item.get("method", "unknown"),
                "score": float(item.get("metrics", {}).get("rectangularity_score", 0.0)),
                "status": "auto" if item.get("method") else "unknown",
                "image_exists": image_path.exists(),
                "overlay_exists": overlay_path.exists(),
                "warped_exists": warped_path.exists(),
                "overlay_path": str(overlay_path.relative_to(ROOT)) if overlay_path.exists() else "",
                "warped_path": str(warped_path.relative_to(ROOT)) if warped_path.exists() else "",
                "image_path": str(image_path.relative_to(ROOT)) if image_path.exists() else "",
            }
        )
    return rows


def write_review_markdown(summary_path: str | Path = EVAL_DIR / "summary.json", out_path: str | Path = EVAL_DIR / "geometry_review.md") -> str:
    rows = build_review_rows(summary_path)
    rows_sorted = sorted(rows, key=lambda row: row["score"], reverse=True)

    lines = [
        "# Geometry review summary",
        "",
        "## Status overview",
        "",
        "- Total photos evaluated: " + str(len(rows_sorted)),
        "- Original_or_deskew: " + str(sum(1 for r in rows_sorted if r["method"] == "original_or_deskew")),
        "- Deskew: " + str(sum(1 for r in rows_sorted if r["method"] == "deskew")),
        "- Perspective: " + str(sum(1 for r in rows_sorted if r["method"] == "perspective")),
        "- Needs review: " + str(sum(1 for r in rows_sorted if r["method"] == "needs_review")),
        "",
        "## Per-photo checklist",
        "",
        "| Photo | Method | Score | Debug overlay | Debug warped |",
        "| --- | --- | ---: | --- | --- |",
    ]

    for row in rows_sorted:
        overlay = f"[overlay]({row['overlay_path']})" if row["overlay_path"] else "-"
        warped = f"[warped]({row['warped_path']})" if row["warped_path"] else "-"
        lines.append(f"| {row['file']} | {row['method']} | {row['score']:.3f} | {overlay} | {warped} |")

    lines += [
        "",
        "## Remaining global plan",
        "",
        "- [ ] OCR before/after comparison for each selected method",
        "- [ ] Choose method per photo using OCR evidence, not only geometry score",
        "- [ ] Add manual review fallback for low-confidence cases",
        "- [ ] Add local dewarp / curved-paper branch only when the geometry shows residual curvature",
        "- [ ] Integrate the final decision into the main OCR pipeline",
        "- [ ] Validate on new real photos after tuning thresholds",
    ]

    content = "\n".join(lines) + "\n"
    out_path = Path(out_path)
    out_path.write_text(content, encoding="utf-8")
    return content


if __name__ == "__main__":
    write_review_markdown()
    print(f"Wrote review summary to {EVAL_DIR / 'geometry_review.md'}")
