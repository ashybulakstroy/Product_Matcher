from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

from rapidocr_onnxruntime import RapidOCR

try:
    import openpyxl
except ImportError:  # pragma: no cover - optional dependency for reference matching
    openpyxl = None

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "in_data"
REFERENCE_DIR = ROOT / "data" / "reference" / "torgsoft_ref"
EVAL_DIR = ROOT / "eval_outputs"
SUMMARY_PATH = EVAL_DIR / "summary.json"


def _coerce_text(entry: object) -> str:
    if isinstance(entry, dict):
        text = entry.get("text") or entry.get("txt") or ""
        return str(text).strip()
    if isinstance(entry, (list, tuple)):
        if len(entry) >= 2:
            return str(entry[1]).strip()
        if len(entry) == 1:
            return str(entry[0]).strip()
    return str(entry).strip()


def ocr_text(image_path: str | Path) -> str:
    image_path = Path(image_path)
    if not image_path.exists():
        return ""
    ocr = RapidOCR()
    result, _ = ocr(str(image_path))
    items = result or []
    texts: list[str] = []
    for item in items:
        text = _coerce_text(item)
        if text:
            texts.append(text)
    return " ".join(texts)


def text_metrics(text: str) -> dict:
    compact = " ".join(text.split())
    letters = re.findall(r"[A-Za-zА-Яа-яЁё]+", compact)
    digits = re.findall(r"\d+[.,\d]*%?", compact)
    price_like = re.findall(r"\d+[.,]\d{2}", compact)
    tokens = re.findall(r"[A-Za-zА-Яа-яЁё0-9.,:%/+-]+", compact)
    return {
        "char_count": len(compact),
        "word_count": len(letters),
        "digit_count": len(digits),
        "price_like_count": len(price_like),
        "token_count": len(tokens),
        "preview": compact[:200],
    }


def normalize_catalog_text(value: str | None) -> str:
    if value is None:
        return ""
    text = str(value).lower()
    text = re.sub(r"[\u2010-\u2015\u2212\-]", " ", text)
    text = re.sub(r"[^a-zа-я0-9\s.,/+%:()-]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_reference_catalog(reference_dir: str | Path = REFERENCE_DIR) -> list[str]:
    reference_path = Path(reference_dir)
    if not reference_path.exists() or openpyxl is None:
        return []

    names: list[str] = []
    for workbook_file in sorted(reference_path.glob("*.xlsx")):
        workbook = openpyxl.load_workbook(workbook_file, read_only=True, data_only=True)
        for sheet in workbook.worksheets:
            for row in sheet.iter_rows(values_only=True):
                if not row:
                    continue
                first_cell = row[0]
                if first_cell is None:
                    continue
                text = normalize_catalog_text(str(first_cell))
                if not text or text in {"название", "title"}:
                    continue
                names.append(text)
    deduped: list[str] = []
    seen: set[str] = set()
    for item in names:
        if item and item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def best_reference_match(ocr_text: str, catalog: list[str]) -> dict:
    normalized_ocr = normalize_catalog_text(ocr_text)
    if not normalized_ocr or not catalog:
        return {
            "best": "",
            "score": 0.0,
            "top_matches": [],
        }

    scored: list[tuple[float, str]] = []
    for item in catalog:
        if not item:
            continue
        score = SequenceMatcher(None, normalized_ocr, item).ratio()
        if normalized_ocr in item or item in normalized_ocr:
            score = max(score, 0.9)
        scored.append((score, item))

    scored.sort(key=lambda t: t[0], reverse=True)
    top_matches = [{"name": name, "score": round(score, 4)} for score, name in scored[:5]]
    best_name = scored[0][1] if scored else ""
    best_score = scored[0][0] if scored else 0.0
    return {
        "best": best_name,
        "score": float(best_score),
        "top_matches": top_matches,
    }


def compare_ocr_for_summary(summary_path: str | Path = SUMMARY_PATH) -> list[dict]:
    summary_path = Path(summary_path)
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    catalog = load_reference_catalog()
    rows: list[dict] = []

    for item in summary:
        file_name = item["file"]
        stem = Path(file_name).stem
        image_path = DATA_DIR / file_name
        warped_path = EVAL_DIR / stem / "warped.png"

        orig_text = ocr_text(image_path)
        warped_text = ocr_text(warped_path) if warped_path.exists() else ""

        orig_metrics = text_metrics(orig_text)
        warped_metrics = text_metrics(warped_text)

        orig_match = best_reference_match(orig_text, catalog)
        warped_match = best_reference_match(warped_text, catalog)

        winner = "original"
        if warped_text and not orig_text:
            winner = "warped"
        elif warped_text and orig_text:
            if warped_match["score"] > orig_match["score"]:
                winner = "warped"
            elif warped_match["score"] == orig_match["score"] and warped_metrics["token_count"] > orig_metrics["token_count"]:
                winner = "warped"

        rows.append(
            {
                "file": file_name,
                "method": item.get("method", "unknown"),
                "score": item.get("metrics", {}).get("rectangularity_score", 0.0),
                "original_text_length": orig_metrics["char_count"],
                "warped_text_length": warped_metrics["char_count"],
                "original_tokens": orig_metrics["token_count"],
                "warped_tokens": warped_metrics["token_count"],
                "original_reference_match": orig_match["score"],
                "warped_reference_match": warped_match["score"],
                "original_best_reference": orig_match["best"],
                "warped_best_reference": warped_match["best"],
                "winner": winner,
                "original_preview": orig_metrics["preview"],
                "warped_preview": warped_metrics["preview"],
            }
        )

    output = EVAL_DIR / "ocr_comparison.json"
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def write_ocr_summary(report_path: str | Path = EVAL_DIR / "ocr_comparison.json") -> str:
    rows = compare_ocr_for_summary()
    lines = [
        "# OCR comparison summary",
        "",
        "| Photo | Method | Geometry score | Original tokens | Warped tokens | Winner |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in sorted(rows, key=lambda r: r["score"], reverse=True):
        lines.append(
            f"| {row['file']} | {row['method']} | {row['score']:.3f} | {row['original_tokens']} | {row['warped_tokens']} | {row['winner']} |"
        )
    content = "\n".join(lines) + "\n"
    Path(report_path).write_text(content, encoding="utf-8")
    return content


if __name__ == "__main__":
    rows = compare_ocr_for_summary()
    print(json.dumps(rows, ensure_ascii=False, indent=2)[:4000])
