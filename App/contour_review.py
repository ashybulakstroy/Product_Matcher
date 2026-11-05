"""Reproducible visual review: python -m App.contour_review."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from App.table_boundary import detect_table_boundary

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/in_data")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "debug_notes/contour_fix")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows, tiles, cards = [], [], []
    for index, path in enumerate(sorted(args.input_dir.iterdir())):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Unable to decode {path}")
        started = perf_counter()
        result = detect_table_boundary(image)
        result["elapsed_ms"] = round((perf_counter() - started) * 1000)
        polygon = result.pop("polygon")
        result["polygon"] = None if polygon is None else polygon.tolist()
        result["file"] = path.name
        rows.append(result)
        overlay = image.copy()
        if polygon is not None:
            cv2.polylines(overlay, [np.rint(polygon).astype(np.int32)], True, (0, 255, 0), 2)
        overlay_name = f"after_{index:02}.png"
        cv2.imwrite(str(args.out_dir / overlay_name), overlay)
        tile = np.full((460, 300, 3), 230, np.uint8)
        factor = min(300 / image.shape[1], 425 / image.shape[0])
        small = cv2.resize(overlay, None, fx=factor, fy=factor)
        tile[:small.shape[0], :small.shape[1]] = small
        cv2.putText(tile, f"{index}: {path.stem.removeprefix('WhatsApp Image ')}", (3, 438), 0, .37, (0, 0, 0), 1)
        cv2.putText(tile, result["status"], (3, 454), 0, .37, (0, 0, 0), 1)
        tiles.append(tile)
        cards.append(f'<article><h2>{index}: {html.escape(path.name)}</h2>'
                     f'<p>{result["status"]}; {result["elapsed_ms"]} ms</p>'
                     f'<p>{html.escape(", ".join(result["warnings"]))}</p>'
                     f'<a href="{overlay_name}"><img src="{overlay_name}"></a></article>')
    if tiles:
        blank = np.full_like(tiles[0], 230)
        montage = np.vstack([np.hstack(tiles[i:i+4] + [blank] * (4-len(tiles[i:i+4])))
                             for i in range(0, len(tiles), 4)])
        cv2.imwrite(str(args.out_dir / "after.jpg"), montage)
    (args.out_dir / "results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out_dir / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Проверка контуров</title>'
        '<style>body{font:16px system-ui;margin:24px;background:#eee}'
        'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:20px}'
        'article{background:white;padding:12px}h2{font-size:16px}img{width:100%;height:auto}</style>'
        '<h1>Контуры таблиц</h1><p>Зелёная линия — найденный полигон. '
        'detected означает найденную границу, а не разрешение на warp или доказанную OCR-точность. '
        'needs_review — неоднозначность или обрывы. Нажмите фото для полного разрешения.</p>'
        '<main>' + ''.join(cards) + '</main>', encoding="utf-8")
    print(json.dumps({"photos": len(rows), "needs_review": sum(r["status"] == "needs_review" for r in rows),
                      "report": str(args.out_dir / "index.html")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
