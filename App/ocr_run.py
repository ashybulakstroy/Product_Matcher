"""Local OCR of rectified tables and equivalent original-image crops."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/cyrillic_PP-OCRv5_rec_mobile.onnx"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    entries = json.loads((ROOT / "eval_outputs/rectification_summary.json").read_text(encoding="utf-8"))
    profile = dict(engine="rapidocr_onnxruntime", version=importlib.metadata.version("rapidocr_onnxruntime"),
                   recognition_model=MODEL.name, model_sha256=sha(MODEL.read_bytes()), text_score=.3,
                   rec_img_shape=[3, 48, 320], use_angle_cls=True)
    started = perf_counter()
    engine = RapidOCR(rec_model_path=str(MODEL), rec_img_shape=profile["rec_img_shape"], text_score=profile["text_score"])
    profile["initialization_ms"] = round((perf_counter()-started)*1000)
    records = []
    for index, entry in enumerate(entries):
        source = ROOT / "data/in_data" / entry["file"]
        directory = ROOT / "eval_outputs" / source.stem
        original = cv2.imread(str(source))
        polygon = np.asarray(entry["source_polygon"])
        x0, y0 = np.maximum(0, np.floor(polygon.min(axis=0)).astype(int)-16)
        x1, y1 = np.minimum([original.shape[1], original.shape[0]], np.ceil(polygon.max(axis=0)).astype(int)+17)
        crop = original[y0:y1, x0:x1]
        crop_path = directory / "original_table_crop.png"
        encoded = cv2.imencode(".png", crop)[1].tobytes()
        if not crop_path.exists() or crop_path.read_bytes() != encoded:
            crop_path.write_bytes(encoded)
        for variant, image_path in (("original_crop", crop_path), ("rectified", directory / "rectified.png")):
            signature = sha(image_path.read_bytes())
            output_path = directory / f"ocr_{variant}.json"
            cached = json.loads(output_path.read_text(encoding="utf-8")) if output_path.exists() else None
            if cached and cached.get("image_sha256") == signature and cached.get("profile", {}).get("model_sha256") == profile["model_sha256"] and cached.get("profile", {}).get("text_score") == profile["text_score"]:
                records.append(dict(file=source.name, variant=variant, lines=len(cached["lines"]), cached=True))
                continue
            begin = perf_counter()
            result, phase_times = engine(str(image_path))
            elapsed_ms = round((perf_counter()-begin)*1000)
            lines = []
            maps = np.load(directory / "rectification_map.npz") if variant == "rectified" else None
            for box, text, confidence in result or []:
                box = np.asarray(box, np.float32)
                if maps is not None:
                    mx = np.clip(box[:, 0], 0, maps["map_x"].shape[1]-1).reshape(1, -1)
                    my = np.clip(box[:, 1], 0, maps["map_x"].shape[0]-1).reshape(1, -1)
                    original_box = np.column_stack([cv2.remap(maps[key], mx, my, cv2.INTER_LINEAR).reshape(-1)
                                                    for key in ("map_x", "map_y")])
                else:
                    original_box = box + [x0, y0]
                lines.append(dict(text=str(text), confidence=float(confidence), box=box.tolist(),
                                  box_original=original_box.tolist()))
            if maps is not None:
                maps.close()
            payload = dict(file=source.name, variant=variant, image_sha256=signature, profile=profile,
                           elapsed_ms=elapsed_ms, phase_seconds=[float(t) for t in phase_times or []],
                           crop_origin=[int(x0), int(y0)] if variant == "original_crop" else None,
                           lines=lines)
            output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            (directory / f"ocr_{variant}.txt").write_text("\n".join(line["text"] for line in lines)+"\n", encoding="utf-8")
            records.append(dict(file=source.name, variant=variant, lines=len(lines), elapsed_ms=elapsed_ms))
            print(f"{index+1}/{len(entries)} {variant}: {len(lines)} lines, {elapsed_ms} ms | {source.name}", flush=True)
    (ROOT / "eval_outputs/ocr_run.json").write_text(json.dumps(dict(profile=profile, results=records), indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
