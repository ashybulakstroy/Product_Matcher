"""Read-only discovery of current OCR output; no V1/V2/old Trial dependency."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import cv2
import numpy as np

ALLOWED_IMAGE_NAMES = ("original.png", "upload.bin", "original.jpg", "original.jpeg")


@dataclass(frozen=True)
class Case:
    job_id: str
    directory: Path
    image_path: Path
    sha256: str


def within(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def load_image(case: Case):
    # Decode bytes, not filename extension: upload.bin contains JPEG/PNG bytes.
    arr = np.frombuffer(case.image_path.read_bytes(), dtype=np.uint8)
    result = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if result is None or result.size == 0:
        raise ValueError("unreadable_image")
    return result


def discover(root: Path, limit=0):
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"No input directory: {root}")
    rows = []
    seen = set()
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or directory.is_symlink() or not within(directory, root):
            continue
        candidates = [directory / name for name in ALLOWED_IMAGE_NAMES]
        image = next((p for p in candidates if p.is_file() and not p.is_symlink() and within(p,root)), None)
        if image is None:
            continue
        # Prefer upload bytes for identity even if the oriented original was regenerated.
        original = directory / "upload.bin"
        identity = original if original.is_file() and not original.is_symlink() else image
        sha = hashlib.sha256(identity.read_bytes()).hexdigest()
        if sha in seen:
            continue
        seen.add(sha)
        rows.append(Case(directory.name, directory, image, sha))
        if limit and len(rows) >= limit:
            break
    return rows


def load_baseline(case: Case):
    """Load *main project's* artifacts if present, never previous Trial files."""
    try:
        pic = case.directory / "rectified.png"
        maps = case.directory / "rectification_map.npz"
        layout_file = case.directory / "layout.json"
        if not all(p.is_file() and not p.is_symlink() for p in (pic,maps,layout_file)):
            return None, "main_ocr_artifacts_missing"
        img = cv2.imdecode(np.frombuffer(pic.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None, "invalid_rectified_png"
        with np.load(maps, allow_pickle=False) as data:
            mx, my = data["map_x"], data["map_y"]
        d = json.loads(layout_file.read_text(encoding="utf-8"))
        layout = d.get("layout", d)
        xs = [float(layout.get("body_left", layout.get("name_left", 0) - 20)),
              float(layout["name_left"]),float(layout["name_right"]),
              float(layout["quantity_right"]),float(layout["right"])]
        ys = [float(x) for x in layout["row_boundaries"]]
        if mx.shape != img.shape[:2] or my.shape != mx.shape:
            return None, "baseline_map_shape_mismatch"
        from .algorithms import Result
        return Result("baseline", img, mx.astype(np.float32), my.astype(np.float32),
                      xs,ys,{"source":"main_project_web_outputs","grid_evidence":"main_OCR_layout"}), None
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as error:
        return None, f"invalid_main_ocr_artifacts:{type(error).__name__}:{error}"
