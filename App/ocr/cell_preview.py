"""Read-only crops of original photo pixels for electronic receipt cells."""
from functools import lru_cache
import json
from pathlib import Path
import threading

import cv2
import numpy as np

_lock = threading.Lock()
FIELDS = {'number', 'name', 'quantity', 'price'}


@lru_cache(maxsize=1)
def _load(directory, signature):
    root = Path(directory)
    layout = json.loads((root / 'layout.json').read_text(encoding='utf-8'))
    image = cv2.imdecode(np.frombuffer((root / 'original.png').read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    with np.load(root / 'rectification_map.npz', allow_pickle=False) as maps:
        mx, my = maps['map_x'], maps['map_y']
    if image is None or mx.shape != my.shape or mx.ndim != 2:
        raise ValueError('Карта исходного фото недоступна')
    return layout['cells'], image, mx, my


def preview_png(directory, row, field):
    """Use the entire mapped cell, including empty/misread numeric cells.

    Cropping OCR text boxes alone would hide the very digits a user needs to
    diagnose. Return an original-photo rectangle, never the corrected OCR text.
    """
    if field not in FIELDS or not isinstance(row, int) or isinstance(row, bool) or row < 1:
        raise ValueError('Неизвестная ячейка')
    root = Path(directory)
    paths = [root / name for name in ('layout.json', 'original.png', 'rectification_map.npz')]
    signature = tuple((p.stat().st_mtime_ns, p.stat().st_size) for p in paths)
    with _lock:
        cells, original, mx, my = _load(str(root.resolve()), signature)
    cell = next((c for c in cells if c.get('row') == row and c.get('role') == field), None)
    if cell is None:
        raise ValueError('Ячейка не найдена')
    points = np.asarray(cell['box'], dtype=float)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError('Граница ячейки недоступна')
    lo = np.maximum(0, np.floor(points.min(axis=0)).astype(int))
    hi = np.minimum([mx.shape[1], mx.shape[0]], np.ceil(points.max(axis=0)).astype(int) + 1)
    if np.any(hi <= lo):
        raise ValueError('Ячейка вне фотографии')
    xs, ys = mx[lo[1]:hi[1], lo[0]:hi[0]], my[lo[1]:hi[1], lo[0]:hi[0]]
    valid = np.isfinite(xs) & np.isfinite(ys)
    if not valid.any():
        raise ValueError('Координаты ячейки недоступны')
    left = max(0, int(np.floor(xs[valid].min())) - 3)
    right = min(original.shape[1], int(np.ceil(xs[valid].max())) + 4)
    top = max(0, int(np.floor(ys[valid].min())) - 3)
    bottom = min(original.shape[0], int(np.ceil(ys[valid].max())) + 4)
    if right <= left or bottom <= top:
        raise ValueError('Ячейка вне исходного фото')
    ok, encoded = cv2.imencode('.png', original[top:bottom, left:right])
    if not ok:
        raise ValueError('Фрагмент фото недоступен')
    return encoded.tobytes()
