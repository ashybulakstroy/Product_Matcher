"""One typed SSE contract for OCR UI. No OCR engines or UI dependencies.

The event is informational: it must never hold up OCR or change recognition results.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def ready_cells(rows, layout, pending_boxes=()):
    """Cells with text whose detected blocks have all finished recognition.

    This only describes existing OCR results; it never delays or reruns OCR.
    """
    columns = [('number', layout.get('body_left', 0), layout['name_left']),
               ('name', layout['name_left'], layout['name_right']),
               ('quantity', layout['name_right'], layout['quantity_right']),
               ('price', layout['quantity_right'], layout['right'])]
    pending = [np.asarray(box, dtype=float).mean(axis=0) for box in pending_boxes]
    result = []
    for row in rows:
        if row.get('kind') not in {'product', 'tax', 'service'}:
            continue
        top, bottom = row['y_range']
        for field, left, right in columns:
            value = (row.get('item_number_raw') or row.get('item_number_recovered', '')) if field == 'number' else row.get(field + '_raw', '')
            if not str(value or '').strip():
                continue
            if any(left <= x < right and top <= y < bottom for x, y in pending):
                continue
            result.append(dict(row=int(row['row']), field=field, value=str(value), kind=row['kind']))
    return result

SCHEMA_VERSION = "visual-sync-2.0"
ACTIONS = frozenset({"geometry_found", "geometry_ready", "scan_start", "scan_stop",
                    "row_ready", "rows_ready", "checking", "matching"})


def emit(job: Any, action: str, **fields: Any) -> None:
    """Publish through existing Job.publish and existing /events SSE endpoint."""
    if action not in ACTIONS:
        raise ValueError(f"Unknown visual action: {action}")
    job.publish("visual", schema=SCHEMA_VERSION, action=action, **fields)


def raw_row(row: dict) -> dict:
    """Extract raw text only. No catalog match or fabricated numeric values."""
    return {
        "row": int(row["row"]),
        "kind": str(row.get("kind", "product")),
        "name": str(row.get("name_raw") or "")[:320],
        "quantity": str(row.get("quantity_raw") or "")[:48],
        "price": str(row.get("price_raw") or "")[:48],
        "status": "raw",
        "target": "rectified",
    }
