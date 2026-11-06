"""One typed SSE contract for OCR UI. No OCR engines or UI dependencies.

The event is informational: it must never hold up OCR or change recognition results.
"""
from __future__ import annotations

from typing import Any

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
