"""Conservative lazy header OCR for verified printed Torgsoft grid.

No heuristics about expected totals or number of rows; no OCR text is fabricated.
Default: OFF. All decisions are per-call, using exact .env or environment setting.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERSION = 'ocr-lazy-header-phases-v1'


def early_opt_mode() -> str:
    raw = os.environ.get('OCR_EARLY_OPT_MODE')
    if raw is None:
        try:
            from dotenv import dotenv_values
            raw = dotenv_values(ROOT / '.env', encoding='utf-8-sig').get('OCR_EARLY_OPT_MODE')
        except ImportError:
            raw = None
    value = str(raw or 'off').strip().lower()
    if value not in {'off', 'select'}:
        raise ValueError('OCR_EARLY_OPT_MODE: используйте off или select')
    return value


def early_opt_enabled() -> bool:
    return early_opt_mode() == 'select'


def can_skip_header_ocr(*, four_columns, header_row, grid_support,
                        dense_bars, recognize_header) -> bool:
    """Skip ONLY when original validator already accepts geometry alone.

    `four_columns AND header_row AND grid_support` is the original validator's
    existing acceptance condition. Never broaden that rule by guessing.
    """
    # The independent geometry fast path requires OCR-confirmed header text.
    # Preserve that evidence when it is enabled rather than silently disabling it.
    from App.ocr.ocr_fast_path import fast_mode
    return bool(
        early_opt_enabled()
        and fast_mode(ROOT) != 'select'
        and recognize_header is not None
        and four_columns and header_row and grid_support
        and not dense_bars
    )
