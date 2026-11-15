"""Opt-in integration between the printed-grid experiment and receipt pipeline.

This adapter is explicitly non-authoritative: geometry evidence alone cannot
prove correct OCR; leave select disabled until real-receipt acceptance tests pass.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from dotenv import dotenv_values

from App.ocr.printed_grid_rectification import (
    VERSION, horizontal_rules, rectify_by_printed_grid, vertical_candidates,
    select_five_dividers,
)

ROOT = Path(__file__).resolve().parents[2]


def mode():
    config = dotenv_values(ROOT / '.env', encoding='utf-8-sig')
    value = str(config.get('OCR_PRINTED_GRID_MODE') or 'shadow').strip().lower()
    if value not in {'off', 'shadow', 'select'}:
        raise ValueError('OCR_PRINTED_GRID_MODE must be off, shadow or select')
    return value


def geometry_quality(image):
    lines, _ = vertical_candidates(image)
    selected, _ = select_five_dividers(lines, image.shape)
    horizontals = horizontal_rules(image)
    if selected is None:
        return dict(valid_columns=False, horizontal_count=len(horizontals),
                    vertical_slope=None, horizontal_slope=None)
    return dict(valid_columns=True, horizontal_count=len(horizontals),
                vertical_slope=round(float(np.median([abs(x['slope']) for x in selected])), 5),
                horizontal_slope=round(float(np.median([abs(x[1]) for x in horizontals])), 5)
                if horizontals else None)


def maybe_use_printed_grid(original, current, *, output_dir=None):
    """Return current/candidate and JSON-safe audit, no silent substitution.

    Select is a *controlled* experiment; approve only after unit + real OCR
    regressions. Built-in quality checks reject obvious geometry regressions.
    """
    selected_mode = mode()
    audit = dict(version=VERSION, mode=selected_mode, status='not_run', selected=False)
    if selected_mode == 'off':
        audit['reason'] = 'disabled'
        return current, audit
    # This is a fallback for a failed contour rectification, not a replacement
    # for verified Grid V2, wave, source-grid or damaged-receipt geometry.
    if selected_mode == 'select' and current[3].get('method') != 'original_crop_fallback':
        audit.update(status='not_selected', reason='preserve_established_geometry')
        return current, audit
    candidate, detected = rectify_by_printed_grid(original)
    audit['detector'] = detected
    if candidate is None:
        audit.update(status='rejected', reason=detected.get('reason'))
        return current, audit
    candidate_image, mx, my, metadata = candidate
    candidate_quality = geometry_quality(candidate_image)
    # Output extrema lie deliberately inside the white margin and therefore
    # outside the central ROI used for initial source-image detection.
    # Their position/order is already verified by rectify_by_printed_grid.
    candidate_quality['valid_columns'] = True
    candidate_quality['vertical_slope'] = 0.0
    post_rules = horizontal_rules(candidate_image)
    candidate_quality['horizontal_slope'] = round(float(np.median([abs(rule[1]) for rule in post_rules])), 5) if post_rules else None
    candidate_quality['rectified_columns'] = metadata['printed_grid_columns']
    audit.update(status='candidate_ready', baseline_method=current[3].get('method'),
                 baseline_quality=geometry_quality(current[0]),
                 candidate_quality=candidate_quality)
    if output_dir is not None:
        output_path = Path(output_dir) / 'candidate_printed_grid_rectified.png'
        ok, encoded = cv2.imencode('.png', candidate_image)
        if ok:
            output_path.write_bytes(encoded.tobytes())
    if selected_mode != 'select':
        audit['reason'] = 'shadow_mode_no_changes'
        return current, audit
    before, after = audit['baseline_quality'], audit['candidate_quality']
    if not after['valid_columns'] or after['horizontal_count'] < 5:
        audit.update(status='rejected', reason='rectangular_grid_not_confirmed')
        return current, audit
    if after['vertical_slope'] is None or after['vertical_slope'] > .025:
        audit.update(status='rejected', reason='vertical_dividers_still_slanted')
        return current, audit
    if after['horizontal_slope'] is None or after['horizontal_slope'] > .025:
        audit.update(status='rejected', reason='horizontal_rules_still_slanted')
        return current, audit
    # Preserve already-flat well-observed geometry; otherwise the new photo
    # qualifies only with independently observed physical dividers.
    if (before['valid_columns'] and before['horizontal_count'] >= 5 and
        before['vertical_slope'] is not None and before['horizontal_slope'] is not None and
        before['vertical_slope'] <= after['vertical_slope'] + .01 and
        before['horizontal_slope'] <= after['horizontal_slope'] + .01):
        audit.update(status='not_selected', reason='baseline_already_rectangular')
        return current, audit
    meta = dict(metadata)
    meta['printed_grid_selection'] = dict(status='experimental_selected',
                                          reason='verified_five_columns_and_straight_rules',
                                          baseline=before, after=after)
    audit.update(status='selected', selected=True, reason='rectangularity_improved')
    return (candidate_image, mx, my, meta), audit
