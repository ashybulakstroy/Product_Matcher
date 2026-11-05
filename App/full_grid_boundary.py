"""Conservative fallback boundary reconstructed from long printed grid rules."""
from __future__ import annotations

import numpy as np

from App.wave_dewarp import find_rules


def _longest_run(rules, image_shape):
    height, width = image_shape[:2]
    entries = []
    for rule in rules:
        xs = rule.samples[:, 0]
        entries.append(dict(y=float(np.median(rule.values)), left=float(np.min(xs)),
                            right=float(np.max(xs)), coverage=float(rule.coverage), rule=rule))
    # Full-width page edges are not table rules.
    entries = [item for item in entries if item["coverage"] < .92 and item["right"] - item["left"] >= width * .50]
    entries.sort(key=lambda item: item["y"])
    runs, current = [], []
    for item in entries:
        if current and item["y"] - current[-1]["y"] > max(150., height * .16):
            runs.append(current)
            current = []
        current.append(item)
    if current:
        runs.append(current)
    return max(runs, key=lambda run: (len(run), run[-1]["y"] - run[0]["y"]), default=[])


def _robust_line(y, values):
    """Fit a perspective edge while discarding occlusion endpoint outliers."""
    keep = np.ones(len(y), dtype=bool)
    for _ in range(4):
        fitted = np.polyfit(y[keep], values[keep], 1)
        residuals = np.abs(values - np.polyval(fitted, y))
        median = float(np.median(residuals[keep]))
        mad = float(np.median(np.abs(residuals[keep] - median)))
        limit = max(8., median + 3. * max(1., 1.4826 * mad))
        updated = residuals <= limit
        if updated.sum() < 6 or np.array_equal(updated, keep):
            break
        keep = updated
    return np.polyfit(y[keep], values[keep], 1), keep


def detect_full_grid_boundary(image):
    """Return a four-corner full-grid hypothesis, otherwise ``None``.

    Text cannot pass this check: at least six long parallel rules must cover a
    large vertical span and their two ends must independently fit perspective
    edges.  It is deliberately a fallback, never a replacement for a sound
    contour boundary.
    """
    rules, _ = find_rules(image)
    run = _longest_run(rules, image.shape)
    height, width = image.shape[:2]
    if len(run) < 6 or not run or run[-1]["y"] - run[0]["y"] < height * .35:
        return None, {"reason": "not_enough_full_width_grid_rules", "rule_count": len(run)}
    y = np.asarray([item["y"] for item in run])
    left = np.asarray([item["left"] for item in run])
    right = np.asarray([item["right"] for item in run])
    left_fit, left_keep = _robust_line(y, left)
    right_fit, right_keep = _robust_line(y, right)
    residual = max(float(np.max(np.abs(left[left_keep] - np.polyval(left_fit, y[left_keep])))),
                   float(np.max(np.abs(right[right_keep] - np.polyval(right_fit, y[right_keep])))))
    if residual > max(22., width * .055):
        return None, {"reason": "grid_rule_endpoints_not_perspective_aligned", "residual_px": residual,
                      "rule_count": len(run)}
    top, bottom = float(y[0]), float(y[-1])
    tl_x, tr_x = np.polyval(left_fit, top), np.polyval(right_fit, top)
    bl_x, br_x = np.polyval(left_fit, bottom), np.polyval(right_fit, bottom)
    # Keep the observed slope of the first and last printed grid rules.  Their
    # medians locate a row, but using the median as both corner heights turns
    # this into a rectangle and silently discards the perspective correction.
    def rule_y(item, x):
        values = item["rule"].values
        return float(values[int(np.clip(round(x), 0, len(values) - 1))])
    polygon = np.array([[tl_x, rule_y(run[0], tl_x)], [tr_x, rule_y(run[0], tr_x)],
                        [br_x, rule_y(run[-1], br_x)], [bl_x, rule_y(run[-1], bl_x)]], np.float32)
    return polygon, {"reason": "full_grid_hypothesis", "rule_count": len(run),
                     "height_ratio": (bottom - top) / height, "residual_px": residual,
                     "kept_rules": int((left_keep & right_keep).sum())}
