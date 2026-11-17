"""Safe unit checks for the opt-in perspective + 2D grid experiment."""
from pathlib import Path

import numpy as np
import pytest

from App.ocr import perspective_2d_grid_dewarp as d


def _rule(level, profile, start, end, p95=0.5):
    return d.Rule(float(level), np.asarray(profile, np.float32), int(start), int(end),
                  (end - start) / len(profile), float(p95), int(end - start))


def test_module_lives_in_app_and_has_no_experiment_package_dependency():
    path = Path(d.__file__).resolve()
    root = Path(__file__).resolve().parents[3]
    assert path == (root / 'App/ocr/perspective_2d_grid_dewarp.py').resolve()
    assert d.VERSION == 'perspective-2d-grid-dewarp-v1.0'


def test_setting_mode_defaults_to_shadow_and_rejects_unknown_values(tmp_path):
    assert d.setting_mode(tmp_path) == 'shadow'
    (tmp_path / '.env').write_text('OCR_PERSPECTIVE_2D_MODE=select\n', encoding='utf-8')
    assert d.setting_mode(tmp_path) == 'select'
    (tmp_path / '.env').write_text('OCR_PERSPECTIVE_2D_MODE=INVALID\n', encoding='utf-8')
    with pytest.raises(ValueError):
        d.setting_mode(tmp_path)


@pytest.mark.parametrize('method', [
    'grid_curvature_v2', 'wave_grid', 'printed_grid_rectified',
    'source_grid_crop', 'contour_coons_patch', 'experimental_horizontal_only',
])
def test_existing_2d_geometry_is_never_replaced(method):
    image = np.full((900, 600, 3), 250, np.uint8)
    current = (image, np.zeros(image.shape[:2], np.float32),
               np.zeros(image.shape[:2], np.float32), {'method': method})
    selected, report = d.maybe_refine_perspective_grid(
        image, np.float32([[20, 20], [580, 20], [580, 880], [20, 880]]),
        current, mode_override='select')
    assert selected is current
    assert report['status'] == 'not_needed'


def test_two_fragments_of_one_curved_rule_are_merged():
    width = 600
    x = np.arange(width, dtype=np.float32)
    upper = np.full(width, 400.0, np.float32)
    left = 500.0 + .02 * x
    right = 506.0 + .02 * (x - 300.0)
    rules = [
        _rule(400.0, upper, 20, 580),
        _rule(float(np.median(left[50:300])), left, 50, 300),
        _rule(float(np.median(right[300:550])), right, 300, 550),
    ]
    merged = d._merge_horizontal_fragments(rules, width, 900)
    assert len(merged) == 2
    curved = next(rule for rule in merged if rule.level > 450)
    assert curved.start == 50 and curved.end == 550
    assert abs(float(curved.profile[300]) - 506.0) < .1
    levels = np.asarray([rule.level for rule in merged], float)
    paths = np.asarray([rule.profile for rule in merged], float)
    assert np.min(np.diff(levels)) >= 9.0
    assert np.min(np.diff(paths, axis=0)) > 0


def test_two_long_parallel_rules_are_not_merged():
    width = 600
    upper = np.full(width, 400.0, np.float32)
    lower = np.full(width, 410.0, np.float32)
    rules = [_rule(400.0, upper, 10, 590), _rule(410.0, lower, 10, 590)]
    assert len(d._merge_horizontal_fragments(rules, width, 900)) == 2
