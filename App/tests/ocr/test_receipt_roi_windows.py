import cv2
import numpy as np

from App.ocr.receipt_roi_recovery import recover_table_boundary
from App.ocr.table_boundary import detect_table_boundary


def small_background():
    return dict(polygon=np.array([[700, 400], [780, 400], [780, 480], [700, 480]], np.float32),
                status='detected', warnings=[], selected={'support': [3, 6]})


def test_small_background_contour_is_replaced_by_observed_full_table():
    image = np.full((1600, 900, 3), 150, np.uint8)
    # The paper shares the background brightness: a brightness blob cannot help.
    for x in (340, 370, 530, 580, 650):
        cv2.line(image, (x, 140), (x, 1460), (30, 30, 30), 2)
    for y in range(140, 1461, 40):
        cv2.line(image, (340, y), (650, y), (30, 30, 30), 2)
    boundary, report = recover_table_boundary(image, detect_table_boundary, small_background())
    assert report['status'] == 'recovered'
    assert report['ocr_calls'] == 0
    assert np.ptp(boundary['polygon'][:, 1]) > 1200
    assert boundary['polygon'][:, 0].min() >= 330
    assert boundary['polygon'][:, 0].max() <= 660


def test_valid_baseline_does_not_run_extra_detection():
    baseline = dict(polygon=np.array([[40, 100], [600, 100], [600, 1300], [40, 1300]]))
    def unexpected(_):
        raise AssertionError('Valid baseline must not be rescanned')
    boundary, report = recover_table_boundary(np.zeros((1600, 900, 3), np.uint8), unexpected, baseline)
    assert boundary is baseline
    assert report['status'] == 'not_needed'


def test_background_without_grid_cannot_create_receipt():
    baseline = small_background()
    boundary, report = recover_table_boundary(np.full((1600, 900, 3), 150, np.uint8), detect_table_boundary, baseline)
    assert boundary is baseline
    assert report['status'] == 'not_recovered'


def test_clipped_table_is_rejected():
    baseline = small_background()
    def clipped(sample):
        h, w = sample.shape[:2]
        return dict(polygon=np.array([[0, 100], [w-1, 100], [w-1, h-100], [0, h-100]], np.float32),
                    selected={'support': [20, 4]})
    boundary, report = recover_table_boundary(np.zeros((1600, 900, 3), np.uint8), clipped, baseline)
    assert boundary is baseline
    assert report['status'] == 'not_recovered'
