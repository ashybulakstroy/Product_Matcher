from pathlib import Path

import cv2
import numpy as np
import pytest

from App.ocr.ocr_geometry import detect_table_corners, detect_table_polygon, rectify_receipt
from App.ocr.paths import PROJECT_ROOT
from App.ocr.table_boundary import detect_table_boundary

ROOT = PROJECT_ROOT


def _distance(polygon, point):
    return abs(cv2.pointPolygonTest(polygon.astype(np.float32), point, True))


@pytest.mark.parametrize("filename,landmarks", [
    ("WhatsApp Image 2026-10-23 at 09.34.47.jpeg",
     [(115, 240), (385, 264), (560, 240), (583, 733), (394, 733), (394, 708), (102, 705)]),
    ("WhatsApp Image 2026-10-25 at 17.45.17.jpeg",
     [(155, 354), (654, 352), (644, 932), (451, 931), (450, 901), (154, 900)]),
])
def test_real_boundary_follows_bend_and_total_step(filename, landmarks):
    # Landmarks read from the original photographs, independent of the detector.
    image_path = ROOT / "data/in_data" / filename
    if not image_path.exists():
        pytest.skip("source photo is not present in this workspace")
    image = cv2.imread(str(image_path))
    assert image is not None
    result = detect_table_boundary(image)
    polygon = result["polygon"]
    assert polygon is not None and len(polygon) > 4
    assert max(_distance(polygon, point) for point in landmarks) < 10
    assert polygon[:, 1].max() < (750 if "09.34.47" in filename else 945)
    encoded = cv2.imencode(".png", image)[1].tobytes()
    rectified = rectify_receipt(encoded)
    assert rectified["status"] == "needs_review"
    assert rectified["method"] == "none"
    assert rectified["source_corners"] is None
    assert not rectified["quality_metrics"]["detected_four_corners"]
    assert np.array_equal(rectified["image"], image)


def test_grid_beats_larger_page_border_and_barcode():
    image = np.full((800, 650, 3), 60, np.uint8)
    cv2.rectangle(image, (25, 25), (620, 760), (255, 255, 255), -1)
    expected = np.array([[80, 120], [200, 130], [310, 150], [420, 130], [540, 120],
                         [550, 500], [370, 500], [370, 470], [90, 470]], np.int32)
    cv2.polylines(image, [expected], True, (30, 30, 30), 2)
    for x, top in [(120, 124), (370, 140), (460, 126)]:
        cv2.line(image, (x, top), (x, 470), (30, 30, 30), 2)
    for y in [190, 280, 380]:
        cv2.line(image, (85, y), (542, y), (30, 30, 30), 2)
    for x in range(90, 350, 5):
        cv2.line(image, (x, 560), (x, 630), (0, 0, 0), 2)
    polygon = detect_table_boundary(image)["polygon"]
    assert polygon is not None
    expected_mask = np.zeros(image.shape[:2], np.uint8)
    actual_mask = np.zeros_like(expected_mask)
    cv2.fillPoly(expected_mask, [expected], 1)
    cv2.fillPoly(actual_mask, [polygon.astype(np.int32)], 1)
    assert np.count_nonzero(expected_mask & actual_mask) / np.count_nonzero(expected_mask | actual_mask) > .96
    assert _distance(polygon, (310, 150)) < 6
    assert _distance(polygon, (370, 500)) < 6


def test_no_table_is_not_reported_as_successful_frame_warp():
    image = np.full((320, 400, 3), 255, np.uint8)
    assert detect_table_boundary(image)["polygon"] is None
    # Legacy callers still get an in-bounds frame, never used as evidence in rectify.
    assert detect_table_polygon(image).max(axis=0).tolist() == [399, 319]
    result = rectify_receipt(cv2.imencode(".png", image)[1].tobytes())
    assert result["status"] == "needs_review"
    assert result["source_polygon"] is None
    assert result["method"] == "none"
    assert np.array_equal(result["image"], image)


def test_legacy_detection_paths_agree():
    image_path = ROOT / "data/in_data/WhatsApp Image 2026-10-23 at 09.34.47.jpeg"
    if not image_path.exists():
        pytest.skip("source photo is not present in this workspace")
    image = cv2.imread(str(image_path))
    assert image is not None
    assert np.array_equal(detect_table_polygon(image), detect_table_corners(image))


def test_manual_quad_is_still_available():
    image = np.full((320, 400, 3), 255, np.uint8)
    result = rectify_receipt(cv2.imencode(".png", image)[1].tobytes(), mode="manual",
                             manual_corners=[(50, 50), (350, 50), (350, 250), (50, 250)])
    assert result["status"] == "manual_rectified"
    assert result["image"].shape[:2] == (200, 300)


def test_gap_boundary_is_flagged_on_real_photo():
    image_path = ROOT / "data/in_data/WhatsApp Image 2026-10-23 at 13.57.55.jpeg"
    if not image_path.exists():
        pytest.skip("source photo is not present in this workspace")
    image = cv2.imread(str(image_path))
    assert image is not None
    result = detect_table_boundary(image)
    assert result["polygon"] is not None
    assert result["status"] == "needs_review"
    assert "boundary_has_gaps_or_attached_text" in result["warnings"]


def test_intact_cell_does_not_replace_long_broken_table():
    image_path = ROOT / "data/in_data/WhatsApp Image 2026-10-22 at 16.15.19.jpeg"
    if not image_path.exists():
        pytest.skip("source photo is not present in this workspace")
    image = cv2.imread(str(image_path))
    assert image is not None
    result = detect_table_boundary(image)
    polygon = result["polygon"]
    assert polygon is not None
    assert np.ptp(polygon[:, 1]) > image.shape[0] * .6
    assert result["status"] == "needs_review"
    assert "partial_and_full_table_candidates" in result["warnings"]
