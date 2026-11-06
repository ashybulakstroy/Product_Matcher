import cv2
import numpy as np
import pytest

from App.ocr.batch_geometry_eval import compute_rectangularity, should_generate_warp
from App.ocr.ocr_eval import best_reference_match, load_reference_catalog
from App.ocr.ocr_geometry import clamp_corners_to_image, detect_table_corners, order_corners
from App.ocr.paths import PROJECT_ROOT


def test_detect_table_corners_on_rectangular_document():
    image = np.ones((1200, 1600, 3), dtype=np.uint8) * 255
    pts = np.array([
        [120, 180],
        [1460, 220],
        [1520, 980],
        [80, 1010],
    ], dtype=np.int32)

    cv2.line(image, tuple(pts[0]), tuple(pts[1]), (0, 0, 0), 10)
    cv2.line(image, tuple(pts[1]), tuple(pts[2]), (0, 0, 0), 10)
    cv2.line(image, tuple(pts[2]), tuple(pts[3]), (0, 0, 0), 10)
    cv2.line(image, tuple(pts[3]), tuple(pts[0]), (0, 0, 0), 10)

    corners = detect_table_corners(image)

    assert len(corners) == 4
    assert corners.shape[0] == 4
    assert np.all(corners[:, 0] >= 0)
    assert np.all(corners[:, 1] >= 0)


def test_order_corners_handles_duplicate_points():
    corners = np.array([
        [147.0, 656.0],
        [387.0, 634.0],
        [389.0, 684.0],
        [147.0, 656.0],
    ], dtype=np.float32)

    ordered = detect_table_corners(np.ones((800, 800, 3), dtype=np.uint8) * 255)
    assert ordered.shape == (4, 2)
    assert np.unique(np.round(ordered, 2), axis=0).shape[0] == 4

    ordered_from_duplicates = order_corners(corners)
    assert ordered_from_duplicates.shape == (4, 2)
    assert np.unique(np.round(ordered_from_duplicates, 2), axis=0).shape[0] == 4


def test_should_generate_warp_rejects_extreme_geometry():
    bad_corners = np.array([
        [10, 10],
        [100, 20],
        [1100, 1000],
        [20, 1200],
    ], dtype=np.float32)

    assert not should_generate_warp(bad_corners, (1200, 1200))

    good_corners = np.array([
        [80, 100],
        [1100, 150],
        [1120, 900],
        [50, 950],
    ], dtype=np.float32)

    assert should_generate_warp(good_corners, (1200, 1200))


def test_order_corners_keeps_top_right_and_bottom_right_in_correct_order():
    points = np.array([
        [140.1536, 131.93532],
        [-21.762016, 87.31538],
        [-34.21552, 9.063177],
        [236.15419, 69.31388],
    ], dtype=np.float32)

    ordered = order_corners(points)

    assert np.allclose(ordered[0], np.array([-34.21552, 9.063177], dtype=np.float32))
    assert np.allclose(ordered[1], np.array([236.15419, 69.31388], dtype=np.float32))
    assert np.allclose(ordered[2], np.array([140.1536, 131.93532], dtype=np.float32))
    assert np.allclose(ordered[3], np.array([-21.762016, 87.31538], dtype=np.float32))


def test_order_corners_keeps_top_edge_on_receipt_boundary():
    points = np.array([
        [126.34142165, -198.90459993],
        [142.96171064, -186.56576988],
        [91.86217857, -129.73775176],
        [145.27156894, 16.5844881],
        [-80.11524379, -30.92511152],
    ], dtype=np.float32)

    ordered = order_corners(points)

    assert ordered[0, 1] <= ordered[1, 1] + 1e-6
    assert ordered[0, 1] <= ordered[2, 1] + 1e-6
    assert ordered[0, 1] <= ordered[3, 1] + 1e-6
    assert ordered[0, 0] <= ordered[1, 0] + 1e-6


def test_detect_table_corners_on_real_receipt_image_keeps_complex_receipt_shape():
    image_path = PROJECT_ROOT / "data/in_data/WhatsApp Image 2026-10-23 at 09.34.47.jpeg"
    if not image_path.exists():
        pytest.skip("source photo is not present in this workspace")
    image = cv2.imread(str(image_path))
    assert image is not None

    corners = detect_table_corners(image)

    assert corners.shape[0] >= 5
    assert np.max(corners[:, 0]) > 500
    assert np.max(corners[:, 1]) > 700
    assert not should_generate_warp(corners, image.shape[:2])


def test_detect_table_corners_clips_negative_points_to_image_bounds():
    corners = np.array([
        [162.0, -88.0],
        [557.0, 65.0],
        [82.0, 1275.0],
        [-312.0, 1120.0],
    ], dtype=np.float32)

    clamped = clamp_corners_to_image(corners, (1280, 720))

    assert np.all(clamped[:, 0] >= 0)
    assert np.all(clamped[:, 1] >= 0)
    assert np.all(clamped[:, 0] <= 719)
    assert np.all(clamped[:, 1] <= 1279)


def test_detect_table_corners_falls_back_to_text_region():
    image = np.ones((500, 700, 3), dtype=np.uint8) * 255
    for y in range(80, 320, 30):
        cv2.putText(image, 'товар', (90, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        cv2.putText(image, 'артикул', (300, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

    corners = detect_table_corners(image)

    assert corners.shape == (4, 2)
    assert np.all(corners[:, 0] >= 0)
    assert np.all(corners[:, 1] >= 0)


def test_detect_table_corners_preserves_real_receipt_polygon_edges():
    image = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
    polygon = np.array([
        [115, 238],
        [101, 707],
        [584, 732],
        [560, 239],
        [455, 252],
        [461, 468],
        [392, 470],
        [388, 262],
    ], dtype=np.int32)
    cv2.polylines(image, [polygon], True, (0, 0, 0), 10)

    corners = detect_table_corners(image)

    assert corners.shape[0] >= 5
    assert corners[:, 0].min() >= 80
    assert corners[:, 1].max() > 700
    assert corners[:, 0].max() < 700


def test_detect_table_corners_keeps_bent_top_edge_as_complex_polygon():
    image = np.ones((900, 900, 3), dtype=np.uint8) * 255
    polygon = np.array([
        [90, 160],
        [220, 170],
        [330, 200],
        [430, 210],
        [530, 180],
        [680, 170],
        [700, 700],
        [80, 700],
    ], dtype=np.int32)
    cv2.polylines(image, [polygon], True, (0, 0, 0), 12)

    corners = detect_table_corners(image)

    assert corners.shape[0] >= 5
    assert len(np.unique(np.round(corners, 1), axis=0)) >= 5


def test_reference_catalog_matches_known_product_names():
    catalog = load_reference_catalog()
    assert catalog
    match = best_reference_match('"Грунтовка" 5кг', catalog)
    assert match["best"]
    assert match["score"] > 0.6
