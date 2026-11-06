import cv2
import numpy as np

from App.ocr.full_grid_boundary import detect_full_grid_boundary
from App.ocr.rotation_rectification import rotate_table_crop


def test_long_perspective_grid_produces_full_table_hypothesis():
    image = np.full((1000, 700, 3), 255, np.uint8)
    for fraction in np.linspace(0, 1, 11):
        y = int(120 + fraction * 700)
        left, right = int(70 + fraction * 50), int(570 + fraction * 50)
        cv2.line(image, (left, y), (right, y), (30, 30, 30), 2)
    for top, bottom in ((70, 120), (300, 350), (440, 490), (570, 620)):
        cv2.line(image, (top, 120), (bottom, 820), (30, 30, 30), 2)

    polygon, details = detect_full_grid_boundary(image)

    assert details["reason"] == "full_grid_hypothesis"
    assert polygon is not None
    assert np.ptp(polygon[:, 0]) > 490
    assert np.ptp(polygon[:, 1]) > 690


def test_plain_image_cannot_become_grid_hypothesis():
    polygon, details = detect_full_grid_boundary(np.full((500, 500, 3), 255, np.uint8))
    assert polygon is None
    assert details["reason"] == "not_enough_full_width_grid_rules"


def test_rotation_crop_turns_tilted_table_clockwise_without_non_linear_warp():
    image = np.full((700, 600, 3), 255, np.uint8)
    polygon = np.array([[80, 120], [500, 83], [540, 580], [120, 617]], np.float32)
    output, map_x, map_y, details = rotate_table_crop(image, polygon)

    assert details["method"] == "rotation_crop"
    assert 4 < details["rotation_clockwise_deg"] < 6
    rotated = np.asarray(details["rotated_polygon"])
    assert abs(rotated[1, 1] - rotated[0, 1]) < 1
    assert abs(rotated[2, 1] - rotated[3, 1]) < 1
    assert output.shape[:2] == map_x.shape == map_y.shape
