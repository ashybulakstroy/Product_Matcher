import cv2
import numpy as np

from App.ocr.table_boundary import _dense_vertical_bars, detect_table_boundary


def test_barcode_fragment_is_not_a_table():
    binary = np.zeros((120, 60), np.uint8)
    for x in range(2, 53, 10):
        binary[:, x:x+6] = 255
    assert _dense_vertical_bars(binary, (0, 0, 60, 120))


def test_thin_receipt_column_rules_are_not_barcode():
    binary = np.zeros((1200, 300), np.uint8)
    for x in (2, 30, 190, 230, 295):
        binary[:, x:x+2] = 255
    for y in range(0, 1200, 40):
        binary[y:y+2] = 255
    assert not _dense_vertical_bars(binary, (0, 0, 300, 1200))


def test_one_product_table_wins_over_barcode():
    image = np.full((1280, 720, 3), 210, np.uint8)
    for x in (40, 110, 420, 520, 665):
        cv2.line(image, (x, 430), (x, 670), (15, 15, 15), 2)
    for y in (430, 505, 615, 670):
        cv2.line(image, (40, y), (665, y), (15, 15, 15), 2)
    for x in range(117, 170, 10):
        cv2.rectangle(image, (x, 740), (x+6, 848), (15, 15, 15), -1)
    boundary = detect_table_boundary(image)
    assert boundary['polygon'] is not None
    assert boundary['polygon'][:, 1].max() < 710
    assert np.ptp(boundary['polygon'][:, 0]) > 600
