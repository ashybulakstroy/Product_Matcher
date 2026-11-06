import cv2
import numpy as np

from App.ocr.quantity_price_split_probe import apply_gap_highlight, detect_quantity_price_gap
from App.ocr.x_profile_split_probe import find_quantity_price_split


def test_find_quantity_price_split_detects_middle_gap():
    image = np.full((120, 220, 3), 255, dtype=np.uint8)
    left = np.full((80, 45, 3), 80, dtype=np.uint8)
    right = np.full((80, 45, 3), 80, dtype=np.uint8)
    image[20:100, 20:65] = left
    image[20:100, 110:155] = right
    image[20:100, 70:100] = 255

    split = find_quantity_price_split(image)

    assert split is not None
    assert 60 <= split <= 100


def test_detect_quantity_price_gap_skips_when_no_real_gap_exists():
    image = np.full((120, 220, 3), 80, dtype=np.uint8)

    split = detect_quantity_price_gap(image)
    processed = apply_gap_highlight(image, split)

    assert split is None
    assert np.array_equal(image, processed)
