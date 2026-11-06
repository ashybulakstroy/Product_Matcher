import cv2
import numpy as np

from App.ocr.grid_line_preprocessing import enhance_vertical_grid


def test_enhance_vertical_grid_increases_vertical_structure_without_washing_text():
    image = np.full((120, 220, 3), 255, dtype=np.uint8)
    cv2.line(image, (18, 6), (18, 114), (70, 70, 70), 3)
    cv2.line(image, (38, 6), (38, 114), (70, 70, 70), 3)
    cv2.putText(image, "A", (120, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (50, 50, 50), 3, cv2.LINE_AA)

    enhanced = enhance_vertical_grid(image)

    assert enhanced.shape == image.shape
    assert np.mean(enhanced[:, 10:30]) < np.mean(image[:, 10:30])
    assert np.mean(enhanced[45:95, 115:185]) > 200
