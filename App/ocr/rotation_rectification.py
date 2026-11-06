"""Rotation-only rectification for a correctly located tilted table."""
from __future__ import annotations

import cv2
import numpy as np

VERSION = "rotation-crop-1.0"


def rotation_angle(polygon):
    points = np.asarray(polygon, np.float32).reshape(4, 2)
    tl, tr, br, bl = points
    angles = [np.degrees(np.arctan2(*(tr - tl)[::-1])),
              np.degrees(np.arctan2(*(br - bl)[::-1]))]
    angle = float(np.median(angles))
    if not 2. <= abs(angle) <= 12.:
        raise ValueError(f"Наклон {angle:.2f}° вне безопасного диапазона")
    return angle


def rotate_table_crop(image, polygon, margin=16):
    """Rotate the complete photo around its centre, then crop the table.

    The returned backward map points directly into the original image, so OCR
    boxes still map to the source photo and pixels are sampled only once.
    """
    angle = rotation_angle(polygon)
    height, width = image.shape[:2]
    center = np.array([width / 2., height / 2.], np.float32)
    matrix = cv2.getRotationMatrix2D(tuple(center), angle, 1.)
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    rotated_width = int(np.ceil(height * sine + width * cosine))
    rotated_height = int(np.ceil(height * cosine + width * sine))
    matrix[:, 2] += np.array([rotated_width / 2., rotated_height / 2.]) - center
    rotated_polygon = cv2.transform(np.asarray(polygon, np.float32)[None, :, :], matrix)[0]
    lo = np.maximum(0, np.floor(rotated_polygon.min(axis=0)).astype(int) - margin)
    hi = np.minimum([rotated_width, rotated_height], np.ceil(rotated_polygon.max(axis=0)).astype(int) + margin + 1)
    out_width, out_height = int(hi[0] - lo[0]), int(hi[1] - lo[1])
    yy, xx = np.indices((out_height, out_width), dtype=np.float32)
    inverse = cv2.invertAffineTransform(matrix)
    rotated_x, rotated_y = xx + float(lo[0]), yy + float(lo[1])
    map_x = inverse[0, 0] * rotated_x + inverse[0, 1] * rotated_y + inverse[0, 2]
    map_y = inverse[1, 0] * rotated_x + inverse[1, 1] * rotated_y + inverse[1, 2]
    map_x, map_y = map_x.astype(np.float32), map_y.astype(np.float32)
    output = cv2.remap(image, map_x, map_y, cv2.INTER_CUBIC,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    metadata = dict(method="rotation_crop", algorithm_version=VERSION, margin=margin,
                    rotation_opencv_deg=angle, rotation_clockwise_deg=-angle,
                    source_polygon=np.asarray(polygon, float).tolist(),
                    rotated_polygon=(rotated_polygon - lo).tolist(),
                    roi=[*lo.tolist(), *hi.tolist()], output_size=[out_width, out_height],
                    body_size=[max(16, out_width - 2 * margin), max(16, out_height - 2 * margin)],
                    warnings=[], wave_assessment={"status": "not_run", "reason": "Применён только поворот всего чека"})
    return output, map_x, map_y, metadata
