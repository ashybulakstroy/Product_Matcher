import cv2
import numpy as np

from App.ocr.contour_unwarp import build_contour_map


def test_bent_boundary_maps_to_flat_top_and_preserves_total_strip():
    polygon = np.array([[80, 100], [200, 125], [320, 100], [330, 400],
                        [230, 400], [230, 380], [70, 380]], np.float32)
    image = np.full((460, 400, 3), 255, np.uint8)
    cv2.polylines(image, [polygon[:3].astype(np.int32)], False, (0, 0, 0), 3)
    map_x, map_y, metadata = build_contour_map(polygon)
    result = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    width, height = metadata["body_size"]
    margin = metadata["margin"]
    top_line = result[margin, margin:margin+width, 0]
    assert np.mean(top_line < 50) > .97
    assert 15 < metadata["footer_height"] < 30
    assert result.shape[0] == height + metadata["footer_height"] + 2*margin
    assert metadata["min_jacobian"] > 0
    # The total remains a short extra strip, rather than stretching its step
    # over the entire image. The body bottom still maps to y=380.
    assert abs(float(map_y[margin+height-1, margin+width//2]) - 380) < 1


def test_footer_vertices_do_not_tilt_outer_sides():
    polygon = np.array([[797, 484], [57, 493], [60, 904], [69, 909],
                        [369, 912], [370, 952], [785, 969]], np.float32)
    map_x, map_y, metadata = build_contour_map(polygon, trust_top=False)
    assert 700 < metadata["body_size"][0] < 800
    assert map_x.min() > 20
    assert map_x.max() < 830
    assert map_y.min() > 450
    assert metadata["min_jacobian"] > .5


def test_missing_top_left_corner_uses_stable_extrapolation():
    polygon = np.array([[88, 552], [89, 661], [419, 665], [420, 694],
                        [424, 696], [634, 696], [632, 500], [406, 499],
                        [196, 492], [152, 493], [150, 488], [146, 488],
                        [144, 494], [143, 552]], np.float32)
    _, _, metadata = build_contour_map(polygon)
    assert metadata["source_body_corners"][0][1] < 510
    assert metadata["body_size"][1] > 150
    assert metadata["footer_height"] < 50


def test_supported_top_right_overrides_footer_pulled_side_fit():
    # The first point is the genuine printed top-right corner. The stepped
    # lower total cell pulls a plain outer-side fit 51 pixels too far right.
    polygon = np.array([[511, 370], [59, 377], [58, 883], [63, 944],
                        [203, 936], [202, 928], [188, 924], [110, 933],
                        [76, 930], [75, 921], [367, 904], [370, 934],
                        [562, 921]], np.float32)
    _, _, metadata = build_contour_map(polygon)
    refinement = metadata["top_right_refinement"]
    assert refinement["applied"]
    assert refinement["supported"] == [511.0, 370.0]
    assert refinement["shift_px"] < -45
    assert metadata["footer_height"] > 0


def test_inner_header_intersection_cannot_replace_true_top_right_corner():
    # 457/447 are internal header intersections; 534 is the outer corner.
    # The table is already correctly rectangular and must stay untouched.
    polygon = np.array([[534, 372], [457, 373], [447, 376], [122, 394],
                        [139, 631], [322, 620], [324, 641], [350, 642],
                        [550, 628]], np.float32)
    _, _, metadata = build_contour_map(polygon)
    assert not metadata["top_right_refinement"]["applied"]
